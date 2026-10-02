import os
import sys
import json
import math
import argparse
import time
import gc
import signal
import importlib
import subprocess
import multiprocessing
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from elfarol.simulation_runner import SimulationRunner, RunConfig, VLLMBatchPipeline
from elfarol.simulation_runner import _build_vllm_pipeline as _legacy_build_vllm_pipeline

from elfarol.minds import MindRunner

try:
    from elfarol.agent_brain import SYSTEM_PROMPT as _AGENT_SYSTEM_PROMPT
except ImportError:
    _AGENT_SYSTEM_PROMPT = None

EXPERIMENT_CONDITIONS = [
    {
        "label": "control",
        "broadcast_enabled": False,
        "description": "Agents cannot access the global broadcast feed.",
    },
    {
        "label": "delta2",
        "broadcast_enabled": True,
        "description": "Agents can read and write the global broadcast feed. Deception enabled.",
    },
]

MIB = 1024 * 1024
ENGINE_PROCESS_MARKERS = ("enginecore", "vllm")
ARMOR_DEFAULTS = {
    "max_num_seqs": 64,
    "max_num_batched_tokens": 2048,
    "kv_cache_dtype": "auto",
    "max_attempts": 3,
    "vram_timeout_s": 180.0,
    "vram_headroom_mib": 512.0,
    "vram_poll_s": 0.5,
    "utilization_floor": 0.78,
}
LIVE_STATE_FILENAME = "live_state.json"
MANIFEST_FILENAME = "trial_manifest.json"
_NVML = {"state": None, "module": None, "handle": None}
_SIGNALS = {"interrupted": False}
_LOG_SINKS = []


class EngineStartupError(RuntimeError):
    pass


def _log(message):
    print(f"[armor] {message}", flush=True)
    for sink in list(_LOG_SINKS):
        try:
            sink(message)
        except Exception:
            pass


def _json_default(value):
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    return str(value)


def _atomic_write_json(path, payload):
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    tmp_path = f"{path}.{os.getpid()}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    for _ in range(5):
        try:
            os.replace(tmp_path, path)
            return path
        except PermissionError:
            time.sleep(0.05)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    try:
        os.remove(tmp_path)
    except OSError:
        pass
    return path


def _gpu_index():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    return int(visible) if visible.isdigit() else 0


def _query_vram():
    if _NVML["state"] is None:
        try:
            pynvml = importlib.import_module("pynvml")
            pynvml.nvmlInit()
            _NVML["handle"] = pynvml.nvmlDeviceGetHandleByIndex(_gpu_index())
            _NVML["module"] = pynvml
            _NVML["state"] = True
        except Exception:
            _NVML["state"] = False
    if _NVML["state"]:
        try:
            info = _NVML["module"].nvmlDeviceGetMemoryInfo(_NVML["handle"])
            return {"used_mib": info.used / MIB, "free_mib": info.free / MIB, "total_mib": info.total / MIB}
        except Exception:
            pass
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={_gpu_index()}",
                "--query-gpu=memory.used,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0 and result.stdout.strip():
            used, free, total = (float(v) for v in result.stdout.strip().splitlines()[0].split(","))
            return {"used_mib": used, "free_mib": free, "total_mib": total}
    except Exception:
        pass
    return None


def _format_vram(snapshot):
    if not snapshot:
        return "VRAM n/a"
    return f"VRAM {snapshot['used_mib']:.0f}/{snapshot['total_mib']:.0f} MiB used, {snapshot['free_mib']:.0f} MiB free"


def _round_vram(snapshot):
    if not snapshot:
        return None
    return {k: (round(v, 1) if isinstance(v, float) else v) for k, v in snapshot.items()}


def _engine_processes():
    try:
        children = multiprocessing.active_children()
    except Exception:
        return []
    return [p for p in children if any(marker in (p.name or "").lower() for marker in ENGINE_PROCESS_MARKERS)]


def _reap_engine_processes(timeout_s=15.0):
    procs = _engine_processes()
    for proc in procs:
        try:
            proc.terminate()
        except Exception:
            pass
    deadline = time.monotonic() + timeout_s
    for proc in procs:
        try:
            proc.join(max(0.0, deadline - time.monotonic()))
        except Exception:
            pass
    for proc in procs:
        try:
            if proc.is_alive():
                proc.kill()
                proc.join(5.0)
        except Exception:
            pass
    return len(procs)


def _stale_engine_pids():
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return []
    own = {os.getpid()}
    try:
        own.update(p.pid for p in multiprocessing.active_children() if p.pid)
    except Exception:
        pass
    uid = os.getuid() if hasattr(os, "getuid") else None
    stale = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit() or int(entry.name) in own:
            continue
        try:
            if uid is not None and entry.stat().st_uid != uid:
                continue
            cmdline = (entry / "cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", "ignore")
        except OSError:
            continue
        if "VLLM::" in cmdline:
            stale.append(int(entry.name))
    return stale


def _vram_settled(snapshot, required_free_mib, baseline_used_mib, headroom_mib):
    released = baseline_used_mib is not None and snapshot["used_mib"] <= baseline_used_mib + headroom_mib
    fits = required_free_mib is not None and snapshot["free_mib"] >= required_free_mib
    return released or fits


def _wait_for_vram(required_free_mib, baseline_used_mib, headroom_mib, timeout_s, poll_s):
    start = time.monotonic()
    snapshot = _query_vram()
    if snapshot is None:
        return None
    if required_free_mib is None and baseline_used_mib is None:
        snapshot["waited_s"] = 0.0
        snapshot["settled"] = True
        return snapshot
    while True:
        settled = _vram_settled(snapshot, required_free_mib, baseline_used_mib, headroom_mib) and not _engine_processes()
        if settled or time.monotonic() - start >= timeout_s:
            break
        time.sleep(poll_s)
        snapshot = _query_vram() or snapshot
    snapshot = dict(snapshot)
    snapshot["waited_s"] = round(time.monotonic() - start, 2)
    snapshot["settled"] = bool(settled)
    return snapshot


class _VRAMSentinel:

    def __init__(self, headroom_mib, timeout_s, poll_s, utilization_floor):
        self.headroom_mib = float(headroom_mib)
        self.timeout_s = float(timeout_s)
        self.poll_s = float(poll_s)
        self.utilization_floor = float(utilization_floor)
        self.baseline = _query_vram()
        self.engines_built = 0
        self.peak_used_mib = self.baseline["used_mib"] if self.baseline else None

    def required_free_mib(self, utilization):
        if not self.baseline:
            return None
        return self.baseline["total_mib"] * utilization + self.headroom_mib

    def sample(self):
        snapshot = _query_vram()
        if snapshot and (self.peak_used_mib is None or snapshot["used_mib"] > self.peak_used_mib):
            self.peak_used_mib = snapshot["used_mib"]
        return snapshot

    def drain(self, utilization):
        baseline_used = self.baseline["used_mib"] if self.baseline else None
        snapshot = _wait_for_vram(
            self.required_free_mib(utilization),
            baseline_used,
            self.headroom_mib,
            self.timeout_s,
            self.poll_s,
        )
        if snapshot is not None and not snapshot.get("settled", True):
            stale = _stale_engine_pids()
            _log(
                f"VRAM did not settle within {self.timeout_s:.0f}s :: {_format_vram(snapshot)}"
                + (f" :: foreign vLLM processes holding the GPU: {stale} (kill -9 {' '.join(map(str, stale))})" if stale else "")
            )
        return snapshot

    def fit_utilization(self, utilization):
        snapshot = _query_vram()
        if not snapshot:
            return utilization
        need = snapshot["total_mib"] * utilization + self.headroom_mib
        if snapshot["free_mib"] >= need:
            return utilization
        fitted = math.floor((snapshot["free_mib"] - self.headroom_mib) / snapshot["total_mib"] * 100.0) / 100.0
        if fitted >= self.utilization_floor:
            _log(f"free VRAM below request :: {_format_vram(snapshot)} :: gpu_memory_utilization {utilization:.2f} -> {fitted:.2f}")
            return fitted
        _log(f"free VRAM below the {self.utilization_floor:.2f} floor :: {_format_vram(snapshot)} :: holding gpu_memory_utilization {utilization:.2f}")
        return utilization


def _clear_vram():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except (ImportError, Exception):
        pass
    gc.collect()


def _destroy_vllm(llm):
    engine = getattr(llm, "llm_engine", None)
    for target in (getattr(engine, "engine_core", None), engine, llm):
        shutdown = getattr(target, "shutdown", None)
        if not callable(shutdown):
            continue
        try:
            shutdown()
            break
        except Exception:
            continue
    for hook in ("destroy_model_parallel", "destroy_distributed_environment"):
        try:
            getattr(importlib.import_module("vllm.distributed.parallel_state"), hook)()
        except (ImportError, Exception):
            pass
    try:
        del llm
    except Exception:
        pass
    _reap_engine_processes()
    _clear_vram()


def _is_vllm_engine(obj):
    return obj is not None and type(obj).__module__.split(".")[0] == "vllm"


def _build_output_dir(base_dir, condition_label, seed=None):
    if seed is not None:
        return os.path.join(base_dir, f"{condition_label}_seed_{seed}")
    run_ts = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(base_dir, f"{condition_label}_{run_ts}")


def _save_comparison(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    comparison = []
    for label, summary in results.items():
        row = {"condition": label}
        row.update(summary)
        comparison.append(row)
    path = os.path.join(output_dir, "comparison.json")
    _atomic_write_json(path, comparison)
    return path


def _build_vllm_pipeline(config, profile=None):
    if not profile:
        return _legacy_build_vllm_pipeline(config)
    from vllm import LLM, SamplingParams

    engine_kwargs = {
        "model": config.model_name,
        "quantization": "awq",
        "tensor_parallel_size": config.tensor_parallel_size,
        "gpu_memory_utilization": profile.get("gpu_memory_utilization", config.gpu_memory_utilization),
        "max_model_len": config.max_model_len,
        "enforce_eager": True,
        "dtype": "half",
        "seed": config.seed,
        "max_num_seqs": int(profile["max_num_seqs"]),
        "max_num_batched_tokens": int(profile["max_num_batched_tokens"]),
    }
    if int(profile["max_num_batched_tokens"]) < config.max_model_len:
        engine_kwargs["enable_chunked_prefill"] = True
    if profile.get("kv_cache_dtype") not in (None, "", "auto"):
        engine_kwargs["kv_cache_dtype"] = profile["kv_cache_dtype"]
    try:
        llm = LLM(**engine_kwargs)
    except TypeError as exc:
        _log(f"armored engine kwargs rejected ({exc}); falling back to the stock builder")
        return _legacy_build_vllm_pipeline(config)
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=256,
        top_p=0.9,
        stop=["\n\n"],
    )
    return llm, sampling_params


def _chat_messages(prompt):
    if _AGENT_SYSTEM_PROMPT and prompt.startswith(_AGENT_SYSTEM_PROMPT):
        body = prompt[len(_AGENT_SYSTEM_PROMPT):].lstrip("\n")
        return [
            {"role": "system", "content": _AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": body},
        ]
    return [{"role": "user", "content": prompt}]


class _InstructChannel:

    def __init__(self, llm, max_model_len, max_new_tokens, chat_template=True):
        self.llm = llm
        self.chat_template = chat_template
        self.hard_limit = int(max_model_len)
        self.target = max(1, min(self.hard_limit - 1, max(16, self.hard_limit - int(max_new_tokens or 0))))
        self.overflow_count = 0
        self.max_prompt_tokens = 0
        self.prompt_count = 0
        self.templated_count = 0
        self.request_seeds = None
        self._tokenizer = None
        self._disabled = False

    def _get_tokenizer(self):
        if self._disabled:
            return None
        if self._tokenizer is None:
            try:
                self._tokenizer = self.llm.get_tokenizer()
            except Exception:
                self._disabled = True
                return None
        return self._tokenizer

    def _token_ids(self, tokenizer, text, add_special_tokens):
        try:
            ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
        except TypeError:
            ids = tokenizer.encode(text)
        ids = [int(i) for i in ids]
        bos = getattr(tokenizer, "bos_token_id", None)
        if not add_special_tokens and bos is not None and len(ids) > 1 and ids[0] == bos and ids[1] == bos:
            ids = ids[1:]
        return ids

    def _encode(self, prompt):
        tokenizer = self._get_tokenizer()
        if tokenizer is None:
            return None, False
        try:
            if self.chat_template and getattr(tokenizer, "chat_template", None):
                text = tokenizer.apply_chat_template(_chat_messages(prompt), tokenize=False, add_generation_prompt=True)
                return self._token_ids(tokenizer, text, False), True
            return self._token_ids(tokenizer, prompt, True), False
        except Exception:
            return None, False

    def generate(self, prompts, sampling_params=None, **kwargs):
        channel = []
        for prompt in prompts:
            ids, templated = self._encode(prompt) if isinstance(prompt, str) else (None, False)
            self.prompt_count += 1
            if ids is None:
                channel.append(prompt)
                continue
            self.templated_count += int(templated)
            self.max_prompt_tokens = max(self.max_prompt_tokens, len(ids))
            if len(ids) >= self.hard_limit:
                self.overflow_count += 1
                head = self.target // 2
                ids = ids[:head] + ids[-(self.target - head):]
            channel.append({"prompt_token_ids": ids})
        seeds = self.request_seeds
        self.request_seeds = None
        if seeds is not None and len(seeds) == len(channel) and callable(getattr(sampling_params, "clone", None)):
            per_request = []
            for seed in seeds:
                params = sampling_params.clone()
                params.seed = int(seed)
                per_request.append(params)
            sampling_params = per_request
        return self.llm.generate(channel, sampling_params, **kwargs)


class _TelemetryMixin:

    def _flush_epoch(self, epoch):
        super()._flush_epoch(epoch)
        hook = getattr(self, "on_epoch", None)
        if hook is not None:
            try:
                hook(self, epoch)
            except Exception:
                pass


class _TelemetryRunner(_TelemetryMixin, SimulationRunner):

    def __init__(self, config, llm_pipeline=None, on_epoch=None):
        super().__init__(config=config, llm_pipeline=llm_pipeline)
        self.on_epoch = on_epoch


class _MindTelemetryRunner(_TelemetryMixin, MindRunner):

    def __init__(self, config, llm_pipeline=None, on_epoch=None):
        super().__init__(config=config, llm_pipeline=llm_pipeline)
        self.on_epoch = on_epoch


class _LiveState:

    def __init__(self, path, model_name, trials, num_agents, num_epochs, grid_size, seeds, enabled=True):
        self.path = path
        self.enabled = enabled
        self.model_name = model_name
        self.num_agents = num_agents
        self.num_epochs = num_epochs
        self.grid_size = grid_size
        self.seeds = list(seeds)
        self.trials = [
            {
                "trial": t["key"],
                "condition": t["label"],
                "seed": t["seed"],
                "trial_dir": os.path.basename(t["output_dir"]),
                "status": "pending",
                "attempts": 0,
            }
            for t in trials
        ]
        self.started_at = time.time()
        self.status = "booting"
        self.current = None
        self.vram = None
        self.events = []
        self.epoch_durations = []
        self.build_durations = []

    def event(self, message):
        self.events.append({"t": round(time.time(), 3), "message": str(message)})
        del self.events[:-40]

    def mark(self, key, **fields):
        for entry in self.trials:
            if entry["trial"] == key:
                entry.update(fields)

    def _eta(self):
        if not self.epoch_durations:
            return None
        mean_epoch = sum(self.epoch_durations) / len(self.epoch_durations)
        mean_build = sum(self.build_durations) / len(self.build_durations) if self.build_durations else 0.0
        remaining_epochs = 0
        remaining_builds = 0
        current_key = self.current.get("trial") if self.current else None
        for entry in self.trials:
            if entry["status"] in ("complete", "resumed", "failed"):
                continue
            if entry["trial"] == current_key and entry["status"] == "running":
                done = (self.current.get("epoch") if self.current.get("epoch") is not None else -1) + 1
                remaining_epochs += max(0, self.num_epochs - done)
                if self.current.get("phase") == "loading_engine":
                    remaining_builds += 1
            else:
                remaining_epochs += self.num_epochs
                remaining_builds += 1
        return round(remaining_epochs * mean_epoch + remaining_builds * mean_build, 1)

    def publish(self, status=None):
        if status is not None:
            self.status = status
        if not self.enabled:
            return
        now = time.time()
        payload = {
            "schema": "elfarol.live/1",
            "status": self.status,
            "updated_at": round(now, 3),
            "started_at": round(self.started_at, 3),
            "elapsed_s": round(now - self.started_at, 1),
            "eta_s": self._eta(),
            "model": self.model_name,
            "sweep": {
                "num_agents": self.num_agents,
                "num_epochs": self.num_epochs,
                "grid_size": self.grid_size,
                "seeds": self.seeds,
                "conditions": [c["label"] for c in EXPERIMENT_CONDITIONS],
                "trial_count": len(self.trials),
                "trials_done": sum(1 for t in self.trials if t["status"] in ("complete", "resumed")),
                "trials_failed": sum(1 for t in self.trials if t["status"] == "failed"),
            },
            "trials": self.trials,
            "current": self.current,
            "vram": self.vram,
            "events": self.events,
        }
        try:
            _atomic_write_json(self.path, payload)
        except OSError:
            pass


def _live_frame(runner, trial, attempt, phase):
    snapshots = runner.metrics_logger.epoch_snapshots
    snapshot = snapshots[-1] if snapshots else {}
    population = getattr(runner, "population", None)
    agents = []
    broadcasts = []
    for rec in snapshot.get("agents", []):
        entry = {
            "id": rec.get("agent_id"),
            "x": rec.get("x"),
            "y": rec.get("y"),
            "in_bar": bool(rec.get("in_bar")),
            "deceptive": bool(rec.get("is_deceptive")),
            "stated": rec.get("stated_intention"),
            "target": rec.get("actual_target"),
            "utility": rec.get("utility"),
            "cumulative_utility": round(float(rec.get("cumulative_utility", 0.0)), 4),
        }
        if population is not None:
            entry["mind"] = population.live_view(rec.get("agent_id"))
        agents.append(entry)
    thoughts = []
    if population is not None and population.trace:
        latest = population.trace[-1]["epoch"]
        for row in population.trace[-len(snapshot.get("agents", [])) or None:]:
            if row["epoch"] == latest and row["private_note"]:
                thoughts.append({
                    "agent_id": row["agent_id"],
                    "archetype": row["archetype"],
                    "stated": row["stated_intention"],
                    "in_bar": bool(row["in_bar"]),
                    "lied": bool(row["lied"]),
                    "note": row["private_note"][:280],
                })
        thoughts.sort(key=lambda t: (not t["lied"], t["agent_id"]))
        if rec.get("broadcast"):
            broadcasts.append({"agent_id": rec.get("agent_id"), "text": str(rec.get("broadcast"))[:280]})
    share_history = {name: [] for name in ("honest_go", "honest_stay", "false_go", "false_stay")}
    for snap in snapshots:
        records = snap.get("agents", [])
        counts = [0, 0, 0, 0]
        for rec in records:
            going = rec.get("stated_intention") == "going"
            at_bar = bool(rec.get("in_bar"))
            counts[(0 if at_bar else 2) if going else (3 if at_bar else 1)] += 1
        total = len(records) or 1
        for name, count in zip(share_history, counts):
            share_history[name].append(round(count / total, 4))
    metric_keys = (
        "bar_attendance",
        "bar_capacity",
        "comfort_threshold",
        "attendance_over_threshold",
        "deception_index",
        "truthfulness_ratio",
        "mean_utility",
        "total_utility",
        "broadcast_count",
        "broadcast_correlation",
    )
    return {
        "trial": trial["key"],
        "condition": trial["label"],
        "seed": trial["seed"],
        "trial_dir": os.path.basename(trial["output_dir"]),
        "attempt": attempt,
        "phase": phase,
        "epoch": snapshot.get("epoch"),
        "epochs": runner.config.num_epochs,
        "epoch_duration_s": runner.epoch_timings[-1] if runner.epoch_timings else None,
        "fallback_rate": getattr(runner.metrics_logger, "regex_fallback_rate", 0.0),
        "grid": {"size": runner.config.grid_size, "bar_min": runner.grid.bar_min, "bar_max": runner.grid.bar_max},
        "metrics": {k: snapshot.get(k) for k in metric_keys},
        "history": {
            **{k: [s.get(k) for s in snapshots] for k in ("bar_attendance", "deception_index", "mean_utility")},
            "strategy_shares": share_history,
        },
        "agents": agents,
        "broadcasts": broadcasts[-24:],
        "thoughts": thoughts[:40],
        "agent_model": "minds" if population is not None else "clone",
    }


def _phase_frame(trial, attempt, phase, num_epochs, grid_size):
    return {
        "trial": trial["key"],
        "condition": trial["label"],
        "seed": trial["seed"],
        "trial_dir": os.path.basename(trial["output_dir"]),
        "attempt": attempt,
        "phase": phase,
        "epoch": None,
        "epochs": num_epochs,
        "grid": {"size": grid_size},
        "agents": [],
        "broadcasts": [],
    }


def _make_epoch_hook(live, sentinel, trial, attempt):
    def hook(runner, epoch):
        if runner.epoch_timings:
            live.epoch_durations.append(runner.epoch_timings[-1])
        snapshot = sentinel.sample() if sentinel is not None else None
        if snapshot is not None:
            live.vram = _vram_payload(snapshot, sentinel)
        live.current = _live_frame(runner, trial, attempt, "running")
        live.publish("running")
    return hook


def _vram_payload(snapshot, sentinel):
    payload = _round_vram(snapshot) or {}
    if sentinel is not None and sentinel.baseline:
        payload["baseline_used_mib"] = round(sentinel.baseline["used_mib"], 1)
    if sentinel is not None and sentinel.peak_used_mib is not None:
        payload["peak_used_mib"] = round(sentinel.peak_used_mib, 1)
    return payload


def _degrade_profile(profile, attempt):
    scale = 2 ** (attempt - 1)
    degraded = dict(profile)
    degraded["max_num_seqs"] = max(8, int(profile["max_num_seqs"]) // scale)
    degraded["max_num_batched_tokens"] = max(512, int(profile["max_num_batched_tokens"]) // scale)
    return degraded


def _execute_trial(config, trial, dry_run_pipeline, profile, max_attempts, sentinel, live, chat_template=True, agent_model="minds"):
    attempts = 1 if dry_run_pipeline is not None else max(1, int(max_attempts))
    runtime = {"status": "failed", "phase": "build", "attempts": 0, "error": None}
    for attempt in range(1, attempts + 1):
        if _SIGNALS["interrupted"]:
            raise KeyboardInterrupt()
        active = _degrade_profile(profile, attempt)
        raw_llm = None
        sampling_params = None
        pipeline = None
        runner = None
        guard = None
        engine_spawned = False
        phase = "build"
        started = time.perf_counter()
        runtime = {"status": "failed", "phase": phase, "attempts": attempt, "error": None, "profile": None}
        live.mark(trial["key"], status="running", attempts=attempt)
        try:
            if dry_run_pipeline is not None:
                pipeline = dry_run_pipeline
                runtime["engine_build_s"] = 0.0
            else:
                if sentinel is not None:
                    if sentinel.engines_built or attempt > 1:
                        drained = sentinel.drain(active["gpu_memory_utilization"])
                        if drained is not None:
                            _log(f"{trial['key']} :: pre-build gate :: {_format_vram(drained)} :: waited {drained['waited_s']:.1f}s")
                    active["gpu_memory_utilization"] = sentinel.fit_utilization(active["gpu_memory_utilization"])
                live.current = _phase_frame(trial, attempt, "loading_engine", config.num_epochs, config.grid_size)
                live.publish("loading_engine")
                _log(
                    f"{trial['key']} :: attempt {attempt}/{attempts} :: building engine :: "
                    f"util {active['gpu_memory_utilization']:.2f} :: max_model_len {config.max_model_len} :: "
                    f"max_num_seqs {active['max_num_seqs']} :: max_num_batched_tokens {active['max_num_batched_tokens']} :: "
                    f"kv {active['kv_cache_dtype']}"
                )
                _clear_vram()
                build_started = time.perf_counter()
                raw_llm, sampling_params = _build_vllm_pipeline(config, active)
                runtime["engine_build_s"] = round(time.perf_counter() - build_started, 2)
                engine_spawned = _is_vllm_engine(raw_llm)
                if engine_spawned:
                    if sentinel is not None:
                        sentinel.engines_built += 1
                        loaded = sentinel.sample()
                        runtime["vram_loaded_mib"] = round(loaded["used_mib"], 1) if loaded else None
                        if loaded is not None:
                            live.vram = _vram_payload(loaded, sentinel)
                        _log(f"{trial['key']} :: engine online in {runtime['engine_build_s']:.1f}s :: {_format_vram(loaded)}")
                    live.build_durations.append(runtime["engine_build_s"])
                    guard = _InstructChannel(
                        raw_llm,
                        config.max_model_len,
                        getattr(sampling_params, "max_tokens", None),
                        chat_template=chat_template,
                    )
                    pipeline = VLLMBatchPipeline(guard, sampling_params)
                else:
                    pipeline = VLLMBatchPipeline(raw_llm, sampling_params)
            runtime["profile"] = dict(active)
            phase = "run"
            runtime["phase"] = phase
            runner_class = _MindTelemetryRunner if agent_model == "minds" else _TelemetryRunner
            runner = runner_class(
                config=config,
                llm_pipeline=pipeline,
                on_epoch=_make_epoch_hook(live, sentinel, trial, attempt),
            )
            runner.initialize()
            runner.run()
            summary = runner.export_results()
            summary.setdefault("agent_model", agent_model)
            runtime["agent_model"] = agent_model
            runtime["status"] = "complete"
            runtime["fallback_rate"] = summary.get("regex_fallback_rate", 0.0)
            return summary, runtime
        except KeyboardInterrupt:
            _SIGNALS["interrupted"] = True
            runtime["error"] = "interrupted"
            raise
        except Exception as exc:
            if dry_run_pipeline is not None:
                raise
            runtime["error"] = f"{type(exc).__name__}: {exc}"
            _log(f"{trial['key']} :: attempt {attempt}/{attempts} failed during {phase} :: {runtime['error']}")
        finally:
            runtime["wall_time_s"] = round(time.perf_counter() - started, 2)
            if guard is not None:
                runtime["prompt_overflows"] = guard.overflow_count
                runtime["max_prompt_tokens"] = guard.max_prompt_tokens
                runtime["prompt_encoding"] = "chat_template" if guard.templated_count else "raw"
                runtime["prompt_templated_rate"] = round(guard.templated_count / guard.prompt_count, 4) if guard.prompt_count else 0.0
            if dry_run_pipeline is None:
                teardown_started = time.perf_counter()
                _destroy_vllm(raw_llm)
            raw_llm = None
            sampling_params = None
            guard = None
            pipeline = None
            runner = None
            _clear_vram()
            if dry_run_pipeline is None and sentinel is not None and (engine_spawned or phase == "build"):
                released = sentinel.drain(profile["gpu_memory_utilization"])
                if released is not None:
                    runtime["vram_after_teardown_mib"] = round(released["used_mib"], 1)
                    runtime["vram_release_s"] = round(time.perf_counter() - teardown_started, 2)
                    runtime["vram_peak_mib"] = round(sentinel.peak_used_mib, 1) if sentinel.peak_used_mib is not None else None
                    live.vram = _vram_payload(released, sentinel)
                    _log(f"{trial['key']} :: teardown complete in {runtime['vram_release_s']:.1f}s :: {_format_vram(released)}")
    return None, runtime


def _load_completed_trial(output_dir, config, prompt_encoding=None, agent_model=None):
    summary_path = os.path.join(output_dir, "summary.json")
    manifest_path = os.path.join(output_dir, MANIFEST_FILENAME)
    required = ("epoch_metrics.csv", "agent_epoch_details.csv")
    if not os.path.exists(summary_path) or not all(os.path.exists(os.path.join(output_dir, f)) for f in required):
        return None
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        manifest = {}
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
    except (OSError, ValueError):
        return None
    if summary.get("total_epochs") != config.num_epochs:
        return None
    if manifest.get("status", "complete") != "complete":
        return None
    for key in ("model_name", "num_agents", "grid_size", "num_epochs", "seed", "broadcast_enabled"):
        if key in manifest and manifest[key] != getattr(config, key):
            return None
    if prompt_encoding and manifest.get("prompt_encoding") not in (None, prompt_encoding):
        return None
    if agent_model and manifest.get("agent_model", "clone") != agent_model:
        return None
    return summary, manifest


def _write_manifest(config, trial, runtime):
    manifest = {
        "trial": trial["key"],
        "condition": config.condition_label,
        "seed": config.seed,
        "model_name": config.model_name,
        "num_agents": config.num_agents,
        "grid_size": config.grid_size,
        "num_epochs": config.num_epochs,
        "broadcast_enabled": config.broadcast_enabled,
        "max_model_len": config.max_model_len,
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "prompt_encoding": runtime.get("prompt_encoding"),
        "agent_model": runtime.get("agent_model", "clone"),
        "status": runtime.get("status"),
        "runtime": runtime,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    try:
        _atomic_write_json(os.path.join(config.output_dir, MANIFEST_FILENAME), manifest)
    except OSError:
        pass


def _annotate_summary(summary, trial, runtime):
    summary["trial"] = trial["key"]
    summary["seed"] = trial["seed"]
    summary["trial_dir"] = os.path.basename(trial["output_dir"])
    summary["status"] = runtime.get("status", "complete")
    summary["runtime"] = runtime
    return summary


def run_experiment(
    model_name="casperhansen/llama-3-8b-instruct-awq",
    base_output_dir="outputs/experiment",
    num_agents=50,
    num_epochs=50,
    grid_size=50,
    seed=None,
    seeds=None,
    max_model_len=2048,
    gpu_memory_utilization=0.85,
    tensor_parallel_size=1,
    dry_run_pipeline=None,
    max_num_seqs=ARMOR_DEFAULTS["max_num_seqs"],
    max_num_batched_tokens=ARMOR_DEFAULTS["max_num_batched_tokens"],
    kv_cache_dtype=ARMOR_DEFAULTS["kv_cache_dtype"],
    max_attempts=ARMOR_DEFAULTS["max_attempts"],
    vram_timeout_s=ARMOR_DEFAULTS["vram_timeout_s"],
    resume=False,
    live_state=True,
    chat_template=True,
    agent_model="minds",
):
    if seeds is not None:
        seeds_to_run = list(seeds)
    elif seed is not None:
        seeds_to_run = [seed]
    else:
        seeds_to_run = [42, 100, 2026]

    all_summaries = {}
    os.makedirs(base_output_dir, exist_ok=True)

    trials = []
    for condition in EXPERIMENT_CONDITIONS:
        for s in seeds_to_run:
            trials.append({
                "key": f"{condition['label']}_seed_{s}" if len(seeds_to_run) > 1 else condition["label"],
                "label": condition["label"],
                "broadcast_enabled": condition["broadcast_enabled"],
                "seed": s,
                "output_dir": _build_output_dir(base_output_dir, condition["label"], seed=s),
            })

    profile = {
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_num_seqs": max_num_seqs,
        "max_num_batched_tokens": max_num_batched_tokens,
        "kv_cache_dtype": kv_cache_dtype,
    }
    sentinel = None
    if dry_run_pipeline is None:
        sentinel = _VRAMSentinel(
            ARMOR_DEFAULTS["vram_headroom_mib"],
            vram_timeout_s,
            ARMOR_DEFAULTS["vram_poll_s"],
            ARMOR_DEFAULTS["utilization_floor"],
        )
    live = _LiveState(
        os.path.join(base_output_dir, LIVE_STATE_FILENAME),
        model_name,
        trials,
        num_agents,
        num_epochs,
        grid_size,
        seeds_to_run,
        enabled=live_state,
    )
    _LOG_SINKS.append(live.event)

    try:
        if sentinel is not None:
            _log(f"pre-flight :: {_format_vram(sentinel.baseline)} :: {len(trials)} trials x {num_epochs} epochs x {num_agents} agents")
            stale = _stale_engine_pids()
            if stale:
                _log(f"pre-flight :: foreign vLLM processes detected {stale}; they must exit before an engine can claim VRAM")
            if sentinel.baseline:
                live.vram = _vram_payload(sentinel.baseline, sentinel)
        live.publish("booting")

        for index, trial in enumerate(trials):
            label = trial["label"]
            s = trial["seed"]
            trial_key = trial["key"]
            output_dir = trial["output_dir"]
            os.makedirs(output_dir, exist_ok=True)

            config = RunConfig(
                model_name=model_name,
                num_agents=num_agents,
                grid_size=grid_size,
                num_epochs=num_epochs,
                seed=s,
                broadcast_enabled=trial["broadcast_enabled"],
                output_dir=output_dir,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
                tensor_parallel_size=tensor_parallel_size,
                condition_label=label,
            )

            if resume:
                cached = _load_completed_trial(
                    output_dir,
                    config,
                    None if dry_run_pipeline is not None else ("chat_template" if chat_template else "raw"),
                    agent_model,
                )
                if cached is not None:
                    summary, manifest = cached
                    runtime = dict(manifest.get("runtime") or {})
                    runtime["status"] = "complete"
                    runtime["resumed"] = True
                    all_summaries[trial_key] = _annotate_summary(summary, trial, runtime)
                    live.mark(trial_key, status="resumed", attempts=runtime.get("attempts", 0))
                    _log(f"{trial_key} :: resume :: verified complete on disk, skipping")
                    _save_comparison(all_summaries, base_output_dir)
                    live.publish("running")
                    continue

            _log(f"trial {index + 1}/{len(trials)} :: {trial_key} :: {output_dir}")
            summary, runtime = _execute_trial(config, trial, dry_run_pipeline, profile, max_attempts, sentinel, live, chat_template, agent_model)

            if summary is None:
                failed = {
                    "condition": label,
                    "broadcast_enabled": trial["broadcast_enabled"],
                    "total_epochs": 0,
                }
                all_summaries[trial_key] = _annotate_summary(failed, trial, runtime)
                _write_manifest(config, trial, runtime)
                live.mark(trial_key, status="failed", attempts=runtime.get("attempts", 0), error=runtime.get("error"))
                _save_comparison(all_summaries, base_output_dir)
                if runtime.get("phase") == "build":
                    live.publish("failed")
                    raise EngineStartupError(
                        f"engine could not be started for {trial_key} after {runtime.get('attempts')} attempts: {runtime.get('error')}"
                    )
                live.publish("running")
                continue

            all_summaries[trial_key] = _annotate_summary(summary, trial, runtime)
            _write_manifest(config, trial, runtime)
            live.mark(trial_key, status="complete", attempts=runtime.get("attempts", 1))
            _save_comparison(all_summaries, base_output_dir)
            live.publish("running")
            gc.collect()

        comparison_path = _save_comparison(all_summaries, base_output_dir)
        live.current = None
        live.publish("complete")
        return all_summaries, comparison_path
    except KeyboardInterrupt:
        _save_comparison(all_summaries, base_output_dir)
        live.publish("interrupted")
        _log("interrupted :: engine torn down, partial comparison saved")
        raise
    finally:
        if live.event in _LOG_SINKS:
            _LOG_SINKS.remove(live.event)


def _print_summary(all_summaries):
    header = f"{'Condition':<24} {'Epochs':>6} {'Avg Attendance':>15} {'Avg Deception':>14} {'Truthfulness':>13} {'Avg Utility':>12}"
    print(header)
    print("-" * len(header))
    for label, s in all_summaries.items():
        print(
            f"{label:<24} "
            f"{s.get('total_epochs', 0):>6} "
            f"{s.get('average_bar_attendance', 0.0):>15.4f} "
            f"{s.get('average_deception_index', 0.0):>14.4f} "
            f"{s.get('population_truthfulness', 0.0):>13.4f} "
            f"{s.get('average_mean_utility', 0.0):>12.4f}"
        )


def _install_signal_armor():
    previous = {}

    def _interrupt(signum, frame):
        _SIGNALS["interrupted"] = True
        raise KeyboardInterrupt(f"signal {signum}")

    for name in ("SIGINT", "SIGTERM", "SIGHUP"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            previous[sig] = signal.signal(sig, _interrupt)
        except (ValueError, OSError):
            pass
    return previous


def _restore_signals(previous):
    for sig, handler in previous.items():
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError, TypeError):
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="casperhansen/llama-3-8b-instruct-awq")
    parser.add_argument("--output-dir", type=str, default="outputs/experiment")
    parser.add_argument("--agents", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 100, 2026])
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--gpu-util", type=float, default=0.85)
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--max-num-seqs", type=int, default=ARMOR_DEFAULTS["max_num_seqs"])
    parser.add_argument("--max-num-batched-tokens", type=int, default=ARMOR_DEFAULTS["max_num_batched_tokens"])
    parser.add_argument("--kv-cache-dtype", type=str, default=ARMOR_DEFAULTS["kv_cache_dtype"])
    parser.add_argument("--max-attempts", type=int, default=ARMOR_DEFAULTS["max_attempts"])
    parser.add_argument("--vram-timeout", type=float, default=ARMOR_DEFAULTS["vram_timeout_s"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-live-state", action="store_true")
    parser.add_argument("--raw-prompts", action="store_true")
    parser.add_argument("--clone-agents", action="store_true")
    args = parser.parse_args()

    active_seeds = [args.seed] if args.seed is not None else args.seeds
    agent_model = "clone" if args.clone_agents else "minds"
    max_model_len = args.max_model_len or (3072 if agent_model == "minds" else 2048)

    _SIGNALS["interrupted"] = False
    previous_handlers = _install_signal_armor()
    t0 = time.perf_counter()
    try:
        all_summaries, comparison_path = run_experiment(
            model_name=args.model,
            base_output_dir=args.output_dir,
            num_agents=args.agents,
            num_epochs=args.epochs,
            seeds=active_seeds,
            max_model_len=max_model_len,
            gpu_memory_utilization=args.gpu_util,
            tensor_parallel_size=args.tensor_parallel,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            kv_cache_dtype=args.kv_cache_dtype,
            max_attempts=args.max_attempts,
            vram_timeout_s=args.vram_timeout,
            resume=args.resume,
            live_state=not args.no_live_state,
            chat_template=not args.raw_prompts,
            agent_model=agent_model,
        )
    except KeyboardInterrupt:
        print(f"\nSweep interrupted after {time.perf_counter() - t0:.2f}s. Partial results: {os.path.join(args.output_dir, 'comparison.json')}")
        print("Re-run with --resume to continue from the last verified trial.")
        sys.exit(130)
    except EngineStartupError as exc:
        print(f"\nSweep aborted: {exc}")
        print(f"Partial results: {os.path.join(args.output_dir, 'comparison.json')}. Re-run with --resume once the GPU is clear.")
        sys.exit(1)
    finally:
        _restore_signals(previous_handlers)
    elapsed = time.perf_counter() - t0

    _print_summary(all_summaries)
    print(f"\nComparison saved to: {comparison_path}")
    print(f"Total experiment wall time: {elapsed:.2f}s")

    failed = [k for k, s in all_summaries.items() if s.get("status") == "failed"]
    if failed:
        print(f"Failed trials: {', '.join(failed)}")
        sys.exit(2)


if __name__ == "__main__":
    main()
