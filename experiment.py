import os
import sys
import csv
import json
import math
import random
import argparse
import time
import gc
import signal
import importlib
import subprocess

from elfarol.simulation_runner import SimulationRunner, RunConfig, _strip_broadcasts_from_actions
from elfarol.minds import MindRunner, MINDS_FILENAME, TRACE_FIELDS, TRACE_FILENAME, resolve_targets
from elfarol.action import execute_brain_actions
from elfarol.metrics_logger import (
    COMFORT_THRESHOLD_RATIO,
    UTILITY_AT_BAR_COMFORTABLE,
    UTILITY_AT_BAR_OVERCROWDED,
    UTILITY_AT_HOME,
)

try:
    import torch
except ImportError:
    torch = None

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
ARMOR_DEFAULTS = {
    "max_attempts": 2,
    "hidden": 32,
    "memory": 256,
    "batch_size": 16,
    "updates": 4,
    "target_every": 8,
    "history": 8,
}
LIVE_STATE_FILENAME = "live_state.json"
MANIFEST_FILENAME = "trial_manifest.json"
BRAINS_FILENAME = "brains.pt"
NEURAL_MODEL = "isolated torch brains :: speaker + actor DQN per agent"
OPTIONS = ("honest_go", "honest_stay", "false_go", "false_stay")
CLAIMS = ("staying", "going")
ACTIONS = ("home", "bar")
EXTRA_FEATURES = 5
ACT_EXTRA = 2
SPEAKER_KEYS = ("s_w1", "s_b1", "s_w2", "s_b2", "s_wq", "s_bq")
ACTOR_KEYS = ("a_w1", "a_b1", "a_w2", "a_b2", "a_wq", "a_bq", "a_wf", "a_bf", "a_h")
PARAM_KEYS = SPEAKER_KEYS + ACTOR_KEYS
EPOCH_FIELDS = [
    "epoch", "bar_attendance", "bar_capacity", "comfort_threshold",
    "attendance_over_threshold", "total_utility", "mean_utility",
    "deception_index", "truthfulness_ratio", "deception_count",
    "truthful_count", "broadcast_count", "broadcast_correlation",
    "town_broadcast_ratio", "actual_attendance", "intended_attendance",
]


def observation_dim(history):
    return int(history) * 2 + len(OPTIONS) + EXTRA_FEATURES


def actor_dim(history):
    return observation_dim(history) + ACT_EXTRA
CHOICE_TEXT = {
    "honest_go": "going, and saying so",
    "honest_stay": "staying home, and saying so",
    "false_go": "claiming I'm going, staying home",
    "false_stay": "claiming I'm staying, going anyway",
}
SPEECH = {
    "going": ("Heading to El Farol tonight.", "I'll be at the bar tonight.", "Going out tonight."),
    "staying": ("Staying in tonight.", "Skipping the bar tonight.", "Quiet night at home for me."),
}
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


class _VRAMSentinel:

    def __init__(self):
        self.baseline = _query_vram()
        self.peak_used_mib = self.baseline["used_mib"] if self.baseline else None

    def sample(self):
        snapshot = _query_vram()
        if snapshot and (self.peak_used_mib is None or snapshot["used_mib"] > self.peak_used_mib):
            self.peak_used_mib = snapshot["used_mib"]
        return snapshot


def _clear_vram():
    gc.collect()
    try:
        import torch as _torch
        if _torch.cuda.is_available() and _torch.cuda.is_initialized():
            _torch.cuda.synchronize()
            _torch.cuda.empty_cache()
            _torch.cuda.ipc_collect()
    except (ImportError, Exception):
        pass
    gc.collect()


def _resolve_device(device):
    if torch is None:
        raise EngineStartupError("PyTorch is not installed in this interpreter")
    if device in (None, "", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise EngineStartupError(f"device {device} requested but CUDA is unavailable")
    return str(device)


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


def _check_stop(stop_file):
    if stop_file and os.path.exists(stop_file):
        _SIGNALS["interrupted"] = True
        _log(f"stop requested via {stop_file}")
        raise KeyboardInterrupt("stop requested")


class AgentMemory:

    def __init__(self, capacity, history, device):
        self.capacity = int(capacity)
        self.history = int(history)
        self.input_dim = observation_dim(self.history)
        self.actor_dim = actor_dim(self.history)
        self.act_states = torch.zeros(self.capacity, self.actor_dim, device=device)
        self.act_rewards = torch.zeros(self.capacity, len(ACTIONS), device=device)
        self.act_next = torch.zeros(self.capacity, self.input_dim, device=device)
        self.act_attendance = torch.zeros(self.capacity, device=device)
        self.speak_states = torch.zeros(self.capacity, self.input_dim, device=device)
        self.speak_claims = torch.zeros(self.capacity, dtype=torch.long, device=device)
        self.speak_returns = torch.zeros(self.capacity, device=device)
        self.speak_next = torch.zeros(self.capacity, self.input_dim, device=device)
        self.attendance_history = torch.zeros(self.history, device=device)
        self.reward_history = torch.zeros(self.history, device=device)
        self.last_strategy = torch.zeros(len(OPTIONS), device=device)
        self.last_ratio = 0.0
        self.claim_reliability = 0.0
        self.claims_made = 0
        self.claims_true = 0
        self.act_size = 0
        self.act_cursor = 0
        self.speak_size = 0
        self.speak_cursor = 0
        self.pending_speech = None

    def observation(self, threshold_ratio, channel):
        truth = (self.claims_true + 1.0) / (self.claims_made + 2.0) if channel else 0.0
        extras = torch.tensor(
            [threshold_ratio, self.last_ratio * channel, self.claim_reliability * channel, truth, float(channel)],
            device=self.act_states.device,
        )
        return torch.cat([self.attendance_history, self.reward_history, self.last_strategy, extras])

    def remember(self, attendance_fraction, reward, strategy_index, ratio, channel):
        self.attendance_history = torch.roll(self.attendance_history, 1)
        self.attendance_history[0] = attendance_fraction
        self.reward_history = torch.roll(self.reward_history, 1)
        self.reward_history[0] = reward
        self.last_strategy.zero_()
        self.last_strategy[strategy_index] = 1.0
        if channel:
            self.last_ratio = ratio
            self.claim_reliability = 0.7 * self.claim_reliability + 0.3 * (1.0 - abs(ratio - attendance_fraction))

    def push_act(self, state, rewards, next_state, attendance_fraction):
        i = self.act_cursor
        self.act_states[i] = state
        self.act_rewards[i] = rewards
        self.act_next[i] = next_state
        self.act_attendance[i] = attendance_fraction
        self.act_cursor = (self.act_cursor + 1) % self.capacity
        self.act_size = min(self.act_size + 1, self.capacity)

    def push_speech(self, state, claim, two_night_return, state_after_next):
        i = self.speak_cursor
        self.speak_states[i] = state
        self.speak_claims[i] = int(claim)
        self.speak_returns[i] = two_night_return
        self.speak_next[i] = state_after_next
        self.speak_cursor = (self.speak_cursor + 1) % self.capacity
        self.speak_size = min(self.speak_size + 1, self.capacity)


def _mlp(p, prefix, x):
    h = torch.relu(torch.baddbmm(p[prefix + "b1"].unsqueeze(1), x, p[prefix + "w1"].transpose(1, 2)))
    return torch.relu(torch.baddbmm(p[prefix + "b2"].unsqueeze(1), h, p[prefix + "w2"].transpose(1, 2)))


def speak_forward(p, s):
    h = _mlp(p, "s_", s)
    return torch.baddbmm(p["s_bq"].unsqueeze(1), h, p["s_wq"].transpose(1, 2))


def act_forward(p, x):
    h = _mlp(p, "a_", x)
    q = torch.baddbmm(p["a_bq"].unsqueeze(1), h, p["a_wq"].transpose(1, 2))
    claim = x[..., -1:]
    q = q + p["a_h"].view(-1, 1, 1) * torch.cat([1.0 - claim, claim], dim=-1)
    f = torch.sigmoid(torch.baddbmm(p["a_bf"].unsqueeze(1), h, p["a_wf"].transpose(1, 2))).squeeze(-1)
    return q, f


class AgentBrain:

    def __init__(self, agent_id, seed, input_dim, hidden, device):
        self.agent_id = agent_id
        self.input_dim = int(input_dim)
        self.actor_dim = self.input_dim + ACT_EXTRA
        self.hidden = int(hidden)
        rng = random.Random(f"brain:{seed}:{agent_id}")
        self.temperament = {
            "honesty": round(rng.betavariate(2.2, 2.2), 3),
            "risk_tolerance": round(rng.betavariate(2.2, 2.2), 3),
            "competitiveness": round(rng.betavariate(2.2, 2.2), 3),
            "learning_rate": round(10 ** rng.uniform(-3.2, -2.2), 5),
            "impulsiveness": round(rng.uniform(0.25, 1.1), 3),
            "exploration": round(rng.uniform(0.03, 0.15), 3),
            "discount": round(rng.uniform(0.3, 0.8), 3),
        }
        generator = torch.Generator().manual_seed(rng.getrandbits(31))

        def layer(fan_in, fan_out):
            bound = math.sqrt(6.0 / (fan_in + fan_out))
            weight = (torch.rand(fan_out, fan_in, generator=generator) * 2.0 - 1.0) * bound
            return weight, torch.zeros(fan_out)

        tensors = {}
        tensors["s_w1"], tensors["s_b1"] = layer(self.input_dim, self.hidden)
        tensors["s_w2"], tensors["s_b2"] = layer(self.hidden, self.hidden)
        tensors["s_wq"], tensors["s_bq"] = layer(self.hidden, len(CLAIMS))
        tensors["a_w1"], tensors["a_b1"] = layer(self.actor_dim, self.hidden)
        tensors["a_w2"], tensors["a_b2"] = layer(self.hidden, self.hidden)
        tensors["a_wq"], tensors["a_bq"] = layer(self.hidden, len(ACTIONS))
        tensors["a_wf"], tensors["a_bf"] = layer(self.hidden, 1)
        tensors["a_h"] = torch.tensor([(self.temperament["honesty"] - 0.5) * 2.0])
        tensors["a_bq"][1] += (self.temperament["risk_tolerance"] - 0.5)
        tensors["s_bq"][1] += (self.temperament["competitiveness"] - 0.5)
        self.params = {k: tensors[k].to(device).requires_grad_(True) for k in PARAM_KEYS}
        self.target = {k: v.detach().clone() for k, v in self.params.items()}
        self.optimizer = torch.optim.Adam(list(self.params.values()), lr=self.temperament["learning_rate"])
        self.memory = None
        self.updates = 0
        self.last_loss = None

    def parameter_count(self):
        return int(sum(p.numel() for p in self.params.values()))

    def _single(self, params=None):
        return {k: v.unsqueeze(0) for k, v in (params or self.params).items()}

    def speak(self, s, params=None):
        return speak_forward(self._single(params), s.unsqueeze(0)).squeeze(0)

    def act(self, x, params=None):
        q, f = act_forward(self._single(params), x.unsqueeze(0))
        return q.squeeze(0), f.squeeze(0)

    def sync_target(self):
        with torch.no_grad():
            for key, value in self.params.items():
                self.target[key].copy_(value)

    def state_dict(self):
        return {k: v.detach().cpu().clone() for k, v in self.params.items()}


class BrainColony:

    def __init__(self, seed, agent_ids, num_agents, threshold, broadcast_enabled, device="cpu", hidden=None, memory=None, batch_size=None, updates=None, target_every=None, history=None):
        self.device = torch.device(device)
        self.num_agents = num_agents
        self.threshold = threshold
        self.broadcast_enabled = broadcast_enabled
        self.channel = 1 if broadcast_enabled else 0
        self.hidden = int(hidden or ARMOR_DEFAULTS["hidden"])
        self.batch_size = int(batch_size or ARMOR_DEFAULTS["batch_size"])
        self.updates = int(updates or ARMOR_DEFAULTS["updates"])
        self.target_every = int(target_every or ARMOR_DEFAULTS["target_every"])
        capacity = int(memory or ARMOR_DEFAULTS["memory"])
        self.history = int(history or ARMOR_DEFAULTS["history"])
        self.input_dim = observation_dim(self.history)
        self.actor_dim = actor_dim(self.history)
        self.ids = sorted(agent_ids)
        self.index = {aid: i for i, aid in enumerate(self.ids)}
        self.brains = []
        for aid in self.ids:
            brain = AgentBrain(aid, seed, self.input_dim, self.hidden, self.device)
            brain.memory = AgentMemory(capacity, self.history, self.device)
            self.brains.append(brain)
        self.generator = torch.Generator(device=self.device).manual_seed(int(seed) * 7919 + 17)
        self.speech_rng = random.Random(f"speech:{seed}")
        n = len(self.brains)
        temper = [b.temperament for b in self.brains]
        self.temperature = torch.tensor([t["impulsiveness"] for t in temper], device=self.device)
        self.exploration = torch.tensor([t["exploration"] for t in temper], device=self.device)
        self.discount = torch.tensor([t["discount"] for t in temper], device=self.device)
        self.threshold_ratio = threshold / max(1, num_agents)
        self.pending = None
        self.notes = [None] * n
        self.archetypes = ["PRAGMATIST"] * n
        self.moods = [("calm", 0.0)] * n
        self.last_probs = [dict.fromkeys(OPTIONS, 0.25) for _ in range(n)]
        self.last_forecast = [0.0] * n
        self.forecast_error = [0.25] * n
        self.private_lies = [0] * n
        self.decisions = [0] * n
        self.manipulation_wins = [0] * n
        self.bluffed_last_night = [False] * n
        self.town_broadcast_ratio = None
        self.losses = []
        self.trace = []

    def _stack(self, target=False):
        source = "target" if target else "params"
        return {k: torch.stack([getattr(b, source)[k] for b in self.brains]) for k in PARAM_KEYS}

    def _policy(self, q, epoch):
        decay = max(0.15, 0.97 ** epoch)
        tau = (self.temperature * decay).clamp(min=0.05).unsqueeze(1)
        probs = torch.softmax(q / tau, dim=1)
        eps = (self.exploration * decay).unsqueeze(1)
        probs = (1.0 - eps) * probs + eps / q.shape[1]
        return probs, torch.multinomial(probs, 1, generator=self.generator).squeeze(1)

    def broadcast_phase(self, agents, epoch):
        with torch.no_grad():
            s = torch.stack([b.memory.observation(self.threshold_ratio, self.channel) for b in self.brains])
            q_s = speak_forward(self._stack(), s.unsqueeze(1)).squeeze(1)
            probs, claims = self._policy(q_s, epoch)
        self.pending = {"s": s, "claims": claims, "speak_probs": probs, "speak_q": q_s}
        return [int(claims[self.index[a.id]].item()) for a in agents]

    def action_phase(self, agents, epoch, town_broadcast_ratio):
        self.town_broadcast_ratio = float(town_broadcast_ratio)
        pending = self.pending
        n = len(self.brains)
        with torch.no_grad():
            ratio = torch.full((n, 1), self.town_broadcast_ratio * self.channel, device=self.device)
            claims = pending["claims"].float().unsqueeze(1)
            x = torch.cat([pending["s"], ratio, claims], dim=1)
            q_a, f = act_forward(self._stack(), x.unsqueeze(1))
            q_a = q_a.squeeze(1)
            probs, moves = self._policy(q_a, epoch)
        pending.update({"x": x, "moves": moves, "act_probs": probs, "act_q": q_a, "f": f.squeeze(1)})
        claim_list = pending["claims"].tolist()
        move_list = moves.tolist()
        actions = []
        for agent in agents:
            i = self.index[agent.id]
            stated = "going" if claim_list[i] else "staying"
            actions.append({
                "move": [agent.x, agent.y],
                "broadcast": self.speech_rng.choice(SPEECH[stated]) if self.broadcast_enabled else None,
                "proximity_speech": None,
                "stated_intention": stated,
                "actual_target": "bar" if move_list[i] else "home",
            })
        return actions

    def decide(self, agents, epoch):
        intents = self.broadcast_phase(agents, epoch)
        return self.action_phase(agents, epoch, sum(intents) / max(1, len(intents)))

    def _sample(self, sizes, n):
        return (torch.rand(n, self.batch_size, generator=self.generator, device=self.device) * sizes.clamp(min=1).unsqueeze(1)).long()

    def _train(self):
        n = len(self.brains)
        act_sizes = torch.tensor([b.memory.act_size for b in self.brains], device=self.device, dtype=torch.float32)
        speak_sizes = torch.tensor([b.memory.speak_size for b in self.brains], device=self.device, dtype=torch.float32)
        act_mask = (act_sizes > 0).float()
        speak_mask = (speak_sizes > 0).float()
        active = ((act_mask + speak_mask) > 0).tolist()
        if not any(active):
            return None
        rows = torch.arange(n, device=self.device).unsqueeze(1)
        gamma = self.discount.view(n, 1)
        last = None
        for _ in range(self.updates):
            mems = [b.memory for b in self.brains]
            ia = self._sample(act_sizes, n)
            x = torch.stack([m.act_states for m in mems])[rows, ia]
            r = torch.stack([m.act_rewards for m in mems])[rows, ia]
            s_next = torch.stack([m.act_next for m in mems])[rows, ia]
            att = torch.stack([m.act_attendance for m in mems])[rows, ia]
            js = self._sample(speak_sizes, n)
            s = torch.stack([m.speak_states for m in mems])[rows, js]
            c = torch.stack([m.speak_claims for m in mems])[rows, js]
            g2 = torch.stack([m.speak_returns for m in mems])[rows, js]
            s_after = torch.stack([m.speak_next for m in mems])[rows, js]
            online = self._stack()
            q_a, f = act_forward(online, x)
            q_s = speak_forward(online, s).gather(2, c.unsqueeze(2)).squeeze(2)
            with torch.no_grad():
                frozen = self._stack(target=True)
                v_next = speak_forward(frozen, s_next).max(dim=2).values
                act_target = r + (gamma * v_next).unsqueeze(2)
                v_after = speak_forward(frozen, s_after).max(dim=2).values
                speak_target = g2 + gamma * gamma * v_after
            act_loss = torch.nn.functional.smooth_l1_loss(q_a, act_target, reduction="none").mean(dim=(1, 2))
            fit = torch.nn.functional.mse_loss(f, att, reduction="none").mean(dim=1)
            speak_loss = torch.nn.functional.smooth_l1_loss(q_s, speak_target, reduction="none").mean(dim=1)
            per_agent = (act_loss + 0.5 * fit) * act_mask + speak_loss * speak_mask
            for brain in self.brains:
                brain.optimizer.zero_grad(set_to_none=True)
            per_agent.sum().backward()
            losses = per_agent.detach().cpu().tolist()
            for brain, live, loss in zip(self.brains, active, losses):
                if not live:
                    continue
                torch.nn.utils.clip_grad_norm_(list(brain.params.values()), 5.0)
                brain.optimizer.step()
                brain.updates += 1
                brain.last_loss = loss
            last = losses
        return last

    def _archetype(self, probs, honesty):
        lie = probs["false_go"] + probs["false_stay"]
        go = probs["honest_go"] + probs["false_stay"]
        if lie >= 0.5:
            return "MACHIAVELLIAN"
        if honesty > 0.72 and lie < 0.2:
            return "STRAIGHT SHOOTER"
        if go >= 0.65:
            return "GAMBLER"
        if go <= 0.35:
            return "CAUTIOUS"
        return "PRAGMATIST"

    def observe(self, agents, actions, epoch, utilities, rerouted):
        n_total = max(1, self.num_agents)
        attendance = sum(1 for a in agents if a.in_bar_flag)
        fraction = attendance / n_total
        ratio = self.town_broadcast_ratio if self.town_broadcast_ratio is not None else 0.0
        pending = self.pending
        speak_probs = pending["speak_probs"].cpu().tolist()
        act_probs = pending["act_probs"].cpu().tolist()
        act_q = pending["act_q"].cpu().tolist()
        forecasts = pending["f"].cpu().tolist()
        rows = []
        for agent, action, utility, moved in zip(agents, actions, utilities, rerouted):
            i = self.index[agent.id]
            brain = self.brains[i]
            memory = brain.memory
            in_bar = bool(agent.in_bar_flag)
            stated_go = action.get("stated_intention") == "going"
            would_be = attendance - int(in_bar) + 1
            go_reward = UTILITY_AT_BAR_COMFORTABLE if would_be <= self.threshold else UTILITY_AT_BAR_OVERCROWDED
            realised = OPTIONS.index(("honest_go" if in_bar else "false_go") if stated_go else ("false_stay" if in_bar else "honest_stay"))
            state = pending["s"][i]
            memory.remember(fraction, float(utility), realised, ratio, self.channel)
            lied = stated_go != in_bar
            if self.channel:
                memory.claims_made += 1
                memory.claims_true += int(not lied)
            next_state = memory.observation(self.threshold_ratio, self.channel)
            memory.push_act(pending["x"][i], torch.tensor([UTILITY_AT_HOME, go_reward], device=self.device), next_state, fraction)
            if memory.pending_speech is not None:
                prev_state, prev_claim, prev_reward = memory.pending_speech
                memory.push_speech(prev_state, prev_claim, prev_reward + brain.temperament["discount"] * float(utility), next_state)
            memory.pending_speech = (state, int(stated_go), float(utility))
            if self.bluffed_last_night[i] and in_bar and float(utility) > 0:
                self.manipulation_wins[i] += 1
            self.bluffed_last_night[i] = stated_go and not in_bar
            self.decisions[i] += 1
            self.private_lies[i] += int(lied)
            s_go = speak_probs[i][1]
            a_go = act_probs[i][1]
            probs = {
                "honest_go": s_go * a_go,
                "honest_stay": (1.0 - s_go) * (1.0 - a_go),
                "false_go": s_go * (1.0 - a_go),
                "false_stay": (1.0 - s_go) * a_go,
            }
            q = act_q[i]
            forecast = forecasts[i] * n_total
            self.last_probs[i] = probs
            self.last_forecast[i] = forecast
            self.forecast_error[i] = 0.8 * self.forecast_error[i] + 0.2 * abs(forecasts[i] - fraction)
            self.archetypes[i] = self._archetype(probs, brain.temperament["honesty"])
            surprise = min(1.0, abs(float(utility) - max(q[int(in_bar)], -1.0)) / 2.0)
            if in_bar and float(utility) < 0:
                mood = "frustration"
            elif in_bar:
                mood = "pride"
            elif go_reward > UTILITY_AT_HOME:
                mood = "envy"
            elif attendance > self.threshold:
                mood = "hope"
            else:
                mood = "calm"
            self.moods[i] = (mood, round(surprise, 3) if mood != "calm" else 0.0)
            self.notes[i] = (
                f"Told the town {'going' if stated_go else 'staying'}; town says {ratio:.0%} going. "
                f"Forecast {forecast:.0f} vs line {self.threshold}; Q go {q[1]:+.2f} / stay {q[0]:+.2f}; {CHOICE_TEXT[OPTIONS[realised]]}."
            )
            rows.append((agent, action, utility, moved, i, probs, forecast, lied, in_bar))
        self._train_and_trace(rows, epoch)

    def _train_and_trace(self, rows, epoch):
        losses = self._train()
        if losses is not None:
            self.losses.append(sum(losses) / len(losses))
        if (epoch + 1) % self.target_every == 0:
            for brain in self.brains:
                brain.sync_target()
        for agent, action, utility, moved, i, probs, forecast, lied, in_bar in rows:
            memory = self.brains[i].memory
            emotion, intensity = self.moods[i]
            self.trace.append({
                "epoch": epoch,
                "agent_id": agent.id,
                "archetype": self.archetypes[i],
                "forecast": round(forecast, 2),
                "forecast_confidence": round(max(0.0, 1.0 - 2.0 * self.forecast_error[i]), 3),
                "active_predictor": "neural_forecaster",
                "p_honest_go": round(probs["honest_go"], 4),
                "p_honest_stay": round(probs["honest_stay"], 4),
                "p_false_go": round(probs["false_go"], 4),
                "p_false_stay": round(probs["false_stay"], 4),
                "instinct": max(probs, key=probs.get),
                "stated_intention": action.get("stated_intention", "staying"),
                "actual_target": action.get("actual_target", "home"),
                "in_bar": in_bar,
                "lied": lied,
                "rerouted": moved,
                "utility": utility,
                "top_emotion": emotion,
                "top_emotion_intensity": intensity,
                "trust_given_mean": round(memory.claim_reliability, 4) if self.broadcast_enabled else 0.5,
                "reputation": round(self._reputation(i), 4),
                "private_note": self.notes[i] or "",
            })

    def _reputation(self, i):
        memory = self.brains[i].memory
        if not self.broadcast_enabled:
            return 0.5
        return (memory.claims_true + 1.0) / (memory.claims_made + 2.0)

    def live_view(self, agent_id):
        i = self.index.get(agent_id)
        if i is None:
            return None
        brain = self.brains[i]
        emotion, intensity = self.moods[i]
        return {
            "archetype": self.archetypes[i],
            "traits": brain.temperament,
            "forecast": round(self.last_forecast[i], 1),
            "forecast_confidence": round(max(0.0, 1.0 - 2.0 * self.forecast_error[i]), 3),
            "predictor": "neural forecaster",
            "instinct": {k: round(v, 3) for k, v in self.last_probs[i].items()},
            "mood": emotion,
            "mood_intensity": intensity,
            "reputation": round(self._reputation(i), 3),
            "lies": self.private_lies[i],
            "note": self.notes[i],
        }

    def export(self):
        out = {}
        for i, brain in enumerate(self.brains):
            memory = brain.memory
            out[str(brain.agent_id)] = {
                "archetype": self.archetypes[i],
                "traits": brain.temperament,
                "predictors": ["neural_forecaster"],
                "predictor_error": {"neural_forecaster": round(self.forecast_error[i] * self.num_agents, 3)},
                "beliefs": {k: round(v, 4) for k, v in self.last_probs[i].items()},
                "emotions": {self.moods[i][0]: self.moods[i][1]},
                "reputation": round(self._reputation(i), 4),
                "public_claims": memory.claims_made,
                "public_lies": memory.claims_made - memory.claims_true,
                "private_lies": self.private_lies[i],
                "decisions": self.decisions[i],
                "manipulation_wins": self.manipulation_wins[i],
                "least_trusted": [],
                "most_trusted": [],
                "last_note": self.notes[i],
                "brain": {
                    "hidden": self.hidden,
                    "input_dim": brain.input_dim,
                    "actor_dim": brain.actor_dim,
                    "parameters": brain.parameter_count(),
                    "honesty_coupling": round(float(brain.params["a_h"].detach().cpu()[0]), 4),
                    "updates": brain.updates,
                    "act_memory": memory.act_size,
                    "speech_memory": memory.speak_size,
                    "last_loss": round(brain.last_loss, 5) if brain.last_loss is not None else None,
                },
            }
        return out

    def state_dicts(self):
        return {str(b.agent_id): {"temperament": b.temperament, "params": b.state_dict()} for b in self.brains}


class NeuralRunner(SimulationRunner):

    def __init__(self, config, device="cpu", pace_s=0.0, brain_options=None):
        super().__init__(config=config, llm_pipeline=None)
        self.device = device
        self.pace_s = max(0.0, float(pace_s or 0.0))
        self.brain_options = dict(brain_options or {})
        self.population = None

    def initialize(self):
        super().initialize()
        capacity = (self.grid.bar_max - self.grid.bar_min) ** 2
        self.population = BrainColony(
            self.config.seed,
            [a.id for a in self.agent_pool.all_agents()],
            self.config.num_agents,
            int(COMFORT_THRESHOLD_RATIO * capacity),
            self.config.broadcast_enabled,
            device=self.device,
            **self.brain_options,
        )

    def _step(self, epoch):
        started = time.perf_counter()
        agents = self.agent_pool.all_agents()
        intents = self.population.broadcast_phase(agents, epoch)
        town_broadcast_ratio = sum(intents) / max(1, len(intents))
        brain_actions = self.population.action_phase(agents, epoch, town_broadcast_ratio)
        intended = sum(1 for action in brain_actions if action["actual_target"] == "bar")
        if not self.config.broadcast_enabled:
            brain_actions = _strip_broadcasts_from_actions(brain_actions)
        brain_actions, rerouted = resolve_targets(agents, brain_actions, self.grid, random.Random(f"resolver:{self.config.seed}:{epoch}"))
        execute_brain_actions(agents, brain_actions, self.grid, self.agent_pool, rng=self.rng)
        snapshot = self.metrics_logger.record_epoch(epoch, agents, self.grid, brain_actions)
        snapshot["town_broadcast_ratio"] = round(town_broadcast_ratio, 4)
        snapshot["actual_attendance"] = snapshot.get("bar_attendance")
        snapshot["intended_attendance"] = intended
        self.metrics_logger.record_fallbacks(0, len(agents))
        if self.config.broadcast_enabled:
            for action in brain_actions:
                if action.get("broadcast"):
                    self.broadcast_history.append(action["broadcast"])
        records = {rec["agent_id"]: rec for rec in snapshot.get("agents", [])}
        for agent in agents:
            agent.in_bar_flag = bool(records.get(agent.id, {}).get("in_bar", self.grid.is_in_bar(agent.x, agent.y)))
        utilities = [records.get(a.id, {}).get("utility", 0.0) for a in agents]
        self.population.observe(agents, brain_actions, epoch, utilities, rerouted)
        self.agent_pool.clear_messages()
        remaining = self.pace_s - (time.perf_counter() - started)
        if remaining > 0:
            time.sleep(remaining)

    def _export_epoch_csv(self):
        path = os.path.join(self.config.output_dir, "epoch_metrics.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=EPOCH_FIELDS)
            writer.writeheader()
            for snap in self.metrics_logger.epoch_snapshots:
                writer.writerow({k: snap.get(k, "") for k in EPOCH_FIELDS})

    def export_results(self):
        summary = super().export_results()
        self._export_epoch_csv()
        with open(os.path.join(self.config.output_dir, MINDS_FILENAME), "w", encoding="utf-8") as f:
            json.dump(self.population.export(), f, indent=2)
        with open(os.path.join(self.config.output_dir, TRACE_FILENAME), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=TRACE_FIELDS)
            writer.writeheader()
            writer.writerows(self.population.trace)
        torch.save(self.population.state_dicts(), os.path.join(self.config.output_dir, BRAINS_FILENAME))
        trace = self.population.trace
        snaps = self.metrics_logger.epoch_snapshots
        ratios = [s.get("town_broadcast_ratio") for s in snaps]
        actual = [s.get("actual_attendance") for s in snaps]
        summary["agent_model"] = "neural"
        summary["mind_lie_rate"] = round(sum(1 for row in trace if row["lied"]) / len(trace), 4) if trace else 0.0
        summary["mind_reroute_rate"] = round(sum(1 for row in trace if row["rerouted"]) / len(trace), 4) if trace else 0.0
        summary["average_town_broadcast_ratio"] = round(sum(ratios) / len(ratios), 4) if ratios else 0.0
        summary["average_actual_attendance"] = round(sum(actual) / len(actual), 4) if actual else 0.0
        summary["town_broadcast_ratio"] = ratios
        summary["actual_attendance"] = actual
        summary["manipulation_wins"] = sum(self.population.manipulation_wins)
        summary["brain_count"] = len(self.population.brains)
        summary["brain_parameters"] = self.population.brains[0].parameter_count() if self.population.brains else 0
        summary["brain_device"] = str(self.population.device)
        summary["brain_final_loss"] = round(self.population.losses[-1], 5) if self.population.losses else None
        with open(os.path.join(self.config.output_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary


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


class _NeuralTelemetryRunner(_TelemetryMixin, NeuralRunner):

    def __init__(self, config, device="cpu", pace_s=0.0, brain_options=None, on_epoch=None):
        super().__init__(config=config, device=device, pace_s=pace_s, brain_options=brain_options)
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
        if rec.get("broadcast"):
            broadcasts.append({"agent_id": rec.get("agent_id"), "text": str(rec.get("broadcast"))[:280]})
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
    share_history = {name: [] for name in OPTIONS}
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
        "town_broadcast_ratio",
        "actual_attendance",
        "intended_attendance",
    )
    if isinstance(population, BrainColony):
        agent_model = "neural"
    elif population is not None:
        agent_model = "minds"
    else:
        agent_model = "clone"
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
            **{k: [s.get(k) for s in snapshots] for k in ("bar_attendance", "deception_index", "mean_utility", "town_broadcast_ratio", "actual_attendance")},
            "strategy_shares": share_history,
        },
        "agents": agents,
        "broadcasts": broadcasts[-24:],
        "thoughts": thoughts[:40],
        "agent_model": agent_model,
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


def _vram_payload(snapshot, sentinel):
    payload = _round_vram(snapshot) or {}
    if sentinel is not None and sentinel.baseline:
        payload["baseline_used_mib"] = round(sentinel.baseline["used_mib"], 1)
    if sentinel is not None and sentinel.peak_used_mib is not None:
        payload["peak_used_mib"] = round(sentinel.peak_used_mib, 1)
    return payload


def _make_epoch_hook(live, sentinel, trial, attempt, stop_file=None):
    def hook(runner, epoch):
        if runner.epoch_timings:
            live.epoch_durations.append(runner.epoch_timings[-1])
        snapshot = sentinel.sample() if sentinel is not None else None
        if snapshot is not None:
            live.vram = _vram_payload(snapshot, sentinel)
        live.current = _live_frame(runner, trial, attempt, "running")
        live.publish("running")
        _check_stop(stop_file)
    return hook


def _device_ladder(device, max_attempts):
    ladder = [device]
    if str(device).startswith("cuda"):
        ladder.append("cpu")
    while len(ladder) < max_attempts:
        ladder.append(ladder[-1])
    return ladder[: max(1, int(max_attempts))]


def _execute_trial(config, trial, dry_run_pipeline, profile, max_attempts, sentinel, live, agent_model="neural", stop_file=None):
    if dry_run_pipeline is not None:
        ladder = [None]
    else:
        ladder = _device_ladder(profile["device"], max_attempts)
    runtime = {"status": "failed", "phase": "build", "attempts": 0, "error": None}
    for attempt, device in enumerate(ladder, start=1):
        if _SIGNALS["interrupted"]:
            raise KeyboardInterrupt()
        runner = None
        phase = "build"
        started = time.perf_counter()
        runtime = {"status": "failed", "phase": phase, "attempts": attempt, "error": None, "profile": None}
        live.mark(trial["key"], status="running", attempts=attempt)
        try:
            hook = _make_epoch_hook(live, sentinel, trial, attempt, stop_file)
            if dry_run_pipeline is not None:
                runner_class = _TelemetryRunner if agent_model == "clone" else _MindTelemetryRunner
                runner = runner_class(config=config, llm_pipeline=dry_run_pipeline, on_epoch=hook)
                runner.initialize()
                runtime["engine_build_s"] = 0.0
                runtime["agent_model"] = "clone" if agent_model == "clone" else "minds"
            else:
                live.current = _phase_frame(trial, attempt, "loading_engine", config.num_epochs, config.grid_size)
                live.publish("loading_engine")
                _log(
                    f"{trial['key']} :: attempt {attempt}/{len(ladder)} :: building engine :: "
                    f"{config.num_agents} isolated brains :: device {device} :: hidden {profile['hidden']} :: "
                    f"memory {profile['memory']} :: batch {profile['batch_size']} x {profile['updates']}"
                )
                build_started = time.perf_counter()
                runner = _NeuralTelemetryRunner(
                    config=config,
                    device=device,
                    pace_s=profile.get("pace", 0.0),
                    brain_options={k: profile[k] for k in ("hidden", "memory", "batch_size", "updates", "target_every", "history")},
                    on_epoch=hook,
                )
                runner.initialize()
                if str(device).startswith("cuda"):
                    torch.cuda.synchronize()
                runtime["engine_build_s"] = round(time.perf_counter() - build_started, 2)
                live.build_durations.append(runtime["engine_build_s"])
                loaded = sentinel.sample() if sentinel is not None else None
                runtime["vram_loaded_mib"] = round(loaded["used_mib"], 1) if loaded else None
                if loaded is not None:
                    live.vram = _vram_payload(loaded, sentinel)
                _log(f"{trial['key']} :: engine online in {runtime['engine_build_s']:.1f}s :: {len(runner.population.brains)} brains x {runner.population.brains[0].parameter_count()} params :: {_format_vram(loaded)}")
                runtime["agent_model"] = "neural"
                runtime["device"] = str(device)
            runtime["profile"] = dict(profile, device=str(device)) if device is not None else None
            phase = "run"
            runtime["phase"] = phase
            runner.run()
            summary = runner.export_results()
            summary.setdefault("agent_model", runtime["agent_model"])
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
            _log(f"{trial['key']} :: attempt {attempt}/{len(ladder)} failed during {phase} :: {runtime['error']}")
        finally:
            runtime["wall_time_s"] = round(time.perf_counter() - started, 2)
            if dry_run_pipeline is None:
                teardown_started = time.perf_counter()
                if torch is not None and torch.cuda.is_available():
                    try:
                        runtime["torch_peak_mib"] = round(torch.cuda.max_memory_allocated() / MIB, 1)
                        torch.cuda.reset_peak_memory_stats()
                    except Exception:
                        pass
                runner = None
                _clear_vram()
                released = sentinel.sample() if sentinel is not None else None
                if released is not None:
                    runtime["vram_after_teardown_mib"] = round(released["used_mib"], 1)
                    runtime["vram_release_s"] = round(time.perf_counter() - teardown_started, 2)
                    runtime["vram_peak_mib"] = round(sentinel.peak_used_mib, 1) if sentinel.peak_used_mib is not None else None
                    live.vram = _vram_payload(released, sentinel)
            runner = None
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
        "device": runtime.get("device"),
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
    model_name=NEURAL_MODEL,
    base_output_dir="outputs/experiment",
    num_agents=100,
    num_epochs=50,
    grid_size=50,
    seed=None,
    seeds=None,
    dry_run_pipeline=None,
    max_attempts=ARMOR_DEFAULTS["max_attempts"],
    resume=False,
    live_state=True,
    agent_model="neural",
    stop_file=None,
    device="auto",
    pace=0.0,
    hidden=ARMOR_DEFAULTS["hidden"],
    memory=ARMOR_DEFAULTS["memory"],
    batch_size=ARMOR_DEFAULTS["batch_size"],
    updates=ARMOR_DEFAULTS["updates"],
    target_every=ARMOR_DEFAULTS["target_every"],
    history=ARMOR_DEFAULTS["history"],
):
    if seeds is not None:
        seeds_to_run = list(seeds)
    elif seed is not None:
        seeds_to_run = [seed]
    else:
        seeds_to_run = [42, 100, 2026]

    if dry_run_pipeline is not None:
        effective_model = "clone" if agent_model == "clone" else "minds"
        resolved_device = None
    else:
        effective_model = "neural"
        resolved_device = _resolve_device(device)

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
        "device": resolved_device,
        "pace": float(pace or 0.0),
        "hidden": int(hidden),
        "memory": int(memory),
        "batch_size": int(batch_size),
        "updates": int(updates),
        "target_every": int(target_every),
        "history": int(history),
    }
    sentinel = _VRAMSentinel() if resolved_device is not None and resolved_device.startswith("cuda") else None
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
        if resolved_device is not None:
            _log(f"pre-flight :: {_format_vram(sentinel.baseline if sentinel else None)} :: {len(trials)} trials x {num_epochs} epochs x {num_agents} isolated brains on {resolved_device}")
            if sentinel is not None and sentinel.baseline:
                live.vram = _vram_payload(sentinel.baseline, sentinel)
        live.publish("booting")

        for index, trial in enumerate(trials):
            _check_stop(stop_file)
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
                condition_label=label,
            )

            if resume:
                cached = _load_completed_trial(output_dir, config, None, effective_model)
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
            summary, runtime = _execute_trial(config, trial, dry_run_pipeline, profile, max_attempts, sentinel, live, effective_model, stop_file=stop_file)

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
                        f"brains could not be started for {trial_key} after {runtime.get('attempts')} attempts: {runtime.get('error')}"
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
        for entry in live.trials:
            if entry["status"] == "running":
                entry["status"] = "interrupted"
        live.publish("interrupted")
        _log("interrupted :: brains torn down, partial comparison saved")
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
    parser.add_argument("--output-dir", type=str, default="outputs/experiment")
    parser.add_argument("--agents", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 100, 2026])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--pace", type=float, default=0.0)
    parser.add_argument("--hidden", type=int, default=ARMOR_DEFAULTS["hidden"])
    parser.add_argument("--memory", type=int, default=ARMOR_DEFAULTS["memory"])
    parser.add_argument("--batch-size", type=int, default=ARMOR_DEFAULTS["batch_size"])
    parser.add_argument("--updates", type=int, default=ARMOR_DEFAULTS["updates"])
    parser.add_argument("--target-every", type=int, default=ARMOR_DEFAULTS["target_every"])
    parser.add_argument("--history", type=int, default=ARMOR_DEFAULTS["history"])
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--rehearsal", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=ARMOR_DEFAULTS["max_attempts"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-live-state", action="store_true")
    parser.add_argument("--stop-file", type=str, default=None)
    args = parser.parse_args()

    active_seeds = [args.seed] if args.seed is not None else args.seeds

    _SIGNALS["interrupted"] = False
    previous_handlers = _install_signal_armor()
    t0 = time.perf_counter()
    try:
        device = _resolve_device("cpu" if args.rehearsal else args.device)
        model_name = f"{NEURAL_MODEL} ({args.hidden}h, {observation_dim(args.history)}-d state) on {device}"
        all_summaries, comparison_path = run_experiment(
            model_name=model_name,
            base_output_dir=args.output_dir,
            num_agents=args.agents,
            num_epochs=args.epochs,
            seeds=active_seeds,
            max_attempts=args.max_attempts,
            resume=args.resume,
            live_state=not args.no_live_state,
            stop_file=args.stop_file,
            device=device,
            pace=args.pace,
            hidden=args.hidden,
            memory=args.memory,
            batch_size=args.batch_size,
            updates=args.updates,
            target_every=args.target_every,
            history=args.history,
        )
    except KeyboardInterrupt:
        print(f"\nSweep interrupted after {time.perf_counter() - t0:.2f}s. Partial results: {os.path.join(args.output_dir, 'comparison.json')}")
        print("Re-run with --resume to continue from the last verified trial.")
        sys.exit(130)
    except EngineStartupError as exc:
        print(f"\nSweep aborted: {exc}")
        print(f"Partial results: {os.path.join(args.output_dir, 'comparison.json')}. Re-run with --resume once the device is clear.")
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
