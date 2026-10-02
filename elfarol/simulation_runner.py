import os
import sys
import json
import time
import dataclasses
from pathlib import Path

from elfarol.grid import SpatialGrid
from elfarol.agent import AgentPool
from elfarol.agent_brain import build_brain_batch_prompts, process_brain_batch
from elfarol.action import execute_brain_actions
from elfarol.metrics_logger import MetricsLogger

NUM_AGENTS = 50
GRID_SIZE = 50
NUM_EPOCHS = 100
BAR_CAPACITY = 100
COMFORT_THRESHOLD_RATIO = 0.6
DEFAULT_MODEL = "casperhansen/llama-3-8b-instruct-awq"
DEFAULT_MAX_MODEL_LEN = 4096
DEFAULT_GPU_UTIL = 0.85
DEFAULT_TENSOR_PARALLEL = 1


@dataclasses.dataclass
class RunConfig:
    model_name: str = DEFAULT_MODEL
    num_agents: int = NUM_AGENTS
    grid_size: int = GRID_SIZE
    num_epochs: int = NUM_EPOCHS
    seed: int = 42
    broadcast_enabled: bool = True
    output_dir: str = "outputs/run"
    max_model_len: int = DEFAULT_MAX_MODEL_LEN
    gpu_memory_utilization: float = DEFAULT_GPU_UTIL
    tensor_parallel_size: int = DEFAULT_TENSOR_PARALLEL
    condition_label: str = "delta2"


def _build_vllm_pipeline(config: RunConfig):
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=config.model_name,
        quantization="awq",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_model_len=2048,
        enforce_eager=True,
        dtype="half",
    )
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=256,
        top_p=0.9,
        stop=["\n\n"],
    )
    return llm, sampling_params


class VLLMBatchPipeline:

    def __init__(self, llm, sampling_params):
        self.llm = llm
        self.sampling_params = sampling_params
        self.total_generations = 0
        self.fallback_count = 0

    def generate_batch(self, prompts, fallback_positions=None):
        outputs = self.llm.generate(prompts, self.sampling_params)
        results = []
        for i, output in enumerate(outputs):
            self.total_generations += 1
            text = output.outputs[0].text.strip()
            result = _parse_brain_json(text)
            if result is None:
                self.fallback_count += 1
                if fallback_positions is not None:
                    fx, fy = fallback_positions[i]
                    result = {"move": [fx, fy], "broadcast": None, "proximity_speech": None, "stated_intention": "staying", "actual_target": "home"}
                else:
                    result = {"move": [0, 0], "broadcast": None, "proximity_speech": None, "stated_intention": "staying", "actual_target": "home"}
            results.append(result)
        return results


def _parse_brain_json(text):
    import json as _json
    import re
    try:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return None
        data = _json.loads(match.group(0))
        if "move" not in data:
            return None
        if not isinstance(data["move"], list) or len(data["move"]) != 2:
            return None
        data["move"] = [int(data["move"][0]), int(data["move"][1])]
        data.setdefault("broadcast", None)
        data.setdefault("proximity_speech", None)
        if data.get("stated_intention") not in ("going", "staying"):
            data["stated_intention"] = "staying"
        if data.get("actual_target") not in ("bar", "home"):
            data["actual_target"] = "home"
        return data
    except (ValueError, _json.JSONDecodeError, TypeError, KeyError):
        return None


def _strip_broadcasts_from_actions(brain_actions):
    stripped = []
    for action in brain_actions:
        a = dict(action)
        a["broadcast"] = None
        stripped.append(a)
    return stripped


class SimulationRunner:

    def __init__(self, config: RunConfig, llm_pipeline=None):
        self.config = config
        self.llm_pipeline = llm_pipeline
        self.grid = SpatialGrid(config.grid_size)
        self.agent_pool = AgentPool()
        self.metrics_logger = MetricsLogger(output_dir=config.output_dir)
        self.broadcast_history = []
        self.epoch_timings = []
        import random
        self.rng = random.Random(config.seed)
        self._initialized = False

    def initialize(self):
        import random
        random.seed(self.config.seed)
        self.agent_pool.spawn(self.config.num_agents, self.grid)
        os.makedirs(self.config.output_dir, exist_ok=True)
        self._initialized = True

    def run(self):
        if not self._initialized:
            self.initialize()

        for epoch in range(self.config.num_epochs):
            t_start = time.perf_counter()
            self._step(epoch)
            t_end = time.perf_counter()
            self.epoch_timings.append(round(t_end - t_start, 4))
            self._flush_epoch(epoch)

        return self.metrics_logger

    def _step(self, epoch):
        agents = self.agent_pool.all_agents()

        effective_broadcast = self.broadcast_history if self.config.broadcast_enabled else []

        prompts = build_brain_batch_prompts(
            agents,
            self.grid,
            self.agent_pool,
            effective_broadcast,
            epoch,
        )

        fallback_positions = [(a.x, a.y) for a in agents]
        prev_fallbacks = getattr(self.llm_pipeline, "fallback_count", 0)
        prev_total = getattr(self.llm_pipeline, "total_generations", 0)

        raw_outputs = self.llm_pipeline.generate_batch(
            prompts,
            fallback_positions=fallback_positions,
        )

        new_fallbacks = getattr(self.llm_pipeline, "fallback_count", 0) - prev_fallbacks
        new_total = getattr(self.llm_pipeline, "total_generations", 0) - prev_total

        brain_actions = process_brain_batch(raw_outputs, agents, self.config.grid_size)

        if not self.config.broadcast_enabled:
            brain_actions = _strip_broadcasts_from_actions(brain_actions)

        execute_brain_actions(
            agents,
            brain_actions,
            self.grid,
            self.agent_pool,
            rng=self.rng,
        )

        self.metrics_logger.record_epoch(epoch, agents, self.grid, brain_actions)
        if hasattr(self.metrics_logger, "record_fallbacks"):
            self.metrics_logger.record_fallbacks(new_fallbacks, new_total)
        else:
            self.metrics_logger.fallback_count = getattr(self.llm_pipeline, "fallback_count", 0)
            self.metrics_logger.total_generations = getattr(self.llm_pipeline, "total_generations", 0)
            total = self.metrics_logger.total_generations
            self.metrics_logger.regex_fallback_rate = (
                round(self.metrics_logger.fallback_count / total, 4) if total > 0 else 0.0
            )

        if self.config.broadcast_enabled:
            for action in brain_actions:
                if action.get("broadcast"):
                    self.broadcast_history.append(action["broadcast"])

        self.agent_pool.clear_messages()

    def _flush_epoch(self, epoch):
        snap = self.metrics_logger.epoch_snapshots[-1]
        flush_path = os.path.join(self.config.output_dir, f"epoch_{epoch:04d}.json")
        with open(flush_path, "w", encoding="utf-8") as f:
            epoch_export = {k: v for k, v in snap.items() if k != "agents"}
            epoch_export["agent_count"] = len(snap.get("agents", []))
            epoch_export["epoch_duration_s"] = self.epoch_timings[-1]
            json.dump(epoch_export, f)

    def export_results(self):
        self.metrics_logger.export_json(
            os.path.join(self.config.output_dir, "epoch_metrics.json")
        )
        self.metrics_logger.export_csv(
            os.path.join(self.config.output_dir, "epoch_metrics.csv")
        )
        self.metrics_logger.export_agent_csv(
            os.path.join(self.config.output_dir, "agent_epoch_details.csv")
        )
        summary = self.metrics_logger.summary()
        summary["condition"] = self.config.condition_label
        summary["broadcast_enabled"] = self.config.broadcast_enabled
        summary["epoch_timings_mean_s"] = (
            round(sum(self.epoch_timings) / len(self.epoch_timings), 4)
            if self.epoch_timings else 0.0
        )
        summary_path = os.path.join(self.config.output_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary

    def validate_state(self):
        agents = self.agent_pool.all_agents()
        positions = set()
        for agent in agents:
            if not (0 <= agent.x < self.config.grid_size and 0 <= agent.y < self.config.grid_size):
                return False
            if (agent.x, agent.y) in positions:
                return False
            positions.add((agent.x, agent.y))
            if self.grid.grid[agent.y, agent.x] != agent.id:
                return False
        return True
