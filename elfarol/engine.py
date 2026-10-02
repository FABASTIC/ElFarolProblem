from elfarol.grid import SpatialGrid
from elfarol.agent import AgentPool
from elfarol.perception import build_batch_prompts
from elfarol.action import execute_actions, execute_brain_actions
from elfarol.agent_brain import build_brain_batch_prompts, process_brain_batch
from elfarol.metrics_logger import MetricsLogger
import random


class SimulationEngine:

    def __init__(self, num_agents, grid_size, llm_pipeline, epochs, seed=None):
        self.num_agents = num_agents
        self.grid_size = grid_size
        self.llm_pipeline = llm_pipeline
        self.epochs = epochs
        self.seed = seed
        self.grid = SpatialGrid(grid_size)
        self.agent_pool = AgentPool()
        self.metrics = {
            "bar_occupancy": [],
            "collision_attempts": [],
            "message_counts": [],
        }
        self.rng = random.Random(seed)

    def initialize(self):
        random.seed(self.seed)
        self.agent_pool.spawn(self.num_agents, self.grid)

    def run(self):
        self.initialize()
        for epoch in range(self.epochs):
            self._tick(epoch)
        return self.metrics

    def _tick(self, epoch):
        agents = self.agent_pool.all_agents()

        prompts = build_batch_prompts(agents, self.grid, self.agent_pool)

        fallback_positions = [(a.x, a.y) for a in agents]
        actions = self.llm_pipeline.generate_batch(
            prompts, fallback_positions=fallback_positions
        )

        execute_actions(agents, actions, self.grid, self.agent_pool, rng=self.rng)

        bar_occ = self.grid.occupancy()
        self.metrics["bar_occupancy"].append(bar_occ)
        self.metrics["collision_attempts"].append(0)
        self.metrics["message_counts"].append(
            sum(len(a.local_messages) + len(a.global_messages) for a in agents)
        )

        self.agent_pool.clear_messages()

    def validate_state(self):
        agents = self.agent_pool.all_agents()
        positions = set()
        for agent in agents:
            if not (0 <= agent.x < self.grid_size and 0 <= agent.y < self.grid_size):
                return False
            if (agent.x, agent.y) in positions:
                return False
            positions.add((agent.x, agent.y))
            if self.grid.grid[agent.y, agent.x] != agent.id:
                return False
        return True


class BrainSimulationEngine:

    def __init__(self, num_agents, grid_size, llm_pipeline, epochs, seed=None, output_dir=None):
        self.num_agents = num_agents
        self.grid_size = grid_size
        self.llm_pipeline = llm_pipeline
        self.epochs = epochs
        self.seed = seed
        self.grid = SpatialGrid(grid_size)
        self.agent_pool = AgentPool()
        self.metrics_logger = MetricsLogger(output_dir=output_dir)
        self.broadcast_history = []
        self.rng = random.Random(seed)

    def initialize(self):
        random.seed(self.seed)
        self.agent_pool.spawn(self.num_agents, self.grid)

    def run(self):
        self.initialize()
        for epoch in range(self.epochs):
            self._tick(epoch)
        return self.metrics_logger

    def _tick(self, epoch):
        agents = self.agent_pool.all_agents()

        prompts = build_brain_batch_prompts(
            agents, self.grid, self.agent_pool, self.broadcast_history, epoch
        )

        raw_outputs = self.llm_pipeline.generate_batch(
            prompts, fallback_positions=[(a.x, a.y) for a in agents]
        )

        brain_actions = process_brain_batch(raw_outputs, agents, self.grid_size)

        execute_brain_actions(
            agents, brain_actions, self.grid, self.agent_pool, rng=self.rng
        )

        self.metrics_logger.record_epoch(epoch, agents, self.grid, brain_actions)

        for action in brain_actions:
            if action.get("broadcast"):
                self.broadcast_history.append(action["broadcast"])

        self.agent_pool.clear_messages()

    def validate_state(self):
        agents = self.agent_pool.all_agents()
        positions = set()
        for agent in agents:
            if not (0 <= agent.x < self.grid_size and 0 <= agent.y < self.grid_size):
                return False
            if (agent.x, agent.y) in positions:
                return False
            positions.add((agent.x, agent.y))
            if self.grid.grid[agent.y, agent.x] != agent.id:
                return False
        return True
