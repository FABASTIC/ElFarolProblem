from dataclasses import dataclass, field
import random


@dataclass
class Agent:
    id: int
    x: int
    y: int
    local_messages: list = field(default_factory=list)
    global_messages: list = field(default_factory=list)
    utility_history: list = field(default_factory=list)
    stated_intention: str = field(default="staying")
    actual_target: str = field(default="home")


class AgentPool:

    def __init__(self):
        self.agents = {}

    def spawn(self, num_agents, grid):
        all_positions = [(x, y) for x in range(grid.size) for y in range(grid.size)]
        selected = random.sample(all_positions, num_agents)
        for i, (x, y) in enumerate(selected):
            agent_id = i + 1
            grid.place(agent_id, x, y)
            agent = Agent(id=agent_id, x=x, y=y)
            self.agents[agent_id] = agent
        return list(self.agents.values())

    def get_local_chatter(self, agent, grid, radius=5):
        nearby_ids = grid.nearby_agents(agent.x, agent.y, radius)
        messages = []
        for nid in nearby_ids:
            neighbor = self.agents[nid]
            messages.extend(neighbor.local_messages)
        return messages

    def broadcast(self, message):
        for agent in self.agents.values():
            agent.global_messages.append(message)

    def clear_messages(self):
        for agent in self.agents.values():
            agent.local_messages.clear()
            agent.global_messages.clear()

    def get_agent(self, agent_id):
        return self.agents[agent_id]

    def all_agents(self):
        return list(self.agents.values())
