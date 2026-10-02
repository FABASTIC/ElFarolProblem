import random
import pytest
from elfarol.grid import SpatialGrid
from elfarol.agent import Agent, AgentPool


class TestAgent:

    def test_creation(self):
        a = Agent(id=1, x=10, y=20)
        assert a.id == 1
        assert a.x == 10
        assert a.y == 20

    def test_default_messages_empty(self):
        a = Agent(id=1, x=0, y=0)
        assert a.local_messages == []
        assert a.global_messages == []

    def test_message_isolation(self):
        a = Agent(id=1, x=0, y=0)
        b = Agent(id=2, x=1, y=1)
        a.local_messages.append("hello")
        assert len(b.local_messages) == 0


class TestAgentPool:

    def test_spawn_count(self):
        random.seed(42)
        g = SpatialGrid(50)
        pool = AgentPool()
        agents = pool.spawn(50, g)
        assert len(agents) == 50

    def test_spawn_unique_positions(self):
        random.seed(42)
        g = SpatialGrid(50)
        pool = AgentPool()
        agents = pool.spawn(50, g)
        positions = [(a.x, a.y) for a in agents]
        assert len(positions) == len(set(positions))

    def test_spawn_agents_on_grid(self):
        random.seed(42)
        g = SpatialGrid(50)
        pool = AgentPool()
        agents = pool.spawn(10, g)
        for a in agents:
            assert g.grid[a.y, a.x] == a.id

    def test_get_agent(self):
        random.seed(42)
        g = SpatialGrid(50)
        pool = AgentPool()
        pool.spawn(5, g)
        a = pool.get_agent(3)
        assert a.id == 3

    def test_all_agents(self):
        random.seed(42)
        g = SpatialGrid(50)
        pool = AgentPool()
        pool.spawn(10, g)
        assert len(pool.all_agents()) == 10


class TestAgentPoolMessaging:

    def test_broadcast(self):
        random.seed(42)
        g = SpatialGrid(50)
        pool = AgentPool()
        pool.spawn(5, g)
        pool.broadcast("test message")
        for a in pool.all_agents():
            assert "test message" in a.global_messages

    def test_local_chatter(self):
        g = SpatialGrid(50)
        pool = AgentPool()

        a1 = Agent(id=1, x=10, y=10)
        a2 = Agent(id=2, x=12, y=10)
        a3 = Agent(id=3, x=40, y=40)

        for a in [a1, a2, a3]:
            g.place(a.id, a.x, a.y)
            pool.agents[a.id] = a

        a2.local_messages.append("nearby hello")
        a3.local_messages.append("far away hello")

        chatter = pool.get_local_chatter(a1, g, radius=5)
        assert "nearby hello" in chatter
        assert "far away hello" not in chatter

    def test_clear_messages(self):
        random.seed(42)
        g = SpatialGrid(50)
        pool = AgentPool()
        pool.spawn(5, g)
        pool.broadcast("msg")
        pool.all_agents()[0].local_messages.append("local")
        pool.clear_messages()
        for a in pool.all_agents():
            assert len(a.local_messages) == 0
            assert len(a.global_messages) == 0
