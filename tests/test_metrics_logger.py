import json
import csv
import os
import random
import pytest
from elfarol.grid import SpatialGrid
from elfarol.agent import Agent, AgentPool
from elfarol.metrics_logger import (
    MetricsLogger,
    UTILITY_AT_BAR_COMFORTABLE,
    UTILITY_AT_BAR_OVERCROWDED,
    UTILITY_AT_HOME,
    COMFORT_THRESHOLD_RATIO,
)


def _make_grid_and_agents(positions):
    grid = SpatialGrid(50)
    pool = AgentPool()
    agents = []
    for i, (x, y) in enumerate(positions):
        agent_id = i + 1
        a = Agent(id=agent_id, x=x, y=y)
        grid.place(agent_id, x, y)
        pool.agents[agent_id] = a
        agents.append(a)
    return grid, pool, agents


class TestComputeUtility:

    def test_at_bar_comfortable(self):
        grid, pool, agents = _make_grid_and_agents([(25, 25)])
        logger = MetricsLogger()
        util = logger.compute_utility(agents[0], grid, bar_occupancy=30, bar_capacity=100)
        assert util == UTILITY_AT_BAR_COMFORTABLE

    def test_at_bar_overcrowded(self):
        grid, pool, agents = _make_grid_and_agents([(25, 25)])
        logger = MetricsLogger()
        util = logger.compute_utility(agents[0], grid, bar_occupancy=70, bar_capacity=100)
        assert util == UTILITY_AT_BAR_OVERCROWDED

    def test_at_home(self):
        grid, pool, agents = _make_grid_and_agents([(0, 0)])
        logger = MetricsLogger()
        util = logger.compute_utility(agents[0], grid, bar_occupancy=30, bar_capacity=100)
        assert util == UTILITY_AT_HOME

    def test_at_threshold_boundary(self):
        grid, pool, agents = _make_grid_and_agents([(25, 25)])
        logger = MetricsLogger()
        threshold = int(COMFORT_THRESHOLD_RATIO * 100)
        util = logger.compute_utility(agents[0], grid, bar_occupancy=threshold, bar_capacity=100)
        assert util == UTILITY_AT_BAR_COMFORTABLE

    def test_one_above_threshold(self):
        grid, pool, agents = _make_grid_and_agents([(25, 25)])
        logger = MetricsLogger()
        threshold = int(COMFORT_THRESHOLD_RATIO * 100)
        util = logger.compute_utility(agents[0], grid, bar_occupancy=threshold + 1, bar_capacity=100)
        assert util == UTILITY_AT_BAR_OVERCROWDED


class TestDeceptionDetection:

    def test_says_going_actually_home_is_deceptive(self):
        logger = MetricsLogger()
        assert logger._check_deception("going", "home") is True

    def test_says_staying_actually_bar_is_deceptive(self):
        logger = MetricsLogger()
        assert logger._check_deception("staying", "bar") is True

    def test_says_going_actually_bar_is_truthful(self):
        logger = MetricsLogger()
        assert logger._check_deception("going", "bar") is False

    def test_says_staying_actually_home_is_truthful(self):
        logger = MetricsLogger()
        assert logger._check_deception("staying", "home") is False


class TestRecordEpoch:

    def _make_action(self, stated="going", target="bar", broadcast=None):
        return {
            "move": [25, 25],
            "broadcast": broadcast,
            "proximity_speech": None,
            "stated_intention": stated,
            "actual_target": target,
        }

    def test_snapshot_has_required_keys(self):
        grid, pool, agents = _make_grid_and_agents([(25, 25), (0, 0)])
        logger = MetricsLogger()
        actions = [
            self._make_action("going", "bar"),
            self._make_action("staying", "home"),
        ]
        snap = logger.record_epoch(0, agents, grid, actions)
        assert "epoch" in snap
        assert "bar_attendance" in snap
        assert "deception_index" in snap
        assert "truthfulness_ratio" in snap
        assert "broadcast_correlation" in snap
        assert "agents" in snap

    def test_bar_attendance_correct(self):
        grid, pool, agents = _make_grid_and_agents([(25, 25), (26, 26), (0, 0)])
        logger = MetricsLogger()
        actions = [
            self._make_action("going", "bar"),
            self._make_action("going", "bar"),
            self._make_action("staying", "home"),
        ]
        snap = logger.record_epoch(0, agents, grid, actions)
        assert snap["bar_attendance"] == 2

    def test_deception_index_all_truthful(self):
        grid, pool, agents = _make_grid_and_agents([(25, 25), (0, 0)])
        logger = MetricsLogger()
        actions = [
            self._make_action("going", "bar"),
            self._make_action("staying", "home"),
        ]
        snap = logger.record_epoch(0, agents, grid, actions)
        assert snap["deception_index"] == 0.0
        assert snap["truthfulness_ratio"] == 1.0

    def test_deception_index_all_deceptive(self):
        grid, pool, agents = _make_grid_and_agents([(25, 25), (0, 0)])
        logger = MetricsLogger()
        actions = [
            self._make_action("staying", "home"),
            self._make_action("going", "bar"),
        ]
        snap = logger.record_epoch(0, agents, grid, actions)
        assert snap["deception_index"] == 1.0
        assert snap["truthfulness_ratio"] == 0.0

    def test_deception_index_mixed(self):
        grid, pool, agents = _make_grid_and_agents([(25, 25), (0, 0), (26, 26), (1, 1)])
        logger = MetricsLogger()
        actions = [
            self._make_action("going", "bar"),
            self._make_action("staying", "home"),
            self._make_action("staying", "home"),
            self._make_action("going", "bar"),
        ]
        snap = logger.record_epoch(0, agents, grid, actions)
        assert snap["deception_count"] == 2
        assert snap["truthful_count"] == 2
        assert snap["deception_index"] == 0.5

    def test_utility_recorded_in_agent(self):
        grid, pool, agents = _make_grid_and_agents([(25, 25)])
        logger = MetricsLogger()
        actions = [self._make_action("going", "bar")]
        logger.record_epoch(0, agents, grid, actions)
        assert len(agents[0].utility_history) == 1
        assert agents[0].utility_history[0] == UTILITY_AT_BAR_COMFORTABLE

    def test_cumulative_utility_accumulates(self):
        grid, pool, agents = _make_grid_and_agents([(25, 25)])
        logger = MetricsLogger()
        actions = [self._make_action("going", "bar")]
        logger.record_epoch(0, agents, grid, actions)
        logger.record_epoch(1, agents, grid, actions)
        assert logger.agent_cumulative_utility[1] == 2 * UTILITY_AT_BAR_COMFORTABLE

    def test_broadcast_count(self):
        grid, pool, agents = _make_grid_and_agents([(25, 25), (0, 0)])
        logger = MetricsLogger()
        actions = [
            self._make_action("going", "bar", broadcast="I am going!"),
            self._make_action("staying", "home"),
        ]
        snap = logger.record_epoch(0, agents, grid, actions)
        assert snap["broadcast_count"] == 1

    def test_agent_records_present(self):
        grid, pool, agents = _make_grid_and_agents([(25, 25), (0, 0)])
        logger = MetricsLogger()
        actions = [
            self._make_action("going", "bar"),
            self._make_action("staying", "home"),
        ]
        snap = logger.record_epoch(0, agents, grid, actions)
        assert len(snap["agents"]) == 2
        for rec in snap["agents"]:
            assert "agent_id" in rec
            assert "utility" in rec
            assert "is_deceptive" in rec
            assert "actual_location" in rec


class TestAgentDeceptionRatio:

    def test_no_records_returns_zero(self):
        logger = MetricsLogger()
        assert logger.get_agent_deception_ratio(999) == 0.0

    def test_all_truthful(self):
        logger = MetricsLogger()
        logger.agent_truthful_counts[1] = 10
        logger.agent_deception_counts[1] = 0
        assert logger.get_agent_deception_ratio(1) == 0.0

    def test_all_deceptive(self):
        logger = MetricsLogger()
        logger.agent_truthful_counts[1] = 0
        logger.agent_deception_counts[1] = 5
        assert logger.get_agent_deception_ratio(1) == 1.0

    def test_mixed(self):
        logger = MetricsLogger()
        logger.agent_truthful_counts[1] = 3
        logger.agent_deception_counts[1] = 7
        assert abs(logger.get_agent_deception_ratio(1) - 0.7) < 1e-9


class TestPopulationTruthfulness:

    def test_no_data(self):
        logger = MetricsLogger()
        assert logger.get_population_truthfulness() == 1.0

    def test_mixed_population(self):
        logger = MetricsLogger()
        logger.agent_truthful_counts[1] = 8
        logger.agent_deception_counts[1] = 2
        logger.agent_truthful_counts[2] = 6
        logger.agent_deception_counts[2] = 4
        total_t = 14
        total_d = 6
        assert abs(logger.get_population_truthfulness() - total_t / (total_t + total_d)) < 1e-9


class TestExportJSON:

    def test_export_creates_file(self, tmp_path):
        logger = MetricsLogger()
        grid, pool, agents = _make_grid_and_agents([(25, 25)])
        actions = [{"move": [25, 25], "broadcast": None, "proximity_speech": None, "stated_intention": "going", "actual_target": "bar"}]
        logger.record_epoch(0, agents, grid, actions)

        filepath = str(tmp_path / "metrics.json")
        result = logger.export_json(filepath)
        assert os.path.exists(result)

        with open(result, "r") as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["epoch"] == 0
        assert "deception_index" in data[0]
        assert "agents" not in data[0]

    def test_export_multiple_epochs(self, tmp_path):
        logger = MetricsLogger()
        grid, pool, agents = _make_grid_and_agents([(25, 25), (0, 0)])
        actions = [
            {"move": [25, 25], "broadcast": None, "proximity_speech": None, "stated_intention": "going", "actual_target": "bar"},
            {"move": [0, 0], "broadcast": None, "proximity_speech": None, "stated_intention": "staying", "actual_target": "home"},
        ]
        for i in range(10):
            logger.record_epoch(i, agents, grid, actions)

        filepath = str(tmp_path / "metrics.json")
        logger.export_json(filepath)
        with open(filepath, "r") as f:
            data = json.load(f)
        assert len(data) == 10


class TestExportCSV:

    def test_export_creates_file(self, tmp_path):
        logger = MetricsLogger()
        grid, pool, agents = _make_grid_and_agents([(25, 25)])
        actions = [{"move": [25, 25], "broadcast": None, "proximity_speech": None, "stated_intention": "going", "actual_target": "bar"}]
        logger.record_epoch(0, agents, grid, actions)

        filepath = str(tmp_path / "metrics.csv")
        result = logger.export_csv(filepath)
        assert os.path.exists(result)

        with open(result, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["epoch"] == "0"
        assert "deception_index" in rows[0]

    def test_agent_csv_export(self, tmp_path):
        logger = MetricsLogger()
        grid, pool, agents = _make_grid_and_agents([(25, 25), (0, 0)])
        actions = [
            {"move": [25, 25], "broadcast": None, "proximity_speech": None, "stated_intention": "going", "actual_target": "bar"},
            {"move": [0, 0], "broadcast": None, "proximity_speech": None, "stated_intention": "staying", "actual_target": "home"},
        ]
        logger.record_epoch(0, agents, grid, actions)
        logger.record_epoch(1, agents, grid, actions)

        filepath = str(tmp_path / "agent_details.csv")
        result = logger.export_agent_csv(filepath)
        assert os.path.exists(result)

        with open(result, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 4


class TestSummary:

    def test_empty_summary(self):
        logger = MetricsLogger()
        assert logger.summary() == {}

    def test_summary_keys(self):
        logger = MetricsLogger()
        grid, pool, agents = _make_grid_and_agents([(25, 25), (0, 0)])
        actions = [
            {"move": [25, 25], "broadcast": None, "proximity_speech": None, "stated_intention": "going", "actual_target": "bar"},
            {"move": [0, 0], "broadcast": None, "proximity_speech": None, "stated_intention": "staying", "actual_target": "home"},
        ]
        logger.record_epoch(0, agents, grid, actions)
        s = logger.summary()
        assert "total_epochs" in s
        assert "average_bar_attendance" in s
        assert "average_deception_index" in s
        assert "average_mean_utility" in s
        assert "population_truthfulness" in s

    def test_summary_values(self):
        logger = MetricsLogger()
        grid, pool, agents = _make_grid_and_agents([(25, 25), (0, 0)])
        actions = [
            {"move": [25, 25], "broadcast": None, "proximity_speech": None, "stated_intention": "going", "actual_target": "bar"},
            {"move": [0, 0], "broadcast": None, "proximity_speech": None, "stated_intention": "staying", "actual_target": "home"},
        ]
        for i in range(5):
            logger.record_epoch(i, agents, grid, actions)
        s = logger.summary()
        assert s["total_epochs"] == 5
        assert s["population_truthfulness"] == 1.0


class TestBroadcastCorrelation:

    def test_no_history_returns_zero(self):
        logger = MetricsLogger()
        assert logger._compute_broadcast_correlation(0) == 0.0

    def test_with_attendance_shift(self):
        logger = MetricsLogger()
        logger.attendance_history = [10, 20]
        logger.epoch_snapshots = [
            {"broadcast_count": 3},
            {"broadcast_count": 2},
        ]
        corr = logger._compute_broadcast_correlation(1)
        assert isinstance(corr, float)
