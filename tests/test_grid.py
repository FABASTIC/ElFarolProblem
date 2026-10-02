import numpy as np
import pytest
from elfarol.grid import SpatialGrid


class TestSpatialGridInit:

    def test_grid_shape(self):
        g = SpatialGrid(50)
        assert g.grid.shape == (50, 50)

    def test_grid_dtype(self):
        g = SpatialGrid(50)
        assert g.grid.dtype == np.int32

    def test_grid_initially_empty(self):
        g = SpatialGrid(50)
        assert np.count_nonzero(g.grid) == 0

    def test_bar_zone_centered(self):
        g = SpatialGrid(50)
        assert g.bar_min == 20
        assert g.bar_max == 30

    def test_bar_zone_custom_size(self):
        g = SpatialGrid(100)
        assert g.bar_min == 45
        assert g.bar_max == 55


class TestSpatialGridPlacement:

    def test_place_agent(self):
        g = SpatialGrid(50)
        g.place(1, 10, 20)
        assert g.grid[20, 10] == 1

    def test_place_collision_raises(self):
        g = SpatialGrid(50)
        g.place(1, 10, 20)
        with pytest.raises(ValueError):
            g.place(2, 10, 20)

    def test_place_out_of_bounds_raises(self):
        g = SpatialGrid(50)
        with pytest.raises(ValueError):
            g.place(1, 50, 0)
        with pytest.raises(ValueError):
            g.place(1, -1, 0)

    def test_remove_agent(self):
        g = SpatialGrid(50)
        g.place(1, 5, 5)
        g.remove(1, 5, 5)
        assert g.grid[5, 5] == 0

    def test_remove_wrong_agent_raises(self):
        g = SpatialGrid(50)
        g.place(1, 5, 5)
        with pytest.raises(ValueError):
            g.remove(2, 5, 5)


class TestSpatialGridMove:

    def test_move_to_empty(self):
        g = SpatialGrid(50)
        g.place(1, 0, 0)
        g.move(1, 0, 0, 5, 5)
        assert g.grid[0, 0] == 0
        assert g.grid[5, 5] == 1

    def test_move_to_occupied_raises(self):
        g = SpatialGrid(50)
        g.place(1, 0, 0)
        g.place(2, 5, 5)
        with pytest.raises(ValueError):
            g.move(1, 0, 0, 5, 5)

    def test_move_out_of_bounds_raises(self):
        g = SpatialGrid(50)
        g.place(1, 0, 0)
        with pytest.raises(ValueError):
            g.move(1, 0, 0, 50, 0)

    def test_move_in_place(self):
        g = SpatialGrid(50)
        g.place(1, 10, 10)
        g.move(1, 10, 10, 10, 10)
        assert g.grid[10, 10] == 1

    def test_move_wrong_source_raises(self):
        g = SpatialGrid(50)
        g.place(1, 0, 0)
        with pytest.raises(ValueError):
            g.move(1, 5, 5, 10, 10)


class TestSpatialGridBarZone:

    def test_inside_bar(self):
        g = SpatialGrid(50)
        assert g.is_in_bar(20, 20) is True
        assert g.is_in_bar(29, 29) is True
        assert g.is_in_bar(24, 25) is True

    def test_outside_bar(self):
        g = SpatialGrid(50)
        assert g.is_in_bar(19, 20) is False
        assert g.is_in_bar(30, 20) is False
        assert g.is_in_bar(0, 0) is False

    def test_occupancy_empty(self):
        g = SpatialGrid(50)
        assert g.occupancy() == 0

    def test_occupancy_with_agents(self):
        g = SpatialGrid(50)
        g.place(1, 20, 20)
        g.place(2, 25, 25)
        g.place(3, 0, 0)
        assert g.occupancy() == 2


class TestSpatialGridNearby:

    def test_nearby_finds_agents(self):
        g = SpatialGrid(50)
        g.place(1, 10, 10)
        g.place(2, 12, 10)
        g.place(3, 40, 40)
        nearby = g.nearby_agents(10, 10, radius=5)
        assert 2 in nearby
        assert 3 not in nearby
        assert 1 not in nearby

    def test_nearby_at_corner(self):
        g = SpatialGrid(50)
        g.place(1, 0, 0)
        g.place(2, 3, 3)
        nearby = g.nearby_agents(0, 0, radius=5)
        assert 2 in nearby

    def test_nearby_excludes_self(self):
        g = SpatialGrid(50)
        g.place(1, 10, 10)
        nearby = g.nearby_agents(10, 10, radius=5)
        assert 1 not in nearby

    def test_nearby_empty(self):
        g = SpatialGrid(50)
        g.place(1, 0, 0)
        nearby = g.nearby_agents(49, 49, radius=5)
        assert len(nearby) == 0


class TestSpatialGridSnapshot:

    def test_snapshot_is_copy(self):
        g = SpatialGrid(50)
        g.place(1, 5, 5)
        snap = g.snapshot()
        snap[5, 5] = 999
        assert g.grid[5, 5] == 1

    def test_snapshot_matches_grid(self):
        g = SpatialGrid(50)
        g.place(1, 0, 0)
        g.place(2, 49, 49)
        snap = g.snapshot()
        assert np.array_equal(snap, g.grid)
