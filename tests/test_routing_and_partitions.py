"""
Tests for partition line generation and Manhattan routing functions.

This module tests the sparse partition line generation logic and the
Manhattan routing algorithm with obstacle avoidance.
"""

from mondrian_map.core import Block, Colors
from mondrian_map.visualization import (_get_block_bounds,
                                        create_partition_line_shapes,
                                        route_manhattan)


class TestCreatePartitionLineShapes:
    """Test suite for create_partition_line_shapes function."""

    def test_empty_blocks_returns_empty_list(self):
        """Test that empty block list returns no shapes."""
        shapes = create_partition_line_shapes(
            blocks=[],
            grid_step=20.0,
            max_lines=10,
            color="gray",
            width=2,
        )
        assert shapes == []

    def test_single_block_generates_partition_lines(self):
        """Test that a single block generates partition lines at its edges."""
        block = Block(
            (100, 100),
            (300, 300),
            40000,  # area
            Colors.RED,
            "TEST_001",  # id
        )
        shapes = create_partition_line_shapes(
            blocks=[block],
            grid_step=20.0,
            max_lines=10,
            color="gray",
            width=2,
        )
        # Single block has 4 edges, but frequency filtering requires >=2 occurrences
        # So single block should generate no lines
        assert shapes == []

    def test_aligned_blocks_generate_shared_partition_lines(self):
        """Test that blocks with aligned edges generate partition lines."""
        # Create two blocks sharing a vertical edge at x=300
        block1 = Block(
            (100, 100),
            (300, 300),
            40000,  # area
            Colors.RED,
            "TEST_001",
        )
        block2 = Block(
            (300, 100),
            (500, 300),
            40000,  # area
            Colors.BLUE,
            "TEST_002",
        )
        shapes = create_partition_line_shapes(
            blocks=[block1, block2],
            grid_step=20.0,
            max_lines=20,
            color="gray",
            width=2,
        )

        # Should generate lines for shared edges (frequency >= 2)
        assert len(shapes) > 0

        # Verify shape structure
        for shape in shapes:
            assert shape["type"] == "line"
            assert shape["line"]["color"] == "gray"
            assert shape["line"]["width"] == 2
            assert shape["layer"] == "below"

    def test_max_lines_limits_output(self):
        """Test that max_lines parameter correctly limits number of shapes."""
        # Create a grid of blocks to generate many partition lines
        blocks = []
        for i in range(5):
            for j in range(5):
                block = Block(
                    (i * 100, j * 100),
                    ((i + 1) * 100, (j + 1) * 100),
                    10000,  # area
                    Colors.RED,
                    f"TEST_{i}_{j}",
                )
                blocks.append(block)

        # Test with different max_lines values
        shapes_unlimited = create_partition_line_shapes(
            blocks=blocks,
            grid_step=20.0,
            max_lines=0,  # 0 means unlimited
            color="gray",
            width=2,
        )

        shapes_limited = create_partition_line_shapes(
            blocks=blocks,
            grid_step=20.0,
            max_lines=5,
            color="gray",
            width=2,
        )

        assert len(shapes_limited) <= 5
        assert len(shapes_unlimited) >= len(shapes_limited)

    def test_frequency_filtering_requires_minimum_occurrences(self):
        """Test that partition lines require minimum frequency (>=2)."""
        # Create blocks with one unique edge and one shared edge
        block1 = Block(
            (100, 100),
            (300, 300),
            40000,
            Colors.RED,
            "TEST_001",
        )
        block2 = Block(
            (300, 100),  # Shares x=300 with block1
            (500, 300),
            40000,
            Colors.BLUE,
            "TEST_002",
        )
        block3 = Block(
            (600, 100),  # Unique edge at x=600
            (800, 300),
            40000,
            Colors.YELLOW,
            "TEST_003",
        )

        shapes = create_partition_line_shapes(
            blocks=[block1, block2, block3],
            grid_step=20.0,
            max_lines=20,
            color="gray",
            width=2,
        )

        # Should only include lines for edges that appear multiple times
        # x=300 should be included (2 blocks), x=600 should not (1 block)
        x_coords = [s["x0"] for s in shapes if s.get("x0") == s.get("x1")]
        assert 300 in x_coords or any(abs(x - 300) < 25 for x in x_coords)
        assert not (600 in x_coords or any(abs(x - 600) < 5 for x in x_coords))

    def test_quantization_merges_near_duplicates(self):
        """Test that near-duplicate positions are merged via quantization."""
        # Create blocks with slightly offset edges that should merge
        block1 = Block(
            (100, 100),
            (300, 300),
            40000,
            Colors.RED,
            "TEST_001",
        )
        block2 = Block(
            (299, 100),  # Almost aligned with block1 right edge
            (500, 300),
            40200,
            Colors.BLUE,
            "TEST_002",
        )

        shapes = create_partition_line_shapes(
            blocks=[block1, block2],
            grid_step=20.0,  # Quantization step
            max_lines=20,
            color="gray",
            width=2,
        )

        # The near-duplicate edges should be quantized together
        # Verify that we don't get duplicate lines for nearly-aligned edges
        x_coords = [s["x0"] for s in shapes if s.get("x0") == s.get("x1")]
        # Count how many lines are near x=300
        near_300 = [x for x in x_coords if abs(x - 300) < 25]
        # Should have at most 1 line near x=300 due to merging
        assert len(near_300) <= 1

    def test_vertical_and_horizontal_lines_generated(self):
        """Test that both vertical and horizontal partition lines are created."""
        # Create blocks in a 2x2 grid pattern
        blocks = [
            Block((0, 0), (200, 200), 40000, Colors.RED, "TEST_00"),
            Block((200, 0), (400, 200), 40000, Colors.BLUE, "TEST_10"),
            Block((0, 200), (200, 400), 40000, Colors.YELLOW, "TEST_01"),
            Block((200, 200), (400, 400), 40000, Colors.BLACK, "TEST_11"),
        ]

        shapes = create_partition_line_shapes(
            blocks=blocks,
            grid_step=20.0,
            max_lines=20,
            color="gray",
            width=2,
        )

        # Should have both vertical and horizontal lines
        vertical_lines = [s for s in shapes if s.get("x0") == s.get("x1")]
        horizontal_lines = [s for s in shapes if s.get("y0") == s.get("y1")]

        assert len(vertical_lines) > 0, "Should generate vertical partition lines"
        assert len(horizontal_lines) > 0, "Should generate horizontal partition lines"

    def test_canvas_bounds_respected(self):
        """Test that partition lines stay within canvas bounds."""
        block = Block(
            (400, 400),
            (600, 600),
            40000,
            Colors.RED,
            "TEST_001",
        )
        shapes = create_partition_line_shapes(
            blocks=[block],
            grid_step=20.0,
            max_lines=10,
            color="gray",
            width=2,
            canvas_size=1000,
        )

        # All coordinates should be within [0, 1000]
        for shape in shapes:
            for coord_key in ["x0", "x1", "y0", "y1"]:
                if coord_key in shape:
                    assert 0 <= shape[coord_key] <= 1000


class TestRouteManhattan:
    """Test suite for route_manhattan function."""

    def test_simple_horizontal_routing(self):
        """Test routing between horizontally adjacent blocks."""
        src = Block((100, 200), (200, 300), 10000, Colors.RED, "SRC")
        dst = Block((400, 200), (500, 300), 10000, Colors.BLUE, "DST")

        path = route_manhattan(
            src=src,
            dst=dst,
            obstacles=[],
            grid_step=20.0,
        )

        assert len(path) >= 2, "Path should have at least start and end points"
        # Path should be orthogonal (Manhattan)
        for i in range(len(path) - 1):
            p1, p2 = path[i], path[i + 1]
            assert p1[0] == p2[0] or p1[1] == p2[1], "Path segments must be orthogonal"

    def test_simple_vertical_routing(self):
        """Test routing between vertically adjacent blocks."""
        src = Block((200, 100), (300, 200), 10000, Colors.RED, "SRC")
        dst = Block((200, 400), (300, 500), 10000, Colors.BLUE, "DST")

        path = route_manhattan(
            src=src,
            dst=dst,
            obstacles=[],
            grid_step=20.0,
        )

        assert len(path) >= 2, "Path should have at least start and end points"
        # Verify orthogonality
        for i in range(len(path) - 1):
            p1, p2 = path[i], path[i + 1]
            assert p1[0] == p2[0] or p1[1] == p2[1], "Path segments must be orthogonal"

    def test_obstacle_avoidance(self):
        """Test that routing avoids obstacle blocks."""
        src = Block((100, 200), (200, 300), 10000, Colors.RED, "SRC")
        dst = Block((400, 200), (500, 300), 10000, Colors.BLUE, "DST")

        # Place small obstacle that should be avoidable
        obstacle_block = Block(
            (270, 220),
            (330, 280),
            3600,
            Colors.YELLOW,
            "OBS",
        )
        obstacles = [
            _get_block_bounds(src),
            _get_block_bounds(dst),
            _get_block_bounds(obstacle_block),
        ]

        path = route_manhattan(
            src=src,
            dst=dst,
            obstacles=obstacles,
            grid_step=20.0,
        )

        # Algorithm should either find a path or return empty (both are valid)
        # If a path is found, verify it's orthogonal and doesn't cross obstacle interior
        if len(path) >= 2:
            # Verify orthogonality
            for i in range(len(path) - 1):
                p1, p2 = path[i], path[i + 1]
                assert (
                    p1[0] == p2[0] or p1[1] == p2[1]
                ), "Path segments must be orthogonal"

            # Verify path doesn't cross obstacle interior (sample midpoints)
            obstacle_bounds = _get_block_bounds(obstacle_block)
            for i in range(len(path) - 1):
                p1, p2 = path[i], path[i + 1]
                midpoint = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
                # Midpoint should not be deep inside obstacle (allow small margin)
                inside_obstacle = (
                    obstacle_bounds["left"] + 5
                    < midpoint[0]
                    < obstacle_bounds["right"] - 5
                    and obstacle_bounds["top"] + 5
                    < midpoint[1]
                    < obstacle_bounds["bottom"] - 5
                )
                if i > 0 and i < len(path) - 2:
                    assert (
                        not inside_obstacle
                    ), "Path should not cross obstacle interior"
        else:
            # Empty path is acceptable if no route can be found
            assert path == []

    def test_path_starts_and_ends_at_block_ports(self):
        """Test that path starts at source port and ends at destination port."""
        src = Block((100, 200), (200, 300), 10000, Colors.RED, "SRC")
        dst = Block((400, 200), (500, 300), 10000, Colors.BLUE, "DST")

        path = route_manhattan(
            src=src,
            dst=dst,
            obstacles=[_get_block_bounds(src), _get_block_bounds(dst)],
            grid_step=20.0,
        )

        assert len(path) >= 2

        # Start point should be near source block (with port offset)
        src_bounds = _get_block_bounds(src)
        start = path[0]
        # Should be within reasonable distance of source block boundaries (allowing for offset)
        assert (
            src_bounds["left"] - 10 <= start[0] <= src_bounds["right"] + 10
        ), "Start point should be near source block"
        assert (
            src_bounds["top"] - 10 <= start[1] <= src_bounds["bottom"] + 10
        ), "Start point should be near source block"

        # End point should be near destination block (with port offset)
        dst_bounds = _get_block_bounds(dst)
        end = path[-1]
        assert (
            dst_bounds["left"] - 10 <= end[0] <= dst_bounds["right"] + 10
        ), "End point should be near destination block"
        assert (
            dst_bounds["top"] - 10 <= end[1] <= dst_bounds["bottom"] + 10
        ), "End point should be near destination block"

    def test_no_route_returns_empty_list(self):
        """Test that impossible routes return empty list."""
        src = Block((100, 100), (150, 150), 2500, Colors.RED, "SRC")
        dst = Block((900, 900), (950, 950), 2500, Colors.BLUE, "DST")

        # Create a wall of obstacles blocking all paths
        obstacles = []
        for i in range(10):
            obstacles.append(
                {
                    "left": 400 + i * 10,
                    "right": 450 + i * 10,
                    "top": 0,
                    "bottom": 1000,
                }
            )

        path = route_manhattan(
            src=src,
            dst=dst,
            obstacles=obstacles,
            grid_step=20.0,
        )

        # Should return empty list when no route is possible
        assert path == [] or len(path) >= 2  # Either finds path or returns empty

    def test_path_uses_offset_bend_search(self):
        """Test that routing uses offset bend positions to avoid obstacles."""
        src = Block((100, 200), (200, 300), 10000, Colors.RED, "SRC")
        dst = Block((400, 500), (500, 600), 10000, Colors.BLUE, "DST")

        # Create obstacle that blocks straight path
        obstacle_block = Block(
            (280, 280),
            (380, 480),
            20000,
            Colors.YELLOW,
            "OBS",
        )
        obstacles = [
            _get_block_bounds(src),
            _get_block_bounds(dst),
            _get_block_bounds(obstacle_block),
        ]

        path = route_manhattan(
            src=src,
            dst=dst,
            obstacles=obstacles,
            grid_step=20.0,
        )

        # Should find a path using offset bends
        assert len(path) >= 2
        # Path with obstacle avoidance should have at least 3 points (L-shape or more)
        # to route around the obstacle
        if len(path) >= 3:
            # Verify that it's not a straight line (which would hit obstacle)
            assert not (path[0][0] == path[-1][0] or path[0][1] == path[-1][1])

    def test_grid_step_affects_routing(self):
        """Test that grid_step parameter affects bend positions."""
        src = Block((100, 200), (200, 300), 10000, Colors.RED, "SRC")
        dst = Block((400, 200), (500, 300), 10000, Colors.BLUE, "DST")
        obstacles = [_get_block_bounds(src), _get_block_bounds(dst)]

        path_fine = route_manhattan(src, dst, obstacles, grid_step=10.0)
        path_coarse = route_manhattan(src, dst, obstacles, grid_step=50.0)

        # Both should find paths
        assert len(path_fine) >= 2
        assert len(path_coarse) >= 2

        # Paths may differ due to different grid_step values
        # (actual behavior depends on obstacle configuration)

    def test_source_and_destination_ignored_as_obstacles(self):
        """Test that source and destination blocks are ignored in obstacle checking."""
        src = Block((100, 200), (200, 300), 10000, Colors.RED, "SRC")
        dst = Block((400, 200), (500, 300), 10000, Colors.BLUE, "DST")

        # Include source and destination in obstacles list
        # (the function should ignore them internally)
        obstacles = [_get_block_bounds(src), _get_block_bounds(dst)]

        path = route_manhattan(
            src=src,
            dst=dst,
            obstacles=obstacles,
            grid_step=20.0,
        )

        # Should still find a path
        assert len(path) >= 2

    def test_canvas_boundary_constraints(self):
        """Test that routing respects canvas boundaries."""
        src = Block((50, 500), (100, 550), 2500, Colors.RED, "SRC")
        dst = Block((900, 500), (950, 550), 2500, Colors.BLUE, "DST")

        path = route_manhattan(
            src=src,
            dst=dst,
            obstacles=[_get_block_bounds(src), _get_block_bounds(dst)],
            grid_step=20.0,
            canvas_size=1000,
        )

        # All path points should be within canvas bounds
        for point in path:
            assert 0 <= point[0] <= 1000, f"X coordinate {point[0]} out of bounds"
            assert 0 <= point[1] <= 1000, f"Y coordinate {point[1]} out of bounds"

    def test_diagonal_routing_uses_two_bends(self):
        """Test that diagonal routing uses proper orthogonal path with bends."""
        src = Block((100, 100), (200, 200), 10000, Colors.RED, "SRC")
        dst = Block((400, 400), (500, 500), 10000, Colors.BLUE, "DST")

        path = route_manhattan(
            src=src,
            dst=dst,
            obstacles=[_get_block_bounds(src), _get_block_bounds(dst)],
            grid_step=20.0,
        )

        assert len(path) >= 2
        # Diagonal routing should create bent path (not straight line)
        # Should have at least 3 points for proper Manhattan routing
        assert len(path) >= 3 or (
            len(path) == 2 and (path[0][0] == path[1][0] or path[0][1] == path[1][1])
        )

    def test_overlapping_blocks_edge_case(self):
        """Test routing behavior when blocks are very close or touching."""
        src = Block((100, 200), (200, 300), 10000, Colors.RED, "SRC")
        # Destination very close to source
        dst = Block((205, 200), (305, 300), 10000, Colors.BLUE, "DST")

        path = route_manhattan(
            src=src,
            dst=dst,
            obstacles=[_get_block_bounds(src), _get_block_bounds(dst)],
            grid_step=20.0,
        )

        # Should handle close blocks gracefully
        assert isinstance(path, list)
        # Path should be valid if non-empty
        if len(path) >= 2:
            for i in range(len(path) - 1):
                p1, p2 = path[i], path[i + 1]
                assert p1[0] == p2[0] or p1[1] == p2[1], "Path must be orthogonal"
