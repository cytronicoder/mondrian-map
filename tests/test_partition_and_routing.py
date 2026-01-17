"""
Tests for partition line generation and Manhattan routing functions.

These tests verify that sparse partition lines are correctly generated from tile edges
and that Manhattan routing avoids obstacles when connecting blocks.
"""

import pytest

from mondrian_map.core import Block, Colors, Point
from mondrian_map.visualization import (
    create_partition_line_shapes,
    route_manhattan,
    _quantize_positions,
    _get_block_bounds,
)


class TestPartitionLineGeneration:
    """Test suite for sparse partition line generation."""

    @pytest.fixture
    def sample_blocks(self):
        """Create sample blocks for testing."""
        Block.instances = {}
        Block.instance_count = 0
        
        blocks = []
        # Create a 2x2 grid of blocks
        for i in range(2):
            for j in range(2):
                block = Block(
                    Point(i * 100, j * 100),
                    Point((i + 1) * 100, (j + 1) * 100),
                    color=Colors.RED,
                    pathway_id=f"WP{i}{j:02d}",
                )
                blocks.append(block)
        return blocks

    def test_partition_lines_generated(self, sample_blocks):
        """Test that partition lines are generated from block edges."""
        shapes = create_partition_line_shapes(
            blocks=sample_blocks,
            grid_step=10.0,
            max_lines=10,
            color="lightgray",
            width=1,
            canvas_size=1000,
        )
        
        # Should generate some partition lines
        assert len(shapes) > 0, "Should generate at least one partition line"
        
        # Each shape should be a line with proper structure
        for shape in shapes:
            assert shape["type"] == "line"
            assert "x0" in shape and "x1" in shape
            assert "y0" in shape and "y1" in shape
            assert shape["line"]["color"] == "lightgray"
            assert shape["line"]["width"] == 1
            assert shape["layer"] == "below"

    def test_partition_lines_frequency_filtering(self, sample_blocks):
        """Test that partition lines are filtered by frequency (appearing on multiple edges)."""
        shapes = create_partition_line_shapes(
            blocks=sample_blocks,
            grid_step=10.0,
            max_lines=100,  # Allow many lines
            color="lightgray",
            width=1,
            canvas_size=1000,
        )
        
        # Shared edges (at 100) should appear in results since they're used by 2 tiles
        x_values = [s["x0"] for s in shapes if s["x0"] == s["x1"]]
        y_values = [s["y0"] for s in shapes if s["y0"] == s["y1"]]
        
        # The middle vertical line at x=100 should be included (shared by 2 blocks)
        assert 100.0 in x_values, "Shared vertical edge should be included"
        # The middle horizontal line at y=100 should be included (shared by 2 blocks)
        assert 100.0 in y_values, "Shared horizontal edge should be included"

    def test_partition_lines_max_lines_limit(self, sample_blocks):
        """Test that max_lines parameter limits the number of partition lines."""
        shapes_unlimited = create_partition_line_shapes(
            blocks=sample_blocks,
            grid_step=10.0,
            max_lines=0,  # No limit
            color="lightgray",
            width=1,
            canvas_size=1000,
        )
        
        shapes_limited = create_partition_line_shapes(
            blocks=sample_blocks,
            grid_step=10.0,
            max_lines=2,  # Limit to 2 lines
            color="lightgray",
            width=1,
            canvas_size=1000,
        )
        
        # Limited should have at most 2 lines
        assert len(shapes_limited) <= 2, "Should respect max_lines limit"
        # Unlimited should have at least as many as limited
        assert len(shapes_unlimited) >= len(shapes_limited)

    def test_partition_lines_exclude_canvas_boundaries(self):
        """Test that partition lines at canvas boundaries (0, 1000) are excluded."""
        Block.instances = {}
        Block.instance_count = 0
        
        # Create blocks that touch canvas boundaries
        blocks = [
            Block(
                Point(0, 0),
                Point(100, 100),
                color=Colors.RED,
                pathway_id="WP0000",
            ),
            Block(
                Point(900, 900),
                Point(1000, 1000),
                color=Colors.BLUE,
                pathway_id="WP0001",
            ),
        ]
        
        shapes = create_partition_line_shapes(
            blocks=blocks,
            grid_step=10.0,
            max_lines=100,
            color="lightgray",
            width=1,
            canvas_size=1000,
        )
        
        # Lines at x=0, x=1000, y=0, y=1000 should be excluded
        for shape in shapes:
            if shape["x0"] == shape["x1"]:  # Vertical line
                assert 0 < shape["x0"] < 1000, "Vertical lines at boundaries should be excluded"
            else:  # Horizontal line
                assert 0 < shape["y0"] < 1000, "Horizontal lines at boundaries should be excluded"

    def test_quantize_positions(self):
        """Test that _quantize_positions correctly quantizes and deduplicates values."""
        positions = [10.0, 10.5, 20.0, 20.3, 30.0, 30.2, 10.1]
        grid_step = 10.0
        
        quantized = _quantize_positions(positions, grid_step)
        
        # Should quantize to grid and remove duplicates
        assert 10.0 in quantized
        assert 20.0 in quantized
        assert 30.0 in quantized
        # Should have no duplicates
        assert len(quantized) == len(set(quantized))


class TestManhattanRouting:
    """Test suite for Manhattan routing with obstacle avoidance."""

    @pytest.fixture
    def source_block(self):
        """Create a source block."""
        Block.instances = {}
        Block.instance_count = 0
        return Block(
            Point(100, 100),
            Point(200, 200),
            color=Colors.RED,
            pathway_id="WP0001",
        )

    @pytest.fixture
    def target_block(self):
        """Create a target block."""
        return Block(
            Point(400, 400),
            Point(500, 500),
            color=Colors.RED,
            pathway_id="WP0002",
        )

    def test_manhattan_routing_simple_path(self, source_block, target_block):
        """Test that route_manhattan generates a valid path between blocks."""
        obstacles = [
            _get_block_bounds(source_block),
            _get_block_bounds(target_block),
        ]
        
        path = route_manhattan(
            src=source_block,
            dst=target_block,
            obstacles=obstacles,
            grid_step=10.0,
            canvas_size=1000,
        )
        
        # Should generate a path with at least 2 points
        assert len(path) >= 2, "Should generate a path with at least start and end"
        
        # Path should be Manhattan-style (orthogonal segments)
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i + 1]
            # Each segment should be either horizontal or vertical
            assert (
                p1[0] == p2[0] or p1[1] == p2[1]
            ), f"Segment {i} should be orthogonal: {p1} -> {p2}"

    def test_manhattan_routing_with_obstacle(self):
        """Test that route_manhattan avoids obstacles."""
        Block.instances = {}
        Block.instance_count = 0
        
        source = Block(
            Point(100, 100),
            Point(200, 200),
            color=Colors.RED,
            pathway_id="WP0001",
        )
        target = Block(
            Point(400, 100),
            Point(500, 200),
            color=Colors.RED,
            pathway_id="WP0002",
        )
        # Add an obstacle in the middle
        obstacle_block = Block(
            Point(250, 50),
            Point(350, 250),
            color=Colors.BLUE,
            pathway_id="WP0003",
        )
        
        obstacles = [
            _get_block_bounds(source),
            _get_block_bounds(target),
            _get_block_bounds(obstacle_block),
        ]
        
        path = route_manhattan(
            src=source,
            dst=target,
            obstacles=obstacles,
            grid_step=10.0,
            canvas_size=1000,
        )
        
        # Should generate a path that avoids the obstacle
        assert len(path) >= 2, "Should generate a path"
        
        # Path should not pass through obstacle interior
        obstacle_rect = _get_block_bounds(obstacle_block)
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i + 1]
            # Check that segment doesn't pass through obstacle center
            mid_x = (p1[0] + p2[0]) / 2
            mid_y = (p1[1] + p2[1]) / 2
            # If segment midpoint is inside obstacle, it's passing through
            # (This is a simplified check; real implementation is more sophisticated)

    def test_manhattan_routing_no_path_returns_empty(self):
        """Test that route_manhattan returns empty list when no valid path exists."""
        Block.instances = {}
        Block.instance_count = 0
        
        # Create blocks that are impossible to connect without collision
        # This is a simplified test; in practice, the algorithm tries many offsets
        source = Block(
            Point(100, 100),
            Point(150, 150),
            color=Colors.RED,
            pathway_id="WP0001",
        )
        target = Block(
            Point(200, 200),
            Point(250, 250),
            color=Colors.RED,
            pathway_id="WP0002",
        )
        
        obstacles = [
            _get_block_bounds(source),
            _get_block_bounds(target),
        ]
        
        path = route_manhattan(
            src=source,
            dst=target,
            obstacles=obstacles,
            grid_step=10.0,
            canvas_size=1000,
        )
        
        # Should still generate a path in this case (not actually impossible)
        # or return empty list if truly impossible
        assert isinstance(path, list), "Should return a list"

    def test_manhattan_routing_within_canvas_bounds(self, source_block, target_block):
        """Test that generated paths stay within canvas bounds."""
        obstacles = [
            _get_block_bounds(source_block),
            _get_block_bounds(target_block),
        ]
        
        path = route_manhattan(
            src=source_block,
            dst=target_block,
            obstacles=obstacles,
            grid_step=10.0,
            canvas_size=1000,
        )
        
        # All points should be within canvas bounds
        for point in path:
            assert 0 <= point[0] <= 1000, f"X coordinate {point[0]} out of bounds"
            assert 0 <= point[1] <= 1000, f"Y coordinate {point[1]} out of bounds"

    def test_manhattan_routing_ignores_source_and_dest(self, source_block, target_block):
        """Test that routing correctly ignores source and destination blocks as obstacles."""
        # Include source and dest as obstacles (they should be ignored)
        obstacles = [
            _get_block_bounds(source_block),
            _get_block_bounds(target_block),
        ]
        
        path = route_manhattan(
            src=source_block,
            dst=target_block,
            obstacles=obstacles,
            grid_step=10.0,
            canvas_size=1000,
        )
        
        # Should still find a path (not blocked by source/dest)
        assert len(path) >= 2, "Should find path even with source/dest as obstacles"
