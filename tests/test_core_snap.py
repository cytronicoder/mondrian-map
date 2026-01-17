"""
Tests for snap_rect_to_grid edge cases and minimum size validation.

These tests ensure that rectangle snapping maintains minimum dimensions
and handles edge cases where both corners might snap to the same grid point.
"""

import pytest

from mondrian_map.core import snap_rect_to_grid


class TestSnapRectToGrid:
    """Test suite for snap_rect_to_grid function."""

    def test_snap_prevents_zero_width(self):
        """Test that snapping prevents zero width when corners snap together."""
        # Rectangle smaller than grid size
        rect = [(10.1, 50.0), (10.9, 100.0)]
        snapped = snap_rect_to_grid(rect, grid=20, bounds=(0, 1000), min_size=20)
        
        (x0, y0), (x1, y1) = snapped
        width = abs(x1 - x0)
        height = abs(y1 - y0)
        
        assert width >= 20, f"Width {width} is less than minimum 20"
        assert height >= 20, f"Height {height} is less than minimum 20"

    def test_snap_prevents_zero_height(self):
        """Test that snapping prevents zero height when corners snap together."""
        # Rectangle smaller than grid size
        rect = [(50.0, 10.1), (100.0, 10.9)]
        snapped = snap_rect_to_grid(rect, grid=20, bounds=(0, 1000), min_size=20)
        
        (x0, y0), (x1, y1) = snapped
        width = abs(x1 - x0)
        height = abs(y1 - y0)
        
        assert width >= 20, f"Width {width} is less than minimum 20"
        assert height >= 20, f"Height {height} is less than minimum 20"

    def test_snap_handles_both_corners_same_point(self):
        """Test edge case where both corners snap to same grid point."""
        # Tiny rectangle that would snap to single point
        rect = [(100.1, 200.1), (100.9, 200.9)]
        snapped = snap_rect_to_grid(rect, grid=20, bounds=(0, 1000), min_size=20)
        
        (x0, y0), (x1, y1) = snapped
        width = abs(x1 - x0)
        height = abs(y1 - y0)
        
        assert width >= 20, "Width should be at least minimum size"
        assert height >= 20, "Height should be at least minimum size"

    def test_snap_respects_canvas_bounds(self):
        """Test that snapping respects canvas boundaries."""
        # Rectangle near left edge
        rect = [(5.0, 100.0), (8.0, 150.0)]
        snapped = snap_rect_to_grid(rect, grid=20, bounds=(0, 1000), min_size=20)
        
        (x0, y0), (x1, y1) = snapped
        
        assert x0 >= 0, "x0 should be within bounds"
        assert x1 <= 1000, "x1 should be within bounds"
        assert y0 >= 0, "y0 should be within bounds"
        assert y1 <= 1000, "y1 should be within bounds"

    def test_snap_near_right_edge(self):
        """Test snapping behavior near right canvas edge."""
        # Rectangle near right edge
        rect = [(990.0, 100.0), (995.0, 150.0)]
        snapped = snap_rect_to_grid(rect, grid=20, bounds=(0, 1000), min_size=20)
        
        (x0, y0), (x1, y1) = snapped
        width = abs(x1 - x0)
        
        assert x1 <= 1000, "x1 should not exceed canvas bounds"
        assert width >= 20, "Width should be at least minimum size even at edge"

    def test_snap_near_bottom_edge(self):
        """Test snapping behavior near bottom canvas edge."""
        # Rectangle near bottom edge
        rect = [(100.0, 990.0), (150.0, 995.0)]
        snapped = snap_rect_to_grid(rect, grid=20, bounds=(0, 1000), min_size=20)
        
        (x0, y0), (x1, y1) = snapped
        height = abs(y1 - y0)
        
        assert y1 <= 1000, "y1 should not exceed canvas bounds"
        assert height >= 20, "Height should be at least minimum size even at edge"

    def test_snap_large_rectangle_unchanged(self):
        """Test that large rectangles snap correctly without minimum size issues."""
        # Large rectangle
        rect = [(100.0, 100.0), (300.0, 300.0)]
        snapped = snap_rect_to_grid(rect, grid=20, bounds=(0, 1000), min_size=20)
        
        (x0, y0), (x1, y1) = snapped
        
        # Should snap to grid points
        assert x0 % 20 == 0, "x0 should be on grid"
        assert x1 % 20 == 0, "x1 should be on grid"
        assert y0 % 20 == 0, "y0 should be on grid"
        assert y1 % 20 == 0, "y1 should be on grid"

    def test_snap_custom_min_size(self):
        """Test that custom minimum size is respected."""
        rect = [(10.1, 10.1), (10.9, 10.9)]
        snapped = snap_rect_to_grid(rect, grid=20, bounds=(0, 1000), min_size=40)
        
        (x0, y0), (x1, y1) = snapped
        width = abs(x1 - x0)
        height = abs(y1 - y0)
        
        assert width >= 40, f"Width {width} should be at least custom min_size 40"
        assert height >= 40, f"Height {height} should be at least custom min_size 40"

    def test_snap_returns_valid_coordinates(self):
        """Test that snapped coordinates are always sorted correctly."""
        rect = [(200.5, 300.5), (100.5, 150.5)]  # Reversed coordinates
        snapped = snap_rect_to_grid(rect, grid=20, bounds=(0, 1000), min_size=20)
        
        (x0, y0), (x1, y1) = snapped
        
        assert x0 <= x1, "x0 should be less than or equal to x1"
        assert y0 <= y1, "y0 should be less than or equal to y1"
