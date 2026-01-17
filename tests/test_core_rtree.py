"""
Tests for R-tree spatial index optimization in overlap resolution.

These tests verify that the spatial index correctly identifies and resolves
overlaps, improving performance from O(n²) to O(n log n).

Boundary Convention
-------------------
GridSystem is initialized with dimensions 1001x1001 (matching visualization.py),
but snap_rect_to_grid uses bounds=(0, 1000) by default. This means:
- Grid dimensions: 1001x1001 pixels (width/height including endpoints)
- Coordinate space: [0, 1000] (inclusive range for rectangle positions)
- This allows snapping to grid lines at 0, 20, 40, ..., 980, 1000

The extra dimension in GridSystem (1001 vs 1000) accommodates the endpoint
for proper grid line calculation (1000/20 = 50 grid divisions).
"""

import random

from mondrian_map.core import GridSystem, count_overlaps, rect_intersects


class TestSpatialIndexOverlapResolution:
    """Test suite for R-tree optimized overlap resolution."""

    def test_avoid_overlap_with_small_dataset(self):
        """Test overlap avoidance with a small number of rectangles."""
        # Use 1001x1001 to match visualization.py GridSystem initialization
        gs = GridSystem(1001, 1001, 20, 20)

        # 10 points with uniform areas
        random.seed(42)
        points = [
            (random.uniform(100, 900), random.uniform(100, 900)) for _ in range(10)
        ]
        areas = [5000 for _ in points]

        rects = gs.plot_points_fill_blocks(
            points, areas, avoid_overlap=True, padding=5, nudge=True, snap_to_grid=True
        )

        assert len(rects) == len(
            points
        ), "Should return same number of rectangles as points"
        overlaps = count_overlaps(rects, padding=5)
        assert overlaps == 0, f"Expected 0 overlaps, found {overlaps}"

    def test_avoid_overlap_with_medium_dataset(self):
        """Test overlap avoidance with medium-sized dataset (50 rectangles)."""
        gs = GridSystem(1001, 1001, 20, 20)

        random.seed(123)
        points = [
            (random.uniform(100, 900), random.uniform(100, 900)) for _ in range(50)
        ]
        areas = [3000 for _ in points]

        rects = gs.plot_points_fill_blocks(
            points, areas, avoid_overlap=True, padding=5, nudge=True, snap_to_grid=True
        )

        assert len(rects) == len(
            points
        ), "Should return same number of rectangles as points"
        overlaps = count_overlaps(rects, padding=5)
        assert overlaps == 0, f"Expected 0 overlaps after nudging, found {overlaps}"

    def test_avoid_overlap_with_varying_sizes(self):
        """Test overlap avoidance with varying rectangle sizes."""
        gs = GridSystem(1001, 1001, 20, 20)

        random.seed(456)
        points = [
            (random.uniform(100, 900), random.uniform(100, 900)) for _ in range(20)
        ]
        # Varying areas from small to large
        areas = [random.uniform(1000, 8000) for _ in points]

        rects = gs.plot_points_fill_blocks(
            points, areas, avoid_overlap=True, padding=5, nudge=True, snap_to_grid=True
        )

        assert len(rects) == len(points)
        overlaps = count_overlaps(rects, padding=5)
        assert (
            overlaps == 0
        ), f"Expected 0 overlaps with varying sizes, found {overlaps}"

    def test_no_overlap_mode_allows_intersections(self):
        """Test that avoid_overlap=False allows overlaps (baseline behavior)."""
        gs = GridSystem(1001, 1001, 20, 20)

        # Place rectangles very close together
        # With area=8000, approximate square dimensions are ~89x89 pixels
        # Points separated by only 20 pixels should create significant overlaps
        points = [(500, 500), (520, 520), (540, 540)]
        areas = [8000, 8000, 8000]

        rects = gs.plot_points_fill_blocks(points, areas, avoid_overlap=False)

        assert len(rects) == len(points)
        # With large areas (~89x89) and close spacing (20px), all 3 rectangles
        # should overlap with each other, yielding 3 pairwise overlaps
        overlaps = count_overlaps(rects, padding=0)
        assert overlaps >= 3, f"Expected 3 overlaps (all pairs) but got {overlaps}"

    def test_nudging_resolves_initial_overlaps(self):
        """Test that nudging successfully resolves overlapping positions."""
        gs = GridSystem(1001, 1001, 20, 20)

        # Create intentionally overlapping scenario
        points = [(500, 500), (505, 505), (510, 510)]
        areas = [4000, 4000, 4000]

        rects = gs.plot_points_fill_blocks(
            points, areas, avoid_overlap=True, padding=5, nudge=True, snap_to_grid=True
        )

        assert len(rects) == len(points)
        overlaps = count_overlaps(rects, padding=5)
        assert overlaps == 0, "Nudging should resolve all overlaps"

    def test_spatial_index_handles_edge_cases(self):
        """Test spatial index with rectangles near canvas edges."""
        # GridSystem dimensions are 1001x1001 (matching visualization.py)
        # Canvas coordinates range from 0 to 1000 (inclusive)
        gs = GridSystem(1001, 1001, 20, 20)

        # Points near edges
        points = [
            (50, 50),  # Near top-left
            (950, 50),  # Near top-right
            (50, 950),  # Near bottom-left
            (950, 950),  # Near bottom-right
            (500, 500),  # Center
        ]
        areas = [3000 for _ in points]

        rects = gs.plot_points_fill_blocks(
            points, areas, avoid_overlap=True, padding=5, nudge=True, snap_to_grid=True
        )

        assert len(rects) == len(points)

        # Verify all rectangles are within bounds [0, 1000]
        # Note: snap_rect_to_grid uses bounds=(0, 1000) by default
        for rect in rects:
            (x0, y0), (x1, y1) = rect
            assert (
                0 <= x0 <= 1000 and 0 <= x1 <= 1000
            ), f"x coords out of bounds: {rect}"
            assert (
                0 <= y0 <= 1000 and 0 <= y1 <= 1000
            ), f"y coords out of bounds: {rect}"

    def test_rect_intersects_with_padding(self):
        """Test rect_intersects utility function with padding.

        Note: rect_intersects uses <= comparisons, meaning rectangles that
        share only an edge or corner point are NOT considered overlapping.
        Only strict interior overlap is detected. This is the intended behavior
        for the Mondrian map layout algorithm.
        """
        rect1 = [(100, 100), (200, 200)]
        rect2 = [(200, 200), (300, 300)]

        # Without padding, edge-touching rectangles don't overlap (strict intersection)
        # rect1 ends at x=200, rect2 starts at x=200 → maxx1(200) <= minx2(200) → no overlap
        assert not rect_intersects(
            rect1, rect2, padding=0
        ), "Edge-touching rectangles should not be considered overlapping"

        # With padding, they should be considered overlapping
        assert rect_intersects(
            rect1, rect2, padding=5
        ), "Padding should make edge-touching rectangles overlap"

    def test_count_overlaps_accuracy(self):
        """Test count_overlaps utility function."""
        # Non-overlapping rectangles
        rects = [
            [(100, 100), (150, 150)],
            [(200, 200), (250, 250)],
            [(300, 300), (350, 350)],
        ]
        assert count_overlaps(rects, padding=0) == 0

        # Overlapping rectangles
        rects_overlap = [
            [(100, 100), (200, 200)],
            [(150, 150), (250, 250)],  # Overlaps with first
        ]
        assert count_overlaps(rects_overlap, padding=0) == 1

    def test_scaling_reduces_overlaps(self):
        """Test that global scaling phase reduces overlaps compared to no scaling."""
        gs = GridSystem(1001, 1001, 20, 20)

        # Dense cluster of points
        random.seed(789)
        center_x, center_y = 500, 500
        points = [
            (center_x + random.uniform(-50, 50), center_y + random.uniform(-50, 50))
            for _ in range(15)
        ]
        areas = [4000 for _ in points]

        # Generate without overlap avoidance (baseline)
        rects_no_avoid = gs.plot_points_fill_blocks(points, areas, avoid_overlap=False)
        overlaps_baseline = count_overlaps(rects_no_avoid, padding=5)

        # Generate with scaling but no nudging
        rects_with_scaling = gs.plot_points_fill_blocks(
            points, areas, avoid_overlap=True, padding=5, nudge=False, snap_to_grid=True
        )
        overlaps_scaled = count_overlaps(rects_with_scaling, padding=5)

        assert len(rects_with_scaling) == len(points)
        # Scaling should reduce overlaps compared to no avoidance
        assert (
            overlaps_scaled < overlaps_baseline
        ), f"Scaling should reduce overlaps: {overlaps_scaled} vs baseline {overlaps_baseline}"
        # Note: nudge=True is needed to fully eliminate overlaps
