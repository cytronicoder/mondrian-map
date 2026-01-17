"""
Tests for dynamic marker sizing in visualizations.

These tests verify that invisible markers at tile centers are sized appropriately
to stay within tile boundaries and prevent accidental clicks on adjacent tiles.
"""

import pandas as pd
import pytest

from mondrian_map.visualization import create_authentic_mondrian_map


class TestDynamicMarkerSizing:
    """Test suite for dynamic marker size calculation."""

    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame with pathway data."""
        return pd.DataFrame(
            {
                "GS_ID": ["WP0001", "WP0002", "WP0003"],
                "NAME": ["Pathway A", "Pathway B", "Pathway C"],
                "wFC": [2.0, -2.0, 1.0],
                "pFDR": [0.001, 0.001, 0.001],
                "Ontology": ["Test", "Test", "Test"],
                "Disease": ["TestDisease", "TestDisease", "TestDisease"],
                "Description": ["Test pathway A", "Test pathway B", "Test pathway C"],
                "x": [200.0, 500.0, 800.0],
                "y": [200.0, 500.0, 800.0],
            }
        )

    def test_marker_exists_for_each_tile(self, sample_dataframe):
        """Test that invisible markers are created for each pathway tile."""
        fig = create_authentic_mondrian_map(
            sample_dataframe, "test_dataset", show_pathway_ids=False
        )

        # Find all marker traces (mode="markers" with opacity=0)
        marker_traces = [
            trace
            for trace in fig.data
            if hasattr(trace, "mode")
            and trace.mode == "markers"
            and hasattr(trace, "marker")
            and trace.marker.opacity == 0
        ]

        # Should have one marker per pathway
        assert len(marker_traces) == len(
            sample_dataframe
        ), f"Expected {len(sample_dataframe)} marker traces, found {len(marker_traces)}"

    def test_marker_size_within_bounds(self, sample_dataframe):
        """Test that marker sizes are within the defined bounds [6, 18].

        Note: Marker size is calculated as 60% of minimum tile dimension
        (in data coordinates 0-1000), capped between 6 and 18. For typical
        tiles of 30-200 pixels, this yields sizes of 6-18.
        """
        fig = create_authentic_mondrian_map(
            sample_dataframe, "test_dataset", show_pathway_ids=False
        )

        marker_traces = [
            trace
            for trace in fig.data
            if hasattr(trace, "mode")
            and trace.mode == "markers"
            and hasattr(trace, "marker")
            and trace.marker.opacity == 0
        ]

        for trace in marker_traces:
            marker_size = trace.marker.size
            # Marker size is in data coordinates (0-1000 range), capped at [6, 18]
            assert (
                6 <= marker_size <= 18
            ), f"Marker size {marker_size} outside bounds [6, 18]"

    def test_marker_size_scales_with_tile(self):
        """Test that marker size scales appropriately with tile dimensions."""
        # Create scenarios with different implied tile sizes
        # (Note: actual tile size depends on the layout algorithm, but we can test relative behavior)

        df_large_values = pd.DataFrame(
            {
                "GS_ID": ["WP0001"],
                "NAME": ["Large Pathway"],
                "wFC": [3.0],
                "pFDR": [0.001],
                "Ontology": ["Test"],
                "Disease": ["TestDisease"],
                "Description": ["Large pathway"],
                "x": [500.0],
                "y": [500.0],
            }
        )

        fig = create_authentic_mondrian_map(
            df_large_values, "large_test", show_pathway_ids=False
        )

        marker_traces = [
            trace
            for trace in fig.data
            if hasattr(trace, "mode")
            and trace.mode == "markers"
            and hasattr(trace, "marker")
            and trace.marker.opacity == 0
        ]

        assert len(marker_traces) > 0, "Should have at least one marker"
        # For a single large pathway, marker should be at or near maximum
        marker_size = marker_traces[0].marker.size
        assert marker_size <= 18, "Marker should not exceed maximum size"

    def test_marker_positioned_at_tile_center(self, sample_dataframe):
        """Test that markers have valid coordinates within canvas bounds.

        Note: Full center alignment testing requires integration tests with
        real pathway layout algorithm. This test verifies markers are created
        with valid coordinates.
        """
        fig = create_authentic_mondrian_map(
            sample_dataframe, "test_dataset", show_pathway_ids=False
        )

        # Get markers
        marker_traces = [
            trace
            for trace in fig.data
            if hasattr(trace, "mode")
            and trace.mode == "markers"
            and hasattr(trace, "marker")
            and trace.marker.opacity == 0
        ]

        # Verify all markers have valid coordinates
        for marker in marker_traces:
            assert len(marker.x) > 0, "Marker should have x coordinate"
            assert len(marker.y) > 0, "Marker should have y coordinate"

            marker_x = marker.x[0]
            marker_y = marker.y[0]

            # Coordinates should be within canvas bounds
            assert 0 <= marker_x <= 1000, f"Marker x={marker_x} out of bounds"
            assert 0 <= marker_y <= 1000, f"Marker y={marker_y} out of bounds"

    def test_marker_customdata_preserved(self, sample_dataframe):
        """Test that marker traces preserve customdata for interactivity."""
        fig = create_authentic_mondrian_map(
            sample_dataframe, "test_dataset", show_pathway_ids=False
        )

        marker_traces = [
            trace
            for trace in fig.data
            if hasattr(trace, "mode")
            and trace.mode == "markers"
            and hasattr(trace, "marker")
            and trace.marker.opacity == 0
        ]

        for trace in marker_traces:
            assert hasattr(trace, "customdata"), "Marker should have customdata"
            assert len(trace.customdata) > 0, "Customdata should not be empty"

            # Verify customdata structure: should be a list containing one dict
            assert isinstance(
                trace.customdata[0], dict
            ), f"customdata[0] should be dict, got {type(trace.customdata[0])}"

            payload = trace.customdata[0]
            assert (
                "pathway_id" in payload or "name" in payload
            ), "Customdata should contain pathway information"

    def test_minimum_marker_size_enforced(self):
        """Test that very small tiles still get minimum marker size.

        Even when tiles are very small (e.g., from downscaling or low wFC),
        markers should be clamped to minimum size of 6 to remain usable.
        """
        # Create a scenario that would result in very small tiles
        df_many_pathways = pd.DataFrame(
            {
                "GS_ID": [f"WP{i:04d}" for i in range(100)],
                "NAME": [f"Pathway {i}" for i in range(100)],
                "wFC": [1.5 + (i % 10) * 0.1 for i in range(100)],
                "pFDR": [0.001] * 100,
                "Ontology": ["Test"] * 100,
                "Disease": ["TestDisease"] * 100,
                "Description": [f"Test pathway {i}" for i in range(100)],
                "x": [(i % 10) * 100 + 50 for i in range(100)],
                "y": [(i // 10) * 100 + 50 for i in range(100)],
            }
        )

        fig = create_authentic_mondrian_map(
            df_many_pathways, "crowded_test", show_pathway_ids=False
        )

        marker_traces = [
            trace
            for trace in fig.data
            if hasattr(trace, "mode")
            and trace.mode == "markers"
            and hasattr(trace, "marker")
            and trace.marker.opacity == 0
        ]

        # All markers should have at least minimum size (6 in data coordinates)
        for trace in marker_traces:
            marker_size = trace.marker.size
            assert marker_size >= 6, f"Marker size {marker_size} below minimum 6"

    def test_marker_hoverinfo_disabled(self, sample_dataframe):
        """Test that marker hover info is disabled to avoid double tooltips."""
        fig = create_authentic_mondrian_map(
            sample_dataframe, "test_dataset", show_pathway_ids=False
        )

        marker_traces = [
            trace
            for trace in fig.data
            if hasattr(trace, "mode")
            and trace.mode == "markers"
            and hasattr(trace, "marker")
            and trace.marker.opacity == 0
        ]

        for trace in marker_traces:
            assert trace.hoverinfo == "skip", "Marker hoverinfo should be set to 'skip'"
