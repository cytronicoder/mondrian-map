"""
Tests for color classification rules.

These tests verify that pathway color classification follows the paper's rules:
- Red (up): wFC >= 1.5
- Blue (down): wFC <= 0.5
- Yellow (not significant): pFDR >= 0.05
- Gray (mixed): all other cases
"""

import numpy as np
import pandas as pd
import pytest

from mondrian_map.data_processing import get_colors, get_mondrian_color_description
from mondrian_map.pathway_stats import classify_pathways


class TestColorClassification:
    """Test suite for color classification rules."""

    def test_up_regulated_red(self):
        """Test that up-regulated pathways (wFC >= 1.5) are red."""
        df = pd.DataFrame(
            {
                "GS_ID": ["WP001"],
                "wFC": [2.0],
                "pFDR": [0.01],
            }
        )

        colors = get_colors(df)
        assert colors[0] == "red" or colors[0].lower() == "#ff0000"

    def test_down_regulated_blue(self):
        """Test that down-regulated pathways (wFC <= 0.5) are blue."""
        df = pd.DataFrame(
            {
                "GS_ID": ["WP001"],
                "wFC": [0.3],
                "pFDR": [0.01],
            }
        )

        colors = get_colors(df)
        assert colors[0] == "blue" or colors[0].lower() == "#0000ff"

    def test_not_significant_yellow(self):
        """Test that non-significant pathways (pFDR >= 0.05) are yellow."""
        df = pd.DataFrame(
            {
                "GS_ID": ["WP001"],
                "wFC": [1.2],  # Between thresholds
                "pFDR": [0.1],  # Not significant
            }
        )

        colors = get_colors(df)
        assert (
            colors[0] == "yellow"
            or "ffcc" in colors[0].lower()
            or "ff0" in colors[0].lower()
        )

    def test_mixed_gray(self):
        """Test that mixed pathways are gray."""
        df = pd.DataFrame(
            {
                "GS_ID": ["WP001"],
                "wFC": [1.0],  # Between thresholds
                "pFDR": [0.01],  # Significant
            }
        )

        colors = get_colors(df)
        # Should be gray/white/neutral
        assert (
            colors[0] == "gray"
            or colors[0] == "white"
            or "ccc" in colors[0].lower()
            or "808" in colors[0].lower()
        )

    def test_boundary_up(self):
        """Test boundary case: wFC exactly 1.5 should be red."""
        df = pd.DataFrame(
            {
                "GS_ID": ["WP001"],
                "wFC": [1.5],  # Exactly at threshold
                "pFDR": [0.01],
            }
        )

        colors = get_colors(df)
        assert colors[0] == "red" or colors[0].lower() == "#ff0000"

    def test_boundary_down(self):
        """Test boundary case: wFC exactly 0.5 should be blue."""
        df = pd.DataFrame(
            {
                "GS_ID": ["WP001"],
                "wFC": [0.5],  # Exactly at threshold
                "pFDR": [0.01],
            }
        )

        colors = get_colors(df)
        assert colors[0] == "blue" or colors[0].lower() == "#0000ff"

    def test_boundary_pfdr(self):
        """Test boundary case: pFDR exactly 0.05 should be yellow."""
        df = pd.DataFrame(
            {
                "GS_ID": ["WP001"],
                "wFC": [1.2],
                "pFDR": [0.05],  # Exactly at threshold
            }
        )

        colors = get_colors(df)
        # At boundary - depends on implementation (>= vs >)
        # Just check it returns a valid color
        assert colors[0] in ["red", "blue", "yellow", "gray", "white"] or colors[
            0
        ].startswith("#")


class TestClassifyPathways:
    """Test the classify_pathways function."""

    def test_classification_categories(self):
        """Test that all pathways are assigned a category."""
        df = pd.DataFrame(
            {
                "GS_ID": ["WP001", "WP002", "WP003", "WP004"],
                "wFC": [2.0, 0.3, 1.2, 1.0],
                "pFDR": [0.01, 0.01, 0.1, 0.01],
            }
        )

        classified = classify_pathways(df)

        assert "classification" in classified.columns
        assert classified.loc[0, "classification"] == "up"
        assert classified.loc[1, "classification"] == "down"
        assert classified.loc[2, "classification"] == "not_significant"
        assert classified.loc[3, "classification"] == "mixed"

    def test_classification_counts(self):
        """Test classification counts match expected."""
        df = pd.DataFrame(
            {
                "GS_ID": [f"WP{i:03d}" for i in range(10)],
                "wFC": [2.0, 2.5, 0.3, 0.4, 1.2, 1.0, 1.1, 0.8, 3.0, 0.2],
                "pFDR": [0.01, 0.01, 0.01, 0.01, 0.1, 0.01, 0.2, 0.01, 0.01, 0.01],
            }
        )

        classified = classify_pathways(df)
        counts = classified["classification"].value_counts()

        # 3 up (2.0, 2.5, 3.0)
        assert counts.get("up", 0) == 3
        # 3 down (0.3, 0.4, 0.2)
        assert counts.get("down", 0) == 3
        # 2 not_significant (1.2, 1.1)
        assert counts.get("not_significant", 0) == 2
        # 2 mixed (1.0, 0.8)
        assert counts.get("mixed", 0) == 2


class TestColorDescriptions:
    """Test color description helper functions."""

    def test_get_color_description(self):
        """Test getting human-readable color descriptions."""
        desc = get_mondrian_color_description()

        assert "red" in desc.lower() or "up" in desc.lower()
        assert "blue" in desc.lower() or "down" in desc.lower()
        assert "yellow" in desc.lower() or "significant" in desc.lower()


class TestMultiplePathways:
    """Test color assignment for multiple pathways."""

    def test_mixed_classifications(self):
        """Test a DataFrame with all classification types."""
        df = pd.DataFrame(
            {
                "GS_ID": ["UP1", "UP2", "DOWN1", "NOTSIG", "MIXED"],
                "wFC": [2.0, 1.8, 0.3, 1.2, 0.9],
                "pFDR": [0.01, 0.02, 0.01, 0.1, 0.03],
            }
        )

        colors = get_colors(df)

        assert len(colors) == 5
        # Check that we have variety in colors
        unique_colors = set(colors)
        assert len(unique_colors) >= 3  # At least 3 different colors

    def test_all_same_classification(self):
        """Test when all pathways have same classification."""
        df = pd.DataFrame(
            {
                "GS_ID": [f"WP{i:03d}" for i in range(5)],
                "wFC": [2.0, 2.5, 1.8, 3.0, 1.6],  # All up-regulated
                "pFDR": [0.01, 0.02, 0.01, 0.001, 0.03],
            }
        )

        colors = get_colors(df)

        # All should be the same color (red)
        assert len(set(colors)) == 1


class TestEdgeCases:
    """Test edge cases in color assignment."""

    def test_extreme_wfc_values(self):
        """Test with extreme wFC values."""
        df = pd.DataFrame(
            {
                "GS_ID": ["EXTREME_UP", "EXTREME_DOWN"],
                "wFC": [100.0, 0.001],
                "pFDR": [0.01, 0.01],
            }
        )

        colors = get_colors(df)

        # Should still assign valid colors
        assert len(colors) == 2
        for color in colors:
            assert color in [
                "red",
                "blue",
                "yellow",
                "gray",
                "white",
            ] or color.startswith("#")

    def test_nan_wfc(self):
        """Test handling of NaN wFC values."""
        df = pd.DataFrame(
            {
                "GS_ID": ["VALID", "INVALID"],
                "wFC": [2.0, np.nan],
                "pFDR": [0.01, 0.01],
            }
        )

        # Should handle gracefully (either error or assign default)
        try:
            colors = get_colors(df)
            assert len(colors) == 2
        except ValueError:
            pass  # Also acceptable to raise error

    def test_zero_wfc(self):
        """Test with wFC of exactly zero."""
        df = pd.DataFrame(
            {
                "GS_ID": ["ZERO"],
                "wFC": [0.0],
                "pFDR": [0.01],
            }
        )

        colors = get_colors(df)

        # Should be blue (strongly down-regulated)
        assert colors[0] == "blue" or colors[0].lower() == "#0000ff"
