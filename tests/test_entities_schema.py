"""
Tests for entities schema validation.

These tests verify that the entities DataFrame has the correct structure
and that all required columns are present with valid data types.
"""

import numpy as np
import pandas as pd
import pytest

from mondrian_map.io import (ENTITIES_SCHEMA, save_entities,
                             validate_entities_schema)


class TestEntitiesSchema:
    """Test suite for entities schema validation."""

    def test_valid_entities(self):
        """Test that valid entities pass validation."""
        df = pd.DataFrame(
            {
                "GS_ID": ["WP001", "WP002", "WP003"],
                "NAME": ["Pathway 1", "Pathway 2", "Pathway 3"],
                "wFC": [1.5, 2.0, 0.5],
                "pFDR": [0.01, 0.001, 0.05],
                "x": [0.1, 0.5, 0.9],
                "y": [0.2, 0.6, 0.8],
            }
        )

        # Should not raise
        result = validate_entities_schema(df)
        assert result is True

    def test_missing_required_column(self):
        """Test that missing required column raises error."""
        df = pd.DataFrame(
            {
                "GS_ID": ["WP001", "WP002"],
                "NAME": ["Pathway 1", "Pathway 2"],
                # Missing wFC, pFDR, x, y
            }
        )

        with pytest.raises(ValueError, match="Missing required columns"):
            validate_entities_schema(df)

    def test_wrong_dtype_wfc(self):
        """Test that wrong dtype for wFC raises error."""
        df = pd.DataFrame(
            {
                "GS_ID": ["WP001", "WP002"],
                "NAME": ["Pathway 1", "Pathway 2"],
                "wFC": ["high", "low"],  # Should be numeric
                "pFDR": [0.01, 0.001],
                "x": [0.1, 0.5],
                "y": [0.2, 0.6],
            }
        )

        with pytest.raises(ValueError, match="Invalid dtype"):
            validate_entities_schema(df)

    def test_nan_in_coordinates(self):
        """Test that NaN in coordinates raises warning."""
        df = pd.DataFrame(
            {
                "GS_ID": ["WP001", "WP002"],
                "NAME": ["Pathway 1", "Pathway 2"],
                "wFC": [1.5, 2.0],
                "pFDR": [0.01, 0.001],
                "x": [0.1, np.nan],  # NaN coordinate
                "y": [0.2, 0.6],
            }
        )

        with pytest.warns(UserWarning, match="NaN values"):
            validate_entities_schema(df, strict=False)

    def test_coordinates_out_of_range(self):
        """Test that coordinates outside [0,1] raise warning."""
        df = pd.DataFrame(
            {
                "GS_ID": ["WP001", "WP002"],
                "NAME": ["Pathway 1", "Pathway 2"],
                "wFC": [1.5, 2.0],
                "pFDR": [0.01, 0.001],
                "x": [0.1, 1.5],  # Out of range
                "y": [0.2, 0.6],
            }
        )

        with pytest.warns(UserWarning, match="outside"):
            validate_entities_schema(df, strict=False)

    def test_negative_pfdr(self):
        """Test that negative pFDR raises error."""
        df = pd.DataFrame(
            {
                "GS_ID": ["WP001", "WP002"],
                "NAME": ["Pathway 1", "Pathway 2"],
                "wFC": [1.5, 2.0],
                "pFDR": [-0.01, 0.001],  # Negative p-value
                "x": [0.1, 0.5],
                "y": [0.2, 0.6],
            }
        )

        with pytest.raises(ValueError, match="pFDR.*negative"):
            validate_entities_schema(df)

    def test_duplicate_gs_id(self):
        """Test that duplicate GS_ID raises warning."""
        df = pd.DataFrame(
            {
                "GS_ID": ["WP001", "WP001"],  # Duplicate
                "NAME": ["Pathway 1", "Pathway 2"],
                "wFC": [1.5, 2.0],
                "pFDR": [0.01, 0.001],
                "x": [0.1, 0.5],
                "y": [0.2, 0.6],
            }
        )

        with pytest.warns(UserWarning, match="Duplicate"):
            validate_entities_schema(df, strict=False)


class TestEntitiesSchemaDefinition:
    """Test the schema definition itself."""

    def test_required_columns(self):
        """Test that schema defines all required columns."""
        required = ["GS_ID", "wFC", "pFDR", "x", "y"]
        for col in required:
            assert col in ENTITIES_SCHEMA["required_columns"], f"Missing {col} in schema"

    def test_column_types(self):
        """Test that column types are valid numpy/pandas dtypes."""
        for col, dtype in ENTITIES_SCHEMA["dtypes"].items():
            assert dtype in ["object", "float64", "int64", "string"]


class TestSaveEntities:
    """Test saving entities to file."""

    def test_save_csv(self, tmp_path):
        """Test saving entities to CSV."""
        df = pd.DataFrame(
            {
                "GS_ID": ["WP001", "WP002"],
                "NAME": ["Pathway 1", "Pathway 2"],
                "wFC": [1.5, 2.0],
                "pFDR": [0.01, 0.001],
                "x": [0.1, 0.5],
                "y": [0.2, 0.6],
            }
        )

        output_path = tmp_path / "entities.csv"
        save_entities(df, output_path)

        # Read back and verify
        loaded = pd.read_csv(output_path)
        assert len(loaded) == 2
        assert list(loaded.columns) == list(df.columns)

    def test_save_validates_schema(self, tmp_path):
        """Test that save validates schema before writing."""
        df = pd.DataFrame(
            {
                "GS_ID": ["WP001"],
                # Missing required columns
            }
        )

        output_path = tmp_path / "entities.csv"
        with pytest.raises(ValueError):
            save_entities(df, output_path, validate=True)


class TestOptionalColumns:
    """Test handling of optional columns."""

    def test_extra_columns_preserved(self):
        """Test that extra columns are preserved."""
        df = pd.DataFrame(
            {
                "GS_ID": ["WP001"],
                "NAME": ["Pathway 1"],
                "wFC": [1.5],
                "pFDR": [0.01],
                "x": [0.1],
                "y": [0.2],
                "custom_col": ["extra"],  # Extra column
            }
        )

        # Should not raise
        result = validate_entities_schema(df)
        assert result is True
        assert "custom_col" in df.columns

    def test_description_column(self):
        """Test handling of optional Description column."""
        df = pd.DataFrame(
            {
                "GS_ID": ["WP001"],
                "NAME": ["Pathway 1"],
                "wFC": [1.5],
                "pFDR": [0.01],
                "x": [0.1],
                "y": [0.2],
                "Description": ["A pathway description"],
            }
        )

        result = validate_entities_schema(df)
        assert result is True
