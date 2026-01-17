"""
I/O Module for Mondrian Map

This module handles file loading, saving, and validation for all data types
used in the Mondrian Map pipeline.
"""

import hashlib
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Schema definitions for validation
ENTITIES_SCHEMA = {
    "required_columns": ["GS_ID", "wFC", "pFDR", "x", "y"],
    "optional_columns": ["NAME", "Description", "Ontology", "Disease"],
    "dtypes": {
        "GS_ID": "string",
        "wFC": "float64",
        "pFDR": "float64",
        "x": "float64",
        "y": "float64",
    },
}

RELATIONS_SCHEMA = {
    "required_columns": ["GS_A_ID", "GS_B_ID"],
    "optional_columns": ["JACCARD_SIMILARITY", "OVERLAP"],
    "dtypes": {
        "GS_A_ID": "string",
        "GS_B_ID": "string",
    },
}

DEG_SCHEMA = {
    "required_columns": ["gene_symbol", "fold_change", "direction"],
    "optional_columns": ["p_value", "adjusted_p_value"],
    "dtypes": {
        "gene_symbol": "string",
        "fold_change": "float64",
        "direction": "string",
    },
}

EXPRESSION_SCHEMA = {
    "required_index": "Gene_symbol",
    "min_columns": 1,
}


class SchemaValidationError(Exception):
    """Raised when a dataframe doesn't match the expected schema."""

    pass


import warnings


def validate_entities_schema(
    df: pd.DataFrame,
    strict: bool = True,
) -> bool:
    """
    Validate entities DataFrame against the expected schema.

    Args:
        df: DataFrame to validate
        strict: If True, raise errors; if False, emit warnings

    Returns:
        True if valid

    Raises:
        ValueError: If validation fails and strict=True
    """
    required_cols = ["GS_ID", "wFC", "pFDR", "x", "y"]

    # Check required columns
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Check dtypes for numeric columns
    for col in ["wFC", "pFDR", "x", "y"]:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"Invalid dtype for column '{col}': expected numeric")

    # Check for NaN values in coordinates
    if df["x"].isna().any() or df["y"].isna().any():
        if strict:
            raise ValueError("NaN values found in coordinate columns")
        else:
            warnings.warn("NaN values found in coordinate columns", UserWarning)

    # Check coordinate ranges
    if (
        (df["x"] < 0).any()
        or (df["x"] > 1).any()
        or (df["y"] < 0).any()
        or (df["y"] > 1).any()
    ):
        if strict:
            raise ValueError("Coordinates outside [0, 1] range")
        else:
            warnings.warn("Coordinates outside [0, 1] range", UserWarning)

    # Check for negative pFDR values
    if (df["pFDR"] < 0).any():
        raise ValueError("pFDR contains negative values")

    # Check for duplicate GS_ID
    if df["GS_ID"].duplicated().any():
        if strict:
            raise ValueError("Duplicate GS_ID values found")
        else:
            warnings.warn("Duplicate GS_ID values found", UserWarning)

    return True


def validate_dataframe(
    df: pd.DataFrame,
    schema: Dict[str, Any],
    name: str = "DataFrame",
) -> pd.DataFrame:
    """
    Validate a DataFrame against a schema specification.

    Args:
        df: DataFrame to validate
        schema: Schema dictionary with required_columns, optional_columns, dtypes
        name: Name for error messages

    Returns:
        Validated DataFrame with proper dtypes

    Raises:
        SchemaValidationError: If validation fails
    """
    # Check required columns
    required = schema.get("required_columns", [])
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise SchemaValidationError(
            f"{name} missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    # Apply dtype conversions
    dtypes = schema.get("dtypes", {})
    for col, dtype in dtypes.items():
        if col in df.columns:
            try:
                if dtype == "string":
                    df[col] = df[col].astype(str)
                else:
                    df[col] = df[col].astype(dtype)
            except (ValueError, TypeError) as e:
                raise SchemaValidationError(
                    f"{name} column '{col}' cannot be converted to {dtype}: {e}"
                )

    logger.debug(f"Validated {name} with {len(df)} rows")
    return df


def load_expression_matrix(
    path: Union[str, Path],
    index_col: str = "Gene_symbol",
    sep: str = "\t",
    fillna: float = 0.0,
    min_value: Optional[float] = None,
) -> pd.DataFrame:
    """
    Load a gene expression matrix from file.

    Args:
        path: Path to the expression matrix file (TSV or CSV)
        index_col: Column to use as index (gene identifiers)
        sep: Delimiter (default: tab for TSV)
        fillna: Value to fill NaN with
        min_value: If provided, filter rows where all values are >= min_value

    Returns:
        DataFrame with genes as index and samples as columns

    Raises:
        FileNotFoundError: If file doesn't exist
        SchemaValidationError: If format is invalid
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Expression matrix not found: {path}")

    logger.info(f"Loading expression matrix from {path}")

    # Auto-detect separator
    if path.suffix == ".csv":
        sep = ","

    df = pd.read_csv(path, sep=sep)

    if index_col not in df.columns:
        # Try first column as index
        df = pd.read_csv(path, sep=sep, index_col=0)
        logger.warning(f"Column '{index_col}' not found, using first column as index")
    else:
        df = df.set_index(index_col)

    # Fill NaN values
    df = df.fillna(fillna)

    # Filter by minimum value if specified
    if min_value is not None:
        original_count = len(df)
        df = df.loc[(df >= min_value).all(axis=1)]
        logger.info(
            f"Filtered genes: {original_count} -> {len(df)} "
            f"(min_value >= {min_value})"
        )

    logger.info(
        f"Loaded expression matrix: {df.shape[0]} genes × {df.shape[1]} samples"
    )
    return df


def load_deg_table(
    path: Union[str, Path],
    validate: bool = True,
) -> pd.DataFrame:
    """
    Load a differentially expressed genes table.

    Args:
        path: Path to the DEG table (CSV)
        validate: Whether to validate against DEG schema

    Returns:
        DataFrame with DEG information
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"DEG table not found: {path}")

    logger.info(f"Loading DEG table from {path}")
    df = pd.read_csv(path)

    if validate:
        df = validate_dataframe(df, DEG_SCHEMA, "DEG table")

    logger.info(f"Loaded {len(df)} differentially expressed genes")
    return df


def load_entities(
    path: Union[str, Path],
    validate: bool = True,
) -> pd.DataFrame:
    """
    Load an entities table for Mondrian Map visualization.

    Args:
        path: Path to the entities CSV file
        validate: Whether to validate against schema

    Returns:
        DataFrame with columns: GS_ID, wFC, pFDR, x, y, [optional cols]
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Entities file not found: {path}")

    logger.info(f"Loading entities from {path}")
    df = pd.read_csv(path)

    if validate:
        df = validate_dataframe(df, ENTITIES_SCHEMA, "Entities")

    logger.info(f"Loaded {len(df)} pathway entities")
    return df


def save_entities(
    df: pd.DataFrame,
    path: Union[str, Path],
    validate: bool = True,
) -> Path:
    """
    Save an entities table to CSV.

    Args:
        df: DataFrame with pathway entities
        path: Output path
        validate: Whether to validate before saving

    Returns:
        Path to saved file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if validate:
        df = validate_dataframe(df, ENTITIES_SCHEMA, "Entities")

    df.to_csv(path, index=False)
    logger.info(f"Saved {len(df)} entities to {path}")
    return path


def load_relations(
    path: Union[str, Path],
    validate: bool = True,
) -> pd.DataFrame:
    """
    Load a relations table (PAG-PAG network).

    Args:
        path: Path to the relations CSV file
        validate: Whether to validate against schema

    Returns:
        DataFrame with columns: GS_A_ID, GS_B_ID, [optional cols]
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Relations file not found: {path}")

    logger.info(f"Loading relations from {path}")
    df = pd.read_csv(path)

    if validate:
        df = validate_dataframe(df, RELATIONS_SCHEMA, "Relations")

    logger.info(f"Loaded {len(df)} pathway relations")
    return df


def save_relations(
    df: pd.DataFrame,
    path: Union[str, Path],
    validate: bool = True,
) -> Path:
    """
    Save a relations table to CSV.

    Args:
        df: DataFrame with pathway relations
        path: Output path
        validate: Whether to validate before saving

    Returns:
        Path to saved file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if validate:
        df = validate_dataframe(df, RELATIONS_SCHEMA, "Relations")

    df.to_csv(path, index=False)
    logger.info(f"Saved {len(df)} relations to {path}")
    return path


def load_embeddings(
    path: Union[str, Path],
) -> np.ndarray:
    """
    Load embeddings from a NumPy file.

    Args:
        path: Path to the .npy file

    Returns:
        NumPy array of embeddings
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {path}")

    logger.info(f"Loading embeddings from {path}")
    embeddings = np.load(path)
    logger.info(f"Loaded embeddings with shape: {embeddings.shape}")
    return embeddings


def save_embeddings(
    embeddings: np.ndarray,
    path: Union[str, Path],
) -> Path:
    """
    Save embeddings to a NumPy file.

    Args:
        embeddings: NumPy array to save
        path: Output path (.npy)

    Returns:
        Path to saved file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    np.save(path, embeddings)
    logger.info(f"Saved embeddings with shape {embeddings.shape} to {path}")
    return path


def load_pathway_info(
    path: Union[str, Path],
) -> Dict[str, Dict[str, Any]]:
    """
    Load pathway annotation information from JSON.

    Args:
        path: Path to JSON file with pathway annotations

    Returns:
        Dictionary mapping GS_ID to pathway info
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pathway info file not found: {path}")

    logger.info(f"Loading pathway info from {path}")
    with open(path, "r") as f:
        info = json.load(f)

    logger.info(f"Loaded info for {len(info)} pathways")
    return info


def save_pathway_info(
    info: Dict[str, Dict[str, Any]],
    path: Union[str, Path],
) -> Path:
    """
    Save pathway annotation information to JSON.

    Args:
        info: Dictionary mapping GS_ID to pathway info
        path: Output path

    Returns:
        Path to saved file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(info, f, indent=2)

    logger.info(f"Saved info for {len(info)} pathways to {path}")
    return path


def load_prompts(
    path: Union[str, Path],
) -> Dict[str, Any]:
    """
    Load prompt data from JSON.

    Args:
        path: Path to prompts JSON file

    Returns:
        Dictionary of prompts keyed by GS_ID
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Prompts file not found: {path}")

    logger.info(f"Loading prompts from {path}")
    with open(path, "r") as f:
        prompts = json.load(f)

    logger.info(f"Loaded {len(prompts)} prompts")
    return prompts


def save_prompts(
    prompts: Dict[str, Any],
    path: Union[str, Path],
) -> Path:
    """
    Save prompts to JSON.

    Args:
        prompts: Dictionary of prompts
        path: Output path

    Returns:
        Path to saved file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(prompts, f, indent=2)

    logger.info(f"Saved {len(prompts)} prompts to {path}")
    return path


def load_pickle(path: Union[str, Path]) -> Any:
    """
    Load data from a pickle file.

    Args:
        path: Path to pickle file

    Returns:
        Unpickled object
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pickle file not found: {path}")

    logger.info(f"Loading pickle from {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def save_pickle(data: Any, path: Union[str, Path]) -> Path:
    """
    Save data to a pickle file.

    Args:
        data: Object to pickle
        path: Output path

    Returns:
        Path to saved file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(data, f)

    logger.info(f"Saved pickle to {path}")
    return path


def compute_file_hash(path: Union[str, Path], algorithm: str = "md5") -> str:
    """
    Compute hash of a file for caching and verification.

    Args:
        path: Path to file
        algorithm: Hash algorithm (md5, sha256)

    Returns:
        Hex digest of file hash
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    hasher = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def load_manifest(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a manifest JSON file.

    Args:
        path: Path to manifest.json

    Returns:
        Manifest dictionary
    """
    path = Path(path)
    if not path.exists():
        return {}

    with open(path, "r") as f:
        return json.load(f)


def save_manifest(
    manifest: Dict[str, Any],
    path: Union[str, Path],
) -> Path:
    """
    Save a manifest JSON file.

    Args:
        manifest: Manifest dictionary
        path: Output path

    Returns:
        Path to saved file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Saved manifest to {path}")
    return path


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
