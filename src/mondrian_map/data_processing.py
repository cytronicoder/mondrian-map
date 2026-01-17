"""
Data Processing Module for Mondrian Maps

This module handles pathway data loading, processing, and preparation
for visualization in Mondrian maps.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .core import AREA_SCALAR, dn_th, up_th


def get_points(df: pd.DataFrame, scale: float = 1) -> List[Tuple[float, float]]:
    """Extract scaled coordinate points from pathway embeddings.

    Args:
        df: DataFrame containing 'x' and 'y' columns with embedding coordinates
        scale: Scaling factor applied to coordinates (default: 1.0)

    Returns:
        List of (x, y) coordinate tuples rounded to 2 decimal places
    """
    return [
        (round(df["x"].iloc[i] * scale, 2), round(df["y"].iloc[i] * scale, 2))
        for i in range(len(df))
    ]


def get_areas(df: pd.DataFrame, scale: float = AREA_SCALAR) -> List[float]:
    """Calculate block areas proportional to pathway fold change magnitude.

    Block area is derived from log2-transformed absolute weighted fold change,
    representing the magnitude of gene expression perturbation within each pathway.

    Args:
        df: DataFrame with 'wFC' (weighted fold change) column
        scale: Scaling factor for area normalization (default: AREA_SCALAR)

    Returns:
        List of scaled area values for pathway visualization blocks
    """
    return list(abs(np.log2(df["wFC"])) * scale)


def get_colors(
    df: pd.DataFrame, up_threshold: float = up_th, down_threshold: float = dn_th
) -> List[str]:
    """Assign pathway colors based on statistical significance and direction of regulation.

    Color assignment:
    - Red: Up-regulated (pFDR < 0.05 and wFC >= up_threshold)
    - Blue: Down-regulated (pFDR < 0.05 and wFC <= down_threshold)
    - Yellow: Neutral change (pFDR < 0.05 but between thresholds)
    - Black: Non-significant (pFDR >= 0.05)

    Args:
        df: DataFrame with 'pFDR' and 'wFC' columns
        up_threshold: Weighted fold change threshold for up-regulation
        down_threshold: Weighted fold change threshold for down-regulation

    Returns:
        List of color assignments matching pathway rows
    """
    colors = []
    for i, row in df.iterrows():
        if row["pFDR"] < 0.05:
            if row["wFC"] >= up_threshold:
                colors.append("red")
            elif row["wFC"] <= down_threshold:
                colors.append("blue")
            else:
                colors.append("yellow")
        else:
            colors.append("black")
    return colors


def get_IDs(df: pd.DataFrame) -> List[str]:
    """Extract abbreviated pathway identifiers from full gene set IDs.

    Uses the final 4 characters of each GS_ID as a compact identifier
    for visualization and display purposes.

    Args:
        df: DataFrame containing 'GS_ID' column with full pathway identifiers

    Returns:
        List of 4-character abbreviated pathway IDs
    """
    return [i[-4:] for i in df["GS_ID"]]


def get_relations(
    mem_df: Optional[pd.DataFrame], threshold: int = 2
) -> List[Tuple[str, str]]:
    """Extract pathway pairwise relationships with degree constraints.

    Identifies meaningful pathway-to-pathway connections from network data,
    enforcing a maximum degree constraint to prevent over-connection while
    avoiding bidirectional duplicate edges.

    Args:
        mem_df: DataFrame with 'GS_A_ID' and 'GS_B_ID' columns representing
                pathway relationships, or None if no network data available
        threshold: Maximum number of connections per pathway node (default: 2)

    Returns:
        List of (pathway_id_a, pathway_id_b) tuples representing filtered edges
    """
    if mem_df is None or len(mem_df) == 0:
        return []

    relations = []
    rel_count = {}

    for key in set(mem_df["GS_A_ID"]):
        rel_count[key[-4:]] = 0

    for index, row in mem_df.iterrows():
        gs_a_id = row["GS_A_ID"][-4:]
        gs_b_id = row["GS_B_ID"][-4:]

        if (
            (gs_b_id, gs_a_id) not in relations
            and rel_count.get(gs_a_id, 0) < threshold
            and rel_count.get(gs_b_id, 0) < threshold
        ):
            relations.append((gs_a_id, gs_b_id))
            rel_count[gs_a_id] = rel_count.get(gs_a_id, 0) + 1
            rel_count[gs_b_id] = rel_count.get(gs_b_id, 0) + 1

    return relations


def load_pathway_info(info_path: Path) -> Dict:
    """Load pathway metadata and annotation information from JSON file.

    Args:
        info_path: Path to JSON file containing pathway annotations

    Returns:
        Dictionary mapping pathway IDs to metadata (Description, Ontology, Disease, NAME)
    """
    with open(info_path, "r") as f:
        return json.load(f)


def load_dataset(path: Path, pathway_info: Dict) -> pd.DataFrame:
    """Load pathway analysis results and enrich with annotation metadata.

    Merges statistical results (wFC, pFDR, coordinates) with pathway descriptions,
    ontology classifications, and disease associations.

    Args:
        path: CSV file containing pathway analysis results (wFC, pFDR, x, y columns)
        pathway_info: Dictionary mapping pathway IDs to annotation metadata

    Returns:
        DataFrame with statistical results and enriched pathway metadata
    """
    df = pd.read_csv(path)

    df["Description"] = df["GS_ID"].map(
        lambda x: pathway_info.get(x, {}).get("Description", "")
    )
    df["Ontology"] = df["GS_ID"].map(
        lambda x: pathway_info.get(x, {}).get("Pathway Ontology", "")
    )
    df["Disease"] = df["GS_ID"].map(
        lambda x: pathway_info.get(x, {}).get("Disease", "")
    )
    df["NAME"] = df["GS_ID"].map(lambda x: pathway_info.get(x, {}).get("NAME", x))

    return df


def load_uploaded_dataset(uploaded_file, pathway_info: Dict) -> Optional[pd.DataFrame]:
    """Load and validate pathway analysis results from user-uploaded CSV.

    Validates presence of required columns (GS_ID, wFC, pFDR, x, y) and enriches
    data with pathway annotations.

    Args:
        uploaded_file: File-like object containing CSV data
        pathway_info: Dictionary of pathway metadata for enrichment

    Returns:
        Validated and enriched DataFrame, or None if validation fails
    """
    try:
        df = pd.read_csv(uploaded_file)

        required_cols = ["GS_ID", "wFC", "pFDR", "x", "y"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        df["Description"] = df["GS_ID"].map(
            lambda x: pathway_info.get(x, {}).get("Description", "")
        )
        df["Ontology"] = df["GS_ID"].map(
            lambda x: pathway_info.get(x, {}).get("Pathway Ontology", "")
        )
        df["Disease"] = df["GS_ID"].map(
            lambda x: pathway_info.get(x, {}).get("Disease", "")
        )
        df["NAME"] = df["GS_ID"].map(lambda x: pathway_info.get(x, {}).get("NAME", x))

        return df

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


def load_network_data(dataset_name: str, network_dir: Path) -> Optional[pd.DataFrame]:
    """Load pathway relationship network from dataset-specific network file.

    Args:
        dataset_name: Name of the dataset/case study (e.g., 'Aggressive R1')
        network_dir: Directory containing pathway relationship network CSVs

    Returns:
        DataFrame with pathway relationship edges (GS_A_ID, GS_B_ID columns), or None
    """
    network_files = {
        "Aggressive R1": network_dir / "wikipathway_aggressive_R1_TP.csv",
        "Aggressive R2": network_dir / "wikipathway_aggressive_R2_TP.csv",
        "Baseline R1": network_dir / "wikipathway_baseline_R1_TP.csv",
        "Baseline R2": network_dir / "wikipathway_baseline_R2_TP.csv",
        "Nonaggressive R1": network_dir / "wikipathway_nonaggressive_R1_TP.csv",
        "Nonaggressive R2": network_dir / "wikipathway_nonaggressive_R2_TP.csv",
    }

    if dataset_name in network_files and network_files[dataset_name].exists():
        return pd.read_csv(network_files[dataset_name])
    else:
        return None


def get_mondrian_color_description(wfc: float, p_value: float) -> str:
    """Generate human-readable description of pathway regulation status.

    Args:
        wfc: Weighted fold change value for the pathway
        p_value: False discovery rate adjusted p-value

    Returns:
        Categorical description of pathway regulation state
    """
    if p_value > 0.05:
        return "Non-significant"

    if abs(wfc) < 0.5:
        return "Neutral"
    elif abs(wfc) < 1.0:
        return "Moderate change"
    elif wfc > 0:
        return "Up-regulated"
    else:
        return "Down-regulated"


def prepare_pathway_data(
    df: pd.DataFrame, dataset_name: str, network_dir: Optional[Path] = None
) -> Dict:
    """Aggregate all pathway data needed for Mondrian map visualization.

    Compiles pathway embeddings, statistical information, network relationships,
    and visual attributes into a unified data structure.

    Args:
        df: DataFrame with pathway analysis results (wFC, pFDR, x, y, GS_ID columns)
        dataset_name: Case study name for loading correct network file
        network_dir: Optional directory containing pathway relationship networks

    Returns:
        Dictionary with keys: center_points, areas, colors, pathway_ids, relations,
        network_data. Each contains visualization-ready pathway information.
    """
    mem_df = None
    if network_dir:
        mem_df = load_network_data(dataset_name, network_dir)

        if mem_df is not None and len(mem_df) > 0:
            available_pathways = set(df["GS_ID"].unique())
            mem_df = mem_df[
                mem_df["GS_A_ID"].isin(available_pathways)
                & mem_df["GS_B_ID"].isin(available_pathways)
            ].reset_index(drop=True)

    center_points = get_points(df, 1)
    areas = get_areas(df, AREA_SCALAR)
    colors = get_colors(df, up_th, dn_th)
    pathway_ids = get_IDs(df)
    relations = get_relations(mem_df) if mem_df is not None else []

    return {
        "center_points": center_points,
        "areas": areas,
        "colors": colors,
        "pathway_ids": pathway_ids,
        "relations": relations,
        "network_data": mem_df,
    }
