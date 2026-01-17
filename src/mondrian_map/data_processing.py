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
    """Extract coordinate points from dataframe"""
    return [
        (round(df["x"].iloc[i] * scale, 2), round(df["y"].iloc[i] * scale, 2))
        for i in range(len(df))
    ]


def get_areas(df: pd.DataFrame, scale: float = AREA_SCALAR) -> List[float]:
    """Calculate areas based on fold change values"""
    return list(abs(np.log2(df["wFC"])) * scale)


def get_colors(
    df: pd.DataFrame, up_threshold: float = up_th, down_threshold: float = dn_th
) -> List[str]:
    """Determine colors based on fold change and p-value thresholds"""
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
    """Extract pathway IDs (last 4 characters)"""
    return [i[-4:] for i in df["GS_ID"]]


def get_relations(
    mem_df: Optional[pd.DataFrame], threshold: int = 2
) -> List[Tuple[str, str]]:
    """Extract pathway relationships from network data"""
    if mem_df is None or len(mem_df) == 0:
        return []

    relations = []
    rel_count = {}

    # Initialize relationship counts
    for key in set(mem_df["GS_A_ID"]):
        rel_count[key[-4:]] = 0

    # Extract relationships within threshold
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
    """Load pathway annotation information"""
    with open(info_path, "r") as f:
        return json.load(f)


def load_dataset(path: Path, pathway_info: Dict) -> pd.DataFrame:
    """Load and enrich dataset with pathway information"""
    df = pd.read_csv(path)

    # Add pathway information
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
    """Load dataset from uploaded CSV file with validation"""
    try:
        df = pd.read_csv(uploaded_file)

        # Validate required columns
        required_cols = ["GS_ID", "wFC", "pFDR", "x", "y"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Add pathway information
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
    """Load pathway network data for a given dataset"""
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
    """Get human-readable color description for display"""
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
    """Prepare all data needed for Mondrian map creation"""
    # Load network data if directory provided
    mem_df = None
    if network_dir:
        mem_df = load_network_data(dataset_name, network_dir)

        # Filter network data to only include pathways in our dataset
        if mem_df is not None and len(mem_df) > 0:
            available_pathways = set(df["GS_ID"].unique())
            mem_df = mem_df[
                mem_df["GS_A_ID"].isin(available_pathways)
                & mem_df["GS_B_ID"].isin(available_pathways)
            ].reset_index(drop=True)

    # Prepare all required data
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
