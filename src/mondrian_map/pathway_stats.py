"""
Pathway Statistics Module

This module implements pathway-level statistics including weighted fold change (wFC)
computation and pathway entity table construction.
"""

import logging
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from .pager_client import PagerClient


def compute_weighted_fold_change(
    fc_by_gene: pd.Series,
    rp_scores_by_gene: Dict[str, float],
    missing: str = "ignore",
    normalize_weights: bool = True,
    weight_norm: str = "sum_abs",
) -> float:
    """
    Compute RP-score-weighted fold change.

    Formula:
        wFC = sum(w_i * FC_i) / norm
    where norm is sum(|w_i|) or sum(w_i) depending on weight_norm.
    """
    rp_scores = pd.Series(rp_scores_by_gene, dtype=float)
    common = fc_by_gene.index.intersection(rp_scores.index)
    if len(common) == 0:
        return 0.0

    fc_vals = fc_by_gene.loc[common].astype(float)
    weights = rp_scores.loc[common].astype(float)

    if missing == "ignore":
        mask = ~(fc_vals.isna() | weights.isna())
        fc_vals = fc_vals[mask]
        weights = weights[mask]
    else:
        raise ValueError(f"Unknown missing policy: {missing}")

    if len(fc_vals) == 0:
        return 0.0

    if normalize_weights:
        if weight_norm == "sum_abs":
            norm = weights.abs().sum()
        elif weight_norm == "sum":
            norm = weights.sum()
        else:
            raise ValueError(f"Unknown weight_norm: {weight_norm}")
    else:
        norm = 1.0

    if norm == 0:
        return 0.0

    return float((weights * fc_vals).sum() / norm)


def compute_pathway_wfc_table(
    pag_df: pd.DataFrame,
    fc_by_gene: pd.Series,
    rp_scores_map: Dict[str, Dict[str, float]],
    pag_id_col: str = "GS_ID",
) -> pd.DataFrame:
    """Compute wFC and coverage metrics for each PAG in pag_df."""
    records = []
    for _, row in pag_df.iterrows():
        pag_id = str(row[pag_id_col])
        rp_scores = rp_scores_map.get(pag_id, {})
        coverage = 0.0
        used = 0
        if rp_scores:
            common = fc_by_gene.index.intersection(rp_scores.keys())
            used = len(common)
            coverage = used / max(len(rp_scores), 1)
        wfc = compute_weighted_fold_change(fc_by_gene, rp_scores)
        records.append(
            {
                pag_id_col: pag_id,
                "wFC": wfc,
                "wFC_gene_coverage": coverage,
                "wFC_genes_used": used,
            }
        )
    wfc_df = pd.DataFrame(records)
    return pd.merge(pag_df, wfc_df, on=pag_id_col, how="left")


def compute_wfc(
    gene_fc: pd.Series,
    rp_scores: pd.Series,
    preserve_sign: bool = True,
    use_abs_fc: bool = False,
) -> float:
    """
    Compute weighted fold change for a pathway.

    The weighted fold change is computed as:
        wFC = sum(W_i * FC_i) / sum(W_i)

    where W_i is the RP score and FC_i is the fold change for gene i.

    Args:
        gene_fc: Series of fold change values indexed by gene symbol
        rp_scores: Series of RP scores indexed by gene symbol
        preserve_sign: If True, preserve the sign of fold changes (default True)
        use_abs_fc: If True, use absolute fold change values (overrides preserve_sign)

    Returns:
        Weighted fold change value

    Example:
        >>> gene_fc = pd.Series({"BRCA1": 1.5, "TP53": 0.7}, name="fold_change")
        >>> rp_scores = pd.Series({"BRCA1": 0.8, "TP53": 0.6}, name="rp_score")
        >>> wfc = compute_wfc(gene_fc, rp_scores)
    """
    common_genes = gene_fc.index.intersection(rp_scores.index)

    if len(common_genes) == 0:
        logger.warning("No common genes found between FC and RP scores")
        return 0.0

    fc_values = gene_fc.loc[common_genes].astype(float)
    weights = rp_scores.loc[common_genes].astype(float)

    valid_mask = ~(fc_values.isna() | weights.isna())
    fc_values = fc_values[valid_mask]
    weights = weights[valid_mask]

    if len(fc_values) == 0 or weights.sum() == 0:
        return 0.0

    if use_abs_fc:
        fc_values = fc_values.abs()

    weighted_fc = (weights * fc_values).sum() / weights.sum()

    return round(weighted_fc, 4)


def compute_pathway_wfc(
    pag_id: str,
    ranked_genes_df: pd.DataFrame,
    fold_change_df: pd.DataFrame,
    fc_column: str,
    preserve_sign: bool = True,
) -> float:
    """
    Compute weighted fold change for a specific pathway.

    Args:
        pag_id: Pathway identifier
        ranked_genes_df: DataFrame with RP-ranked genes (from PAGER)
                        Must have columns: GENE_SYM, RP_SCORE
        fold_change_df: DataFrame with fold changes (genes as index)
        fc_column: Column name containing fold change values
        preserve_sign: Whether to preserve sign of fold changes

    Returns:
        Weighted fold change value for the pathway
    """
    if "GENE_SYM" not in ranked_genes_df.columns:
        raise ValueError("ranked_genes_df must have 'GENE_SYM' column")
    if "RP_SCORE" not in ranked_genes_df.columns:
        raise ValueError("ranked_genes_df must have 'RP_SCORE' column")

    rp_scores = ranked_genes_df.set_index("GENE_SYM")["RP_SCORE"].astype(float)

    gene_fc = fold_change_df[fc_column]

    return compute_wfc(gene_fc, rp_scores, preserve_sign=preserve_sign)


def compute_all_pathway_wfc(
    pag_df: pd.DataFrame,
    fold_change_df: pd.DataFrame,
    fc_column: str,
    pager_client: "PagerClient",
    preserve_sign: bool = True,
    progress_bar: bool = True,
) -> pd.DataFrame:
    """
    Compute weighted fold change for all pathways in a GNPA result.

    Args:
        pag_df: DataFrame from PAGER GNPA (must have GS_ID column)
        fold_change_df: DataFrame with fold changes (genes as index)
        fc_column: Column name containing fold change values
        pager_client: PagerClient instance for API calls
        preserve_sign: Whether to preserve sign of fold changes
        progress_bar: Whether to show progress bar

    Returns:
        DataFrame with columns: GS_ID, wFC, pFDR
    """
    try:
        from tqdm import tqdm

        iterator = tqdm(pag_df["GS_ID"].unique(), disable=not progress_bar)
    except ImportError:
        iterator = pag_df["GS_ID"].unique()
        if progress_bar:
            logger.warning("tqdm not available, progress bar disabled")

    pag_ids = []
    wfc_values = []
    pfdr_values = []

    for pag_id in iterator:
        ranked_genes = pager_client.get_pag_ranked_genes(pag_id)

        wfc = compute_pathway_wfc(
            pag_id,
            ranked_genes,
            fold_change_df,
            fc_column,
            preserve_sign=preserve_sign,
        )

        pfdr = pag_df[pag_df["GS_ID"] == pag_id]["pFDR"].values[0]

        pag_ids.append(pag_id)
        wfc_values.append(wfc)
        pfdr_values.append(pfdr)

    result = pd.DataFrame(
        {
            "GS_ID": pag_ids,
            "wFC": wfc_values,
            "pFDR": pfdr_values,
        }
    )

    logger.info(f"Computed wFC for {len(result)} pathways")
    return result


def compute_pathway_wfc_batch(
    pag_ids: List[str],
    pag_member_df: pd.DataFrame,
    fold_change_df: pd.DataFrame,
    fc_column: str,
    pager_client: "PagerClient",
    pag_df: pd.DataFrame,
    preserve_sign: bool = True,
    progress_bar: bool = True,
) -> pd.DataFrame:
    """
    Compute weighted fold change for multiple pathways using batch member data.

    This is more efficient when you already have PAG membership data.

    Args:
        pag_ids: List of pathway IDs
        pag_member_df: DataFrame with PAG membership (from get_pag_members)
        fold_change_df: DataFrame with fold changes (genes as index)
        fc_column: Column name containing fold change values
        pager_client: PagerClient instance
        pag_df: Original PAGER results with pFDR values
        preserve_sign: Whether to preserve sign of fold changes
        progress_bar: Whether to show progress bar

    Returns:
        DataFrame with columns: GS_ID, wFC, pFDR
    """
    try:
        from tqdm import tqdm

        iterator = tqdm(pag_ids, disable=not progress_bar)
    except ImportError:
        iterator = pag_ids

    results = []

    for pag_id in iterator:
        ranked_genes = pager_client.get_pag_ranked_genes(pag_id)

        if len(ranked_genes) == 0:
            logger.warning(f"No ranked genes found for {pag_id}")
            continue

        ranked_genes["RP_SCORE"] = pd.to_numeric(
            ranked_genes["RP_SCORE"], errors="coerce"
        )

        weights = []
        wfc_vals = []

        for _, gene_row in ranked_genes.iterrows():
            gene = gene_row["GENE_SYM"]
            if gene in fold_change_df.index:
                w = gene_row["RP_SCORE"]
                fc = fold_change_df.loc[gene, fc_column]
                if not np.isnan(w):
                    weights.append(w)
                    wfc_vals.append(w * fc)

        if weights and sum(weights) > 0:
            wfc = round(sum(wfc_vals) / sum(weights), 4)
        else:
            wfc = 0.0

        pfdr_rows = pag_df[pag_df["GS_ID"] == pag_id]["pFDR"]
        pfdr = pfdr_rows.values[0] if len(pfdr_rows) > 0 else 1.0

        results.append(
            {
                "GS_ID": pag_id,
                "wFC": wfc,
                "pFDR": pfdr,
            }
        )

    return pd.DataFrame(results)


def build_entities_table(
    wfc_df: pd.DataFrame,
    coordinates_df: pd.DataFrame,
    pathway_info: Optional[Dict] = None,
    include_metadata: bool = True,
) -> pd.DataFrame:
    """
    Build the entities table for Mondrian Map visualization.

    Args:
        wfc_df: DataFrame with columns: GS_ID, wFC, pFDR
        coordinates_df: DataFrame with columns: GS_ID, x, y
        pathway_info: Optional dictionary mapping GS_ID to pathway metadata
        include_metadata: Whether to include pathway name/description

    Returns:
        DataFrame with columns: GS_ID, wFC, pFDR, x, y, [NAME, Description, ...]
    """
    # Merge wFC with coordinates
    if "GS_ID" not in wfc_df.columns or "GS_ID" not in coordinates_df.columns:
        raise ValueError("Both DataFrames must have 'GS_ID' column")

    entities = pd.merge(wfc_df, coordinates_df, on="GS_ID", how="inner")

    if len(entities) == 0:
        logger.warning("No matching pathways between wFC and coordinates")
        return entities

    if include_metadata and pathway_info:
        entities["NAME"] = entities["GS_ID"].map(
            lambda x: pathway_info.get(x, {}).get("NAME", x)
        )
        entities["Description"] = entities["GS_ID"].map(
            lambda x: pathway_info.get(x, {}).get("Description", "")
        )
        entities["Ontology"] = entities["GS_ID"].map(
            lambda x: pathway_info.get(x, {}).get("Pathway Ontology", "")
        )
        entities["Disease"] = entities["GS_ID"].map(
            lambda x: pathway_info.get(x, {}).get("Disease", "")
        )

    required = ["GS_ID", "wFC", "pFDR", "x", "y"]
    missing = [col for col in required if col not in entities.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    logger.info(f"Built entities table with {len(entities)} pathways")
    return entities


def determine_regulation_direction(
    wfc: float,
    pfdr: float,
    up_threshold: float = 1.5,
    down_threshold: float = 0.5,
    significance_threshold: float = 0.05,
) -> str:
    """
    Determine regulation direction based on wFC and pFDR.

    Args:
        wfc: Weighted fold change value
        pfdr: FDR-adjusted p-value
        up_threshold: Threshold for up-regulation (default: 1.5)
        down_threshold: Threshold for down-regulation (default: 0.5)
        significance_threshold: p-value threshold for significance

    Returns:
        Direction string: "up", "down", "not_significant", or "mixed"
    """
    if pfdr >= significance_threshold:
        return "not_significant"
    elif wfc >= up_threshold:
        return "up"
    elif wfc <= down_threshold:
        return "down"
    else:
        return "mixed"


def classify_pathways(
    entities_df: pd.DataFrame,
    up_threshold: float = 1.5,
    down_threshold: float = 0.5,
    significance_threshold: float = 0.05,
) -> pd.DataFrame:
    """
    Classify pathways by regulation direction.

    Classification rules (following paper specification):
    - "up": wFC >= up_threshold (default 1.5) and pFDR < significance_threshold
    - "down": wFC <= down_threshold (default 0.5) and pFDR < significance_threshold
    - "not_significant": pFDR >= significance_threshold
    - "mixed": pFDR < significance_threshold but wFC between thresholds

    Args:
        entities_df: DataFrame with wFC and pFDR columns
        up_threshold: Threshold for up-regulation (default: 1.5)
        down_threshold: Threshold for down-regulation (default: 0.5)
        significance_threshold: p-value threshold (default: 0.05)

    Returns:
        DataFrame with added 'classification' column
    """
    df = entities_df.copy()

    df["classification"] = df.apply(
        lambda row: determine_regulation_direction(
            row["wFC"],
            row["pFDR"],
            up_threshold,
            down_threshold,
            significance_threshold,
        ),
        axis=1,
    )

    class_counts = df["classification"].value_counts().to_dict()
    logger.info(f"Pathway classification: {class_counts}")

    return df


def filter_top_pathways(
    entities_df: pd.DataFrame,
    n: int = 10,
    sort_by: str = "pFDR",
    ascending: bool = True,
) -> pd.DataFrame:
    """
    Filter to top N pathways by significance or fold change.

    Args:
        entities_df: Entities DataFrame
        n: Number of top pathways to keep
        sort_by: Column to sort by ("pFDR" or "wFC")
        ascending: Sort order

    Returns:
        Filtered DataFrame with top N pathways
    """
    return (
        entities_df.sort_values(by=sort_by, ascending=ascending)
        .head(n)
        .reset_index(drop=True)
    )


def compute_pathway_statistics(
    entities_df: pd.DataFrame,
) -> Dict:
    """
    Compute summary statistics for pathway analysis.

    Args:
        entities_df: Entities DataFrame with wFC, pFDR columns

    Returns:
        Dictionary with summary statistics
    """
    stats = {
        "total_pathways": len(entities_df),
        "significant_count": (entities_df["pFDR"] < 0.05).sum(),
        "up_regulated_count": (entities_df["wFC"] >= 1.25).sum(),
        "down_regulated_count": (entities_df["wFC"] <= 0.75).sum(),
        "wfc_mean": entities_df["wFC"].mean(),
        "wfc_std": entities_df["wFC"].std(),
        "wfc_min": entities_df["wFC"].min(),
        "wfc_max": entities_df["wFC"].max(),
        "pfdr_median": entities_df["pFDR"].median(),
    }

    return stats
