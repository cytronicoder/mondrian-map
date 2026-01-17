"""
Differentially Expressed Genes (DEGs) Module

This module handles DEG selection and fold change computation
for the Mondrian Map pipeline.
"""

import logging
from typing import List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default thresholds
DEFAULT_UP_THRESHOLD = 1.5
DEFAULT_DOWN_THRESHOLD = 0.5  # abs(1 - (1.5 - 1))
DEFAULT_PSEUDOCOUNT = 1e-6


def compute_fold_change(
    expression_df: pd.DataFrame,
    numerator_col: str,
    denominator_col: str,
    method: str = "ratio",
    pseudocount: float = DEFAULT_PSEUDOCOUNT,
    log_transform: bool = False,
) -> pd.Series:
    """
    Compute fold change between two conditions.

    Args:
        expression_df: DataFrame with gene expression values (genes as index)
        numerator_col: Column name for the numerator (e.g., "R1" timepoint)
        denominator_col: Column name for the denominator (e.g., "TP" baseline)
        method: Computation method - "ratio" for simple ratio, "log2" for log2 ratio
        pseudocount: Small value added to avoid division by zero
        log_transform: Whether to return log2 fold change

    Returns:
        Series of fold change values indexed by gene

    Example:
        >>> fc = compute_fold_change(expr_df, "sample_R1", "sample_TP")
        >>> # Returns Series like: {"BRCA1": 1.5, "TP53": 0.7, ...}
    """
    if numerator_col not in expression_df.columns:
        raise ValueError(f"Numerator column '{numerator_col}' not found")
    if denominator_col not in expression_df.columns:
        raise ValueError(f"Denominator column '{denominator_col}' not found")

    numerator = expression_df[numerator_col].astype(float)
    denominator = expression_df[denominator_col].astype(float) + pseudocount

    if method == "ratio":
        fc = numerator / denominator
    elif method == "log2":
        fc = np.log2(numerator + pseudocount) - np.log2(denominator)
        log_transform = False  # Already log transformed
    else:
        raise ValueError(f"Unknown method: {method}. Use 'ratio' or 'log2'")

    if log_transform:
        fc = np.log2(fc + pseudocount)

    logger.debug(
        f"Computed fold change for {len(fc)} genes "
        f"(median={fc.median():.3f}, range=[{fc.min():.3f}, {fc.max():.3f}])"
    )

    return fc


def compute_temporal_fold_change(
    expression_df: pd.DataFrame,
    case_name: str,
    timepoint_pairs: Optional[List[Tuple[str, str]]] = None,
    pseudocount: float = DEFAULT_PSEUDOCOUNT,
) -> pd.DataFrame:
    """
    Compute fold changes for temporal data (R1/TP and R2/TP ratios).

    This matches the notebook logic for GBM case study data.

    Args:
        expression_df: DataFrame with columns like "{case}_R1", "{case}_R2", "{case}_TP"
        case_name: Case identifier (e.g., "baseline", "aggressive", "nonaggressive")
        timepoint_pairs: List of (numerator, denominator) tuples.
                        Defaults to [("R1", "TP"), ("R2", "TP")]
        pseudocount: Small value to avoid division by zero

    Returns:
        DataFrame with fold change columns (e.g., "{case}_R1/TP", "{case}_R2/TP")

    Example:
        >>> fc_df = compute_temporal_fold_change(expr_df, "aggressive")
        >>> # Returns DataFrame with columns: aggressive_R1/TP, aggressive_R2/TP
    """
    if timepoint_pairs is None:
        timepoint_pairs = [("R1", "TP"), ("R2", "TP")]

    result = pd.DataFrame(index=expression_df.index)

    for num_tp, denom_tp in timepoint_pairs:
        num_col = f"{case_name}_{num_tp}"
        denom_col = f"{case_name}_{denom_tp}"
        fc_col = f"{case_name}_{num_tp}/{denom_tp}"

        if num_col not in expression_df.columns:
            logger.warning(f"Column '{num_col}' not found, skipping")
            continue
        if denom_col not in expression_df.columns:
            logger.warning(f"Column '{denom_col}' not found, skipping")
            continue

        result[fc_col] = compute_fold_change(
            expression_df, num_col, denom_col, pseudocount=pseudocount
        )

    logger.info(
        f"Computed temporal fold changes for {case_name}: {list(result.columns)}"
    )
    return result


def select_degs(
    fold_changes: pd.Series,
    up_threshold: float = DEFAULT_UP_THRESHOLD,
    down_threshold: float = DEFAULT_DOWN_THRESHOLD,
    return_direction: bool = True,
) -> Union[Set[str], pd.DataFrame]:
    """
    Select differentially expressed genes based on fold change thresholds.

    Args:
        fold_changes: Series of fold change values indexed by gene symbol
        up_threshold: Threshold for up-regulation (FC >= threshold)
        down_threshold: Threshold for down-regulation (FC <= threshold)
        return_direction: If True, return DataFrame with direction; else return set of genes

    Returns:
        If return_direction=False: Set of gene symbols
        If return_direction=True: DataFrame with columns [gene_symbol, fold_change, direction]

    Example:
        >>> degs = select_degs(fc_series, up_threshold=1.5, down_threshold=0.5)
        >>> # Returns: {"BRCA1", "TP53", ...} or DataFrame
    """
    # Compute down threshold if not provided
    if down_threshold is None:
        down_threshold = abs(1 - (up_threshold - 1))

    up_genes = set(fold_changes[fold_changes >= up_threshold].index)
    down_genes = set(fold_changes[fold_changes <= down_threshold].index)

    logger.info(
        f"Selected DEGs: {len(up_genes)} up-regulated (FC >= {up_threshold}), "
        f"{len(down_genes)} down-regulated (FC <= {down_threshold})"
    )

    if not return_direction:
        return up_genes | down_genes

    # Build DataFrame with directions
    deg_list = []
    for gene in up_genes:
        deg_list.append(
            {
                "gene_symbol": gene,
                "fold_change": fold_changes[gene],
                "direction": "up",
            }
        )
    for gene in down_genes:
        deg_list.append(
            {
                "gene_symbol": gene,
                "fold_change": fold_changes[gene],
                "direction": "down",
            }
        )

    return pd.DataFrame(deg_list)


def select_degs_from_dataframe(
    fc_df: pd.DataFrame,
    fc_column: str,
    up_threshold: float = DEFAULT_UP_THRESHOLD,
    down_threshold: float = DEFAULT_DOWN_THRESHOLD,
) -> Set[str]:
    """
    Select DEGs from a DataFrame containing fold change values.

    Args:
        fc_df: DataFrame with fold change values (genes as index)
        fc_column: Column name containing fold change values
        up_threshold: Threshold for up-regulation
        down_threshold: Threshold for down-regulation

    Returns:
        Set of gene symbols meeting threshold criteria
    """
    if fc_column not in fc_df.columns:
        raise ValueError(f"Column '{fc_column}' not found in DataFrame")

    return select_degs(
        fc_df[fc_column],
        up_threshold=up_threshold,
        down_threshold=down_threshold,
        return_direction=False,
    )


def compute_deg_stats(
    expression_df: pd.DataFrame,
    group1_cols: List[str],
    group2_cols: List[str],
    method: str = "ttest",
) -> pd.DataFrame:
    """
    Compute differential expression statistics between two groups.

    This implementation provides basic differential expression analysis.
    For comprehensive differential expression analysis workflows,
    consider utilizing established tools such as DESeq2 or edgeR.

    Args:
        expression_df: Expression matrix (genes × samples)
        group1_cols: Column names for group 1
        group2_cols: Column names for group 2
        method: Statistical test ("ttest" or "wilcoxon")

    Returns:
        DataFrame with columns: gene_symbol, fold_change, p_value, direction
    """
    from scipy import stats

    results = []

    for gene in expression_df.index:
        g1_vals = expression_df.loc[gene, group1_cols].values.astype(float)
        g2_vals = expression_df.loc[gene, group2_cols].values.astype(float)

        # Compute fold change (mean ratio)
        mean1 = np.mean(g1_vals)
        mean2 = np.mean(g2_vals)
        fc = mean1 / (mean2 + DEFAULT_PSEUDOCOUNT)

        # Compute p-value
        if method == "ttest":
            try:
                _, p_value = stats.ttest_ind(g1_vals, g2_vals)
            except Exception:
                p_value = 1.0
        elif method == "wilcoxon":
            try:
                _, p_value = stats.mannwhitneyu(g1_vals, g2_vals)
            except Exception:
                p_value = 1.0
        else:
            raise ValueError(f"Unknown method: {method}")

        # Determine direction
        if fc >= DEFAULT_UP_THRESHOLD:
            direction = "up"
        elif fc <= DEFAULT_DOWN_THRESHOLD:
            direction = "down"
        else:
            direction = "neutral"

        results.append(
            {
                "gene_symbol": gene,
                "fold_change": fc,
                "p_value": p_value,
                "direction": direction,
            }
        )

    df = pd.DataFrame(results)

    # Compute FDR (Benjamini-Hochberg)
    from scipy.stats import false_discovery_control

    try:
        df["adjusted_p_value"] = false_discovery_control(df["p_value"])
    except Exception:
        # Fallback for older scipy versions
        df["adjusted_p_value"] = (
            df["p_value"] * len(df) / (df["p_value"].rank(method="first"))
        )
        df["adjusted_p_value"] = df["adjusted_p_value"].clip(upper=1.0)

    logger.info(f"Computed DE stats for {len(df)} genes")
    return df


def filter_expressed_genes(
    expression_df: pd.DataFrame,
    min_value: float = 0.001,
    min_fraction: float = 1.0,
) -> pd.DataFrame:
    """
    Filter genes based on minimum expression threshold.

    Args:
        expression_df: Expression matrix (genes × samples)
        min_value: Minimum expression value
        min_fraction: Fraction of samples that must meet threshold (1.0 = all)

    Returns:
        Filtered expression DataFrame
    """
    if min_fraction == 1.0:
        # All samples must have expression >= min_value
        mask = (expression_df >= min_value).all(axis=1)
    else:
        # At least min_fraction of samples must have expression >= min_value
        mask = (expression_df >= min_value).mean(axis=1) >= min_fraction

    filtered = expression_df.loc[mask]

    logger.info(
        f"Filtered genes: {len(expression_df)} -> {len(filtered)} "
        f"(min_value={min_value}, min_fraction={min_fraction})"
    )
    return filtered


def get_case_samples(
    expression_df: pd.DataFrame,
    patient_ids: List[str],
    case_name: str,
    id_position: int = 2,
    timepoint_position: int = 3,
) -> pd.DataFrame:
    """
    Extract and aggregate samples for a specific case/condition.

    This matches the notebook logic for selecting patient samples.

    Args:
        expression_df: Full expression matrix
        patient_ids: List of patient identifiers to include
        case_name: Name for the output columns (e.g., "baseline")
        id_position: Position of patient ID in column name (split by '.')
        timepoint_position: Position of timepoint in column name

    Returns:
        DataFrame with aggregated expression values
        Columns: {case_name}_R1, {case_name}_R2, {case_name}_TP (or similar)
    """
    # Select columns matching patient IDs
    selected_cols = []
    for col in expression_df.columns:
        parts = col.split(".")
        if len(parts) > id_position:
            if parts[id_position] in patient_ids:
                selected_cols.append(col)

    if not selected_cols:
        raise ValueError(f"No columns found for patient IDs: {patient_ids}")

    case_df = expression_df[sorted(selected_cols)]

    # Rename columns to {case_name}_{timepoint}
    new_cols = {}
    for col in case_df.columns:
        parts = col.split(".")
        if len(parts) > timepoint_position:
            timepoint = parts[timepoint_position]
            new_cols[col] = f"{case_name}_{timepoint}"

    case_df = case_df.rename(columns=new_cols)

    # Aggregate duplicates by median
    case_df = case_df.groupby(case_df.columns, axis=1).median()

    logger.info(f"Extracted {case_name} samples: {list(case_df.columns)}")
    return case_df


def save_deg_sets(
    deg_sets: List[Set[str]],
    path: str,
    names: Optional[List[str]] = None,
) -> None:
    """
    Save DEG sets to a pickle file.

    Args:
        deg_sets: List of gene sets
        path: Output path
        names: Optional names for each set
    """
    import pickle
    from pathlib import Path

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(deg_sets, f)

    logger.info(f"Saved {len(deg_sets)} DEG sets to {path}")


def load_deg_sets(path: str) -> List[Set[str]]:
    """
    Load DEG sets from a pickle file.

    Args:
        path: Path to pickle file

    Returns:
        List of gene sets
    """
    import pickle

    with open(path, "rb") as f:
        deg_sets = pickle.load(f)

    logger.info(f"Loaded {len(deg_sets)} DEG sets from {path}")
    return deg_sets
