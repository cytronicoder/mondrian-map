"""
Projection Module for Mondrian Map

This module handles dimensionality reduction (t-SNE, UMAP) for pathway embeddings
to generate 2D coordinates for visualization.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default t-SNE parameters (matching notebook)
DEFAULT_TSNE_PERPLEXITY = 30.0
DEFAULT_TSNE_LEARNING_RATE = 200.0
DEFAULT_TSNE_N_ITER = 1000
DEFAULT_TSNE_METRIC = "euclidean"
DEFAULT_SEED = 42


@dataclass
class TSNEConfig:
    """Configuration for t-SNE projection."""

    perplexity: float = DEFAULT_TSNE_PERPLEXITY
    learning_rate: float = DEFAULT_TSNE_LEARNING_RATE
    n_iter: int = DEFAULT_TSNE_N_ITER
    metric: str = DEFAULT_TSNE_METRIC
    random_state: int = DEFAULT_SEED
    init: str = "pca"
    n_components: int = 2
    verbose: int = 0


@dataclass
class UMAPConfig:
    """Configuration for UMAP projection."""

    n_neighbors: int = 15
    min_dist: float = 0.1
    n_components: int = 2
    metric: str = "euclidean"
    random_state: int = DEFAULT_SEED


def tsne_project(
    embeddings: np.ndarray,
    seed: int = DEFAULT_SEED,
    perplexity: float = DEFAULT_TSNE_PERPLEXITY,
    learning_rate: float = DEFAULT_TSNE_LEARNING_RATE,
    n_iter: int = DEFAULT_TSNE_N_ITER,
    metric: str = DEFAULT_TSNE_METRIC,
    config: Optional[TSNEConfig] = None,
) -> np.ndarray:
    """
    Project embeddings to 2D using t-SNE.

    Args:
        embeddings: Input embeddings array (n_samples, n_features)
        seed: Random seed for reproducibility
        perplexity: t-SNE perplexity parameter
        learning_rate: t-SNE learning rate
        n_iter: Number of iterations
        metric: Distance metric
        config: TSNEConfig object (overrides individual params)

    Returns:
        2D coordinates array (n_samples, 2)

    Example:
        >>> coords = tsne_project(embeddings, seed=42)
        >>> # coords.shape == (n_samples, 2)
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        raise ImportError(
            "scikit-learn not installed. Install with: pip install scikit-learn"
        )

    # Use config if provided
    if config is not None:
        seed = config.random_state
        perplexity = config.perplexity
        learning_rate = config.learning_rate
        n_iter = config.n_iter
        metric = config.metric

    logger.info(
        f"Running t-SNE: perplexity={perplexity}, learning_rate={learning_rate}, "
        f"n_iter={n_iter}, seed={seed}"
    )

    # Adjust perplexity if too high for sample size
    n_samples = embeddings.shape[0]
    if perplexity >= n_samples:
        old_perplexity = perplexity
        perplexity = max(1, n_samples - 1)
        logger.warning(
            f"Reduced perplexity from {old_perplexity} to {perplexity} "
            f"(n_samples={n_samples})"
        )

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        n_iter=n_iter,
        metric=metric,
        random_state=seed,
        init="pca" if n_samples > 3 else "random",
    )

    coords = tsne.fit_transform(embeddings)

    logger.info(f"t-SNE projection complete: {coords.shape}")
    return coords


def umap_project(
    embeddings: np.ndarray,
    seed: int = DEFAULT_SEED,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    config: Optional[UMAPConfig] = None,
) -> np.ndarray:
    """
    Project embeddings to 2D using UMAP.

    Args:
        embeddings: Input embeddings array (n_samples, n_features)
        seed: Random seed for reproducibility
        n_neighbors: Number of neighbors for UMAP
        min_dist: Minimum distance parameter
        metric: Distance metric
        config: UMAPConfig object (overrides individual params)

    Returns:
        2D coordinates array (n_samples, 2)
    """
    try:
        import umap
    except ImportError:
        raise ImportError(
            "umap-learn not installed. Install with: pip install umap-learn"
        )

    if config is not None:
        seed = config.random_state
        n_neighbors = config.n_neighbors
        min_dist = config.min_dist
        metric = config.metric

    logger.info(
        f"Running UMAP: n_neighbors={n_neighbors}, min_dist={min_dist}, seed={seed}"
    )

    # Adjust n_neighbors if too high
    n_samples = embeddings.shape[0]
    if n_neighbors >= n_samples:
        n_neighbors = max(2, n_samples - 1)
        logger.warning(f"Reduced n_neighbors to {n_neighbors} (n_samples={n_samples})")

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
    )

    coords = reducer.fit_transform(embeddings)

    logger.info(f"UMAP projection complete: {coords.shape}")
    return coords


def normalize_coordinates(
    coords: np.ndarray,
    range_min: float = 0.05,
    range_max: float = 0.95,
    canvas_size: float = 1000.0,
) -> np.ndarray:
    """
    Normalize coordinates to a specified range for visualization.

    Args:
        coords: 2D coordinates (n_samples, 2)
        range_min: Minimum value of normalized range (fraction)
        range_max: Maximum value of normalized range (fraction)
        canvas_size: Canvas size to scale to

    Returns:
        Normalized and scaled coordinates
    """
    try:
        from sklearn.preprocessing import MinMaxScaler
    except ImportError:
        # Fallback implementation
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()

        normalized = np.zeros_like(coords)
        if x_max > x_min:
            normalized[:, 0] = (coords[:, 0] - x_min) / (x_max - x_min)
        if y_max > y_min:
            normalized[:, 1] = (coords[:, 1] - y_min) / (y_max - y_min)

        # Scale to range
        normalized = normalized * (range_max - range_min) + range_min

        # Scale to canvas
        return normalized * canvas_size

    scaler = MinMaxScaler(feature_range=(range_min, range_max))
    normalized = scaler.fit_transform(coords)

    # Scale to canvas size
    scaled = normalized * canvas_size

    logger.debug(
        f"Normalized coordinates: x=[{scaled[:, 0].min():.1f}, {scaled[:, 0].max():.1f}], "
        f"y=[{scaled[:, 1].min():.1f}, {scaled[:, 1].max():.1f}]"
    )

    return scaled


def project_embeddings(
    embeddings: np.ndarray,
    method: str = "tsne",
    seed: int = DEFAULT_SEED,
    normalize: bool = True,
    canvas_size: float = 1000.0,
    **kwargs,
) -> np.ndarray:
    """
    Project embeddings to 2D using specified method.

    Args:
        embeddings: Input embeddings (n_samples, n_features)
        method: Projection method ("tsne" or "umap")
        seed: Random seed
        normalize: Whether to normalize coordinates
        canvas_size: Canvas size for normalization
        **kwargs: Additional arguments for the projection method

    Returns:
        2D coordinates (n_samples, 2)
    """
    method = method.lower()

    if method == "tsne":
        coords = tsne_project(embeddings, seed=seed, **kwargs)
    elif method == "umap":
        coords = umap_project(embeddings, seed=seed, **kwargs)
    else:
        raise ValueError(f"Unknown projection method: {method}")

    if normalize:
        coords = normalize_coordinates(coords, canvas_size=canvas_size)

    return coords


def build_coordinates_table(
    coords: np.ndarray,
    pathway_ids: list,
    x_col: str = "x",
    y_col: str = "y",
) -> pd.DataFrame:
    """
    Build a DataFrame with pathway coordinates.

    Args:
        coords: 2D coordinates (n_samples, 2)
        pathway_ids: List of pathway identifiers
        x_col: Name for x coordinate column
        y_col: Name for y coordinate column

    Returns:
        DataFrame with columns: GS_ID, x, y
    """
    if len(coords) != len(pathway_ids):
        raise ValueError(
            f"Coordinate count ({len(coords)}) doesn't match "
            f"pathway count ({len(pathway_ids)})"
        )

    df = pd.DataFrame(
        {
            "GS_ID": pathway_ids,
            x_col: coords[:, 0],
            y_col: coords[:, 1],
        }
    )

    logger.info(f"Built coordinates table for {len(df)} pathways")
    return df


def verify_determinism(
    embeddings: np.ndarray,
    seed: int,
    method: str = "tsne",
    tolerance: float = 1e-6,
) -> bool:
    """
    Verify that projection is deterministic with given seed.

    Args:
        embeddings: Input embeddings
        seed: Random seed to test
        method: Projection method
        tolerance: Maximum allowed difference

    Returns:
        True if projections are identical across runs
    """
    coords1 = project_embeddings(embeddings, method=method, seed=seed, normalize=False)
    coords2 = project_embeddings(embeddings, method=method, seed=seed, normalize=False)

    max_diff = np.abs(coords1 - coords2).max()
    is_deterministic = max_diff < tolerance

    if is_deterministic:
        logger.info(f"Projection is deterministic (max_diff={max_diff:.2e})")
    else:
        logger.warning(
            f"Projection NOT deterministic (max_diff={max_diff:.2e} > {tolerance})"
        )

    return is_deterministic


def compute_projection_quality(
    embeddings: np.ndarray,
    coords: np.ndarray,
    n_neighbors: int = 10,
) -> dict:
    """
    Compute quality metrics for dimensionality reduction.

    Args:
        embeddings: Original high-dimensional embeddings
        coords: Projected 2D coordinates
        n_neighbors: Number of neighbors for trustworthiness

    Returns:
        Dictionary with quality metrics
    """
    try:
        from sklearn.manifold import trustworthiness
        from sklearn.metrics import pairwise_distances
    except ImportError:
        logger.warning("scikit-learn not available for quality metrics")
        return {}

    # Adjust n_neighbors if needed
    n_samples = embeddings.shape[0]
    if n_neighbors >= n_samples:
        n_neighbors = max(1, n_samples // 2)

    # Compute trustworthiness (how well local structure is preserved)
    trust = trustworthiness(embeddings, coords, n_neighbors=n_neighbors)

    # Compute correlation between high-dim and low-dim distances
    hd_distances = pairwise_distances(embeddings).flatten()
    ld_distances = pairwise_distances(coords).flatten()

    correlation = np.corrcoef(hd_distances, ld_distances)[0, 1]

    return {
        "trustworthiness": trust,
        "distance_correlation": correlation,
        "n_samples": n_samples,
        "n_neighbors": n_neighbors,
    }
