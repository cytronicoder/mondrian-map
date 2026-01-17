"""
Pipeline Module for Mondrian Map

This module orchestrates the complete workflow from data loading through
visualization output.
"""

import hashlib
import json
import logging
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .config import PipelineConfig, get_gbm_case_study_config
from .io import (ensure_directory, load_embeddings, load_pathway_info,
                 save_embeddings, save_entities, save_manifest, save_relations)

logger = logging.getLogger(__name__)


@dataclass
class PipelineOutputs:
    """Container for pipeline outputs."""

    entities_df: pd.DataFrame
    relations_df: pd.DataFrame
    embeddings: np.ndarray
    coordinates: np.ndarray
    pathway_ids: List[str]
    pathway_info: Dict[str, Any]
    config: PipelineConfig
    manifest: Dict[str, Any]
    figure_path: Optional[Path] = None
    html_path: Optional[Path] = None


@dataclass
class CacheInfo:
    """Information about cached artifacts."""

    pager_cache_hits: int = 0
    embedding_cache_hit: bool = False
    tsne_cache_hit: bool = False
    entities_from_cache: bool = False


def get_git_commit_hash() -> Optional[str]:
    """Get current git commit hash if in a git repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()[:8]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def compute_config_hash(config: PipelineConfig) -> str:
    """Compute a hash of the configuration for caching."""
    config_str = json.dumps(asdict(config), sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


class MondrianMapPipeline:
    """
    Main pipeline for generating Mondrian Map visualizations.

    The pipeline consists of these steps:
    1. Load data / DEGs
    2. Call PAGER GNPA to get significant WikiPathways PAGs
    3. Get RP-ranked genes per pathway
    4. Compute pathway-level wFC + pFDR
    5. Build pathway text prompts
    6. Embed + normalize
    7. t-SNE → x, y coordinates
    8. Fetch PAG-PAG edges (m-type)
    9. Write entities + relations tables
    10. Generate visualization

    Example:
        >>> config = PipelineConfig()
        >>> pipeline = MondrianMapPipeline(config)
        >>> outputs = pipeline.run()
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize pipeline.

        Args:
            config: Pipeline configuration. Uses defaults if not provided.
        """
        self.config = config or PipelineConfig()
        self.cache_info = CacheInfo()

        log_level = logging.DEBUG if self.config.debug else logging.INFO
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        self._pager_client = None

    @property
    def pager_client(self):
        """Lazily initialize PAGER client."""
        if self._pager_client is None:
            from .pager_client import PagerClient
            from .pager_client import PagerConfig as PagerClientConfig

            pager_config = PagerClientConfig(
                cache_dir=(
                    Path(self.config.cache_dir) / "pager"
                    if self.config.use_cache
                    else None
                ),
                use_cache=self.config.use_cache,
                rate_limit=self.config.pager.rate_limit,
                max_retries=self.config.pager.max_retries,
                retry_delay=self.config.pager.retry_delay,
            )
            self._pager_client = PagerClient(pager_config)

        return self._pager_client

    def _log_step(self, step_num: int, message: str):
        """Log a pipeline step."""
        logger.info(f"[Step {step_num}/10] {message}")

    def run(
        self,
        use_cache: Optional[bool] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> PipelineOutputs:
        """
        Run the complete pipeline.

        Args:
            use_cache: Override config.use_cache
            output_dir: Override config.output_dir

        Returns:
            PipelineOutputs containing all results
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_dir = Path(output_dir or self.config.output_dir)

        logger.info(f"Starting Mondrian Map pipeline (use_cache={use_cache})")

        ensure_directory(output_dir)
        ensure_directory(output_dir / "figures")

        entities_cache_path = (
            Path(self.config.data_dir)
            / "pathways_prepared_for_visualization"
            / f"wikipathway_{self.config.case_study.name}_R1_TP.csv"
        )

        if use_cache and self._check_cached_entities(entities_cache_path):
            return self._run_from_cache(output_dir)

        return self._run_full_pipeline(output_dir, use_cache)

    def _check_cached_entities(self, path: Path) -> bool:
        """Check if cached entities are available."""
        viz_dir = Path(self.config.data_dir) / "pathways_prepared_for_visualization"
        if viz_dir.exists():
            csv_files = list(viz_dir.glob("*.csv"))
            if csv_files:
                return True
        return False

    def _run_from_cache(self, output_dir: Path) -> PipelineOutputs:
        """Run pipeline using cached artifacts."""
        logger.info("Running pipeline from cached artifacts")
        self.cache_info.entities_from_cache = True

        self._log_step(1, "Loading pathway information")
        pathway_info = self._load_pathway_info()

        self._log_step(2, "Loading cached entities")
        viz_dir = Path(self.config.data_dir) / "pathways_prepared_for_visualization"

        csv_files = list(viz_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {viz_dir}")

        entities_path = csv_files[0]
        for f in csv_files:
            if self.config.case_study.name.lower() in f.name.lower():
                entities_path = f
                break

        entities_df = pd.read_csv(entities_path)
        logger.info(f"Loaded {len(entities_df)} entities from {entities_path}")

        if pathway_info:
            entities_df = self._enrich_entities(entities_df, pathway_info)

        self._log_step(3, "Loading cached relations")
        relations_df = self._load_cached_relations()

        self._log_step(4, "Loading cached embeddings")
        embeddings, pathway_ids = self._load_cached_embeddings(entities_df)

        coordinates = entities_df[["x", "y"]].values

        self._log_step(5, "Skipping PAGER calls (using cache)")
        self._log_step(6, "Skipping embedding generation (using cache)")
        self._log_step(7, "Skipping t-SNE projection (using cache)")

        self._log_step(8, "Filtering relations to available pathways")
        if len(relations_df) > 0:
            available_ids = set(entities_df["GS_ID"])
            relations_df = relations_df[
                relations_df["GS_A_ID"].isin(available_ids)
                & relations_df["GS_B_ID"].isin(available_ids)
            ]

        self._log_step(9, "Saving outputs")
        self._save_outputs(entities_df, relations_df, output_dir)

        self._log_step(10, "Generating visualization")
        fig_path, html_path = self._generate_visualization(
            entities_df, relations_df, output_dir, pathway_info
        )

        manifest = self._build_manifest(output_dir)

        return PipelineOutputs(
            entities_df=entities_df,
            relations_df=relations_df,
            embeddings=embeddings,
            coordinates=coordinates,
            pathway_ids=list(entities_df["GS_ID"]),
            pathway_info=pathway_info,
            config=self.config,
            manifest=manifest,
            figure_path=fig_path,
            html_path=html_path,
        )

    def _run_full_pipeline(self, output_dir: Path, use_cache: bool) -> PipelineOutputs:
        """Run complete pipeline from scratch."""
        from .embeddings import build_prompts
        from .pathway_stats import (build_entities_table,
                                    compute_pathway_wfc_batch)
        from .projection import build_coordinates_table, project_embeddings

        # Step 1: Load expression data and compute DEGs
        self._log_step(1, "Loading data and selecting DEGs")
        fc_df, deg_sets = self._compute_degs()

        self._log_step(2, "Running PAGER GNPA analysis")
        pag_results = self._run_gnpa(deg_sets)

        self._log_step(3, "Getting pathway ranked genes")
        all_pag_ids = set()
        for pag_df in pag_results.values():
            all_pag_ids.update(pag_df["GS_ID"].tolist())

        self._log_step(4, "Computing pathway weighted fold changes")
        wfc_results = {}
        for condition, pag_df in pag_results.items():
            fc_col = f"{condition.split('_')[0]}_{condition.split('_')[1]}/TP"
            wfc_df = compute_pathway_wfc_batch(
                list(pag_df["GS_ID"].unique()),
                pd.DataFrame(),
                fc_df,
                fc_col,
                self.pager_client,
                pag_df,
                progress_bar=self.config.verbose,
            )
            wfc_results[condition] = wfc_df

        self._log_step(5, "Building pathway prompts")
        self._log_step(5, "Building pathway prompts")
        pathway_info = self._load_pathway_info()
        prompts = build_prompts(
            list(all_pag_ids),
            self.config.embedding.prompt_type,
            pathway_info=pathway_info,
            pager_client=self.pager_client,
            max_genes=self.config.embedding.max_genes,
        )

        # Step 6: Generate embeddings
        self._log_step(6, "Generating pathway embeddings")
        embeddings, valid_ids = self._generate_embeddings(prompts)

        # Step 7: Project to 2D
        self._log_step(7, "Projecting embeddings to 2D coordinates")
        coordinates = project_embeddings(
            embeddings,
            method="tsne",
            seed=self.config.random_seed,
            perplexity=self.config.tsne.perplexity,
            learning_rate=self.config.tsne.learning_rate,
            n_iter=self.config.tsne.n_iter,
            normalize=self.config.tsne.normalize_coords,
            canvas_size=self.config.tsne.canvas_size,
        )
        coords_df = build_coordinates_table(coordinates, valid_ids)

        # Step 8: Fetch PAG-PAG network
        self._log_step(8, "Fetching PAG-PAG network")
        relations_df = self.pager_client.get_pag_pag_network(
            valid_ids, network_type="m-type"
        )

        # Step 9: Build and save outputs
        self._log_step(9, "Building and saving outputs")

        # Merge wFC with coordinates for each condition
        all_entities = []
        for condition, wfc_df in wfc_results.items():
            entities = build_entities_table(wfc_df, coords_df, pathway_info)
            entities["condition"] = condition
            all_entities.append(entities)

        # Use first condition as primary
        if all_entities:
            entities_df = all_entities[0]
        else:
            entities_df = pd.DataFrame(columns=["GS_ID", "wFC", "pFDR", "x", "y"])

        self._save_outputs(entities_df, relations_df, output_dir)

        # Step 10: Generate visualization
        self._log_step(10, "Generating visualization")
        fig_path, html_path = self._generate_visualization(
            entities_df, relations_df, output_dir, pathway_info
        )

        # Build manifest
        manifest = self._build_manifest(output_dir)

        return PipelineOutputs(
            entities_df=entities_df,
            relations_df=relations_df,
            embeddings=embeddings,
            coordinates=coordinates,
            pathway_ids=valid_ids,
            pathway_info=pathway_info,
            config=self.config,
            manifest=manifest,
            figure_path=fig_path,
            html_path=html_path,
        )

    def _load_pathway_info(self) -> Dict[str, Any]:
        """Load pathway annotation information."""
        info_path = (
            Path(self.config.data_dir)
            / "pathway_details"
            / "annotations_with_summary.json"
        )
        if info_path.exists():
            return load_pathway_info(info_path)
        logger.warning(f"Pathway info not found at {info_path}")
        return {}

    def _compute_degs(self) -> Tuple[pd.DataFrame, Dict[str, set]]:
        """Compute DEGs from expression data or load from cache."""

        # Try to load cached DEG sets
        deg_cache_path = (
            Path(self.config.data_dir) / "differentially_expressed_genes" / "DEGs.pkl"
        )

        if self.config.use_cache and deg_cache_path.exists():
            from .degs import load_deg_sets

            deg_sets_list = load_deg_sets(str(deg_cache_path))
            # Map to conditions
            conditions = [
                "baseline_R1",
                "baseline_R2",
                "aggressive_R1",
                "aggressive_R2",
                "nonaggressive_R1",
                "nonaggressive_R2",
            ]
            deg_sets = dict(zip(conditions, deg_sets_list))

            # Return empty fold-change dataframe since it is not required with cached data
            return pd.DataFrame(), deg_sets

        # Future implementation: Compute full differential gene expression from raw data
        raise NotImplementedError(
            "Full differential gene expression computation requires raw expression data. "
            "Use --use-cache to execute pipeline using cached artifacts."
        )

    def _run_gnpa(self, deg_sets: Dict[str, set]) -> Dict[str, pd.DataFrame]:
        """Run PAGER GNPA for all DEG sets."""
        # Try to load cached GNPA results
        gnpa_dir = Path(self.config.data_dir) / "geneset_network_and_pathway_analysis"

        if self.config.use_cache and gnpa_dir.exists():
            results = {}
            for condition in deg_sets.keys():
                csv_path = gnpa_dir / f"wikipathway_{condition}_TP.csv"
                if csv_path.exists():
                    results[condition] = pd.read_csv(csv_path)
                    logger.info(
                        f"Loaded {len(results[condition])} PAGs for {condition}"
                    )
            if results:
                return results

        # Run GNPA for each condition
        results = {}
        for condition, genes in deg_sets.items():
            pag_df = self.pager_client.get_significant_pags(
                list(genes),
                source=self.config.pager.source,
                pval_thresh=self.config.pager.pvalue,
                fdr_thresh=self.config.pager.fdr,
            )
            results[condition] = pag_df
            logger.info(f"GNPA returned {len(pag_df)} PAGs for {condition}")

        return results

    def _generate_embeddings(
        self, prompts: Dict[str, Any]
    ) -> Tuple[np.ndarray, List[str]]:
        """Generate embeddings for pathways."""
        from .embeddings import EmbeddingConfig, EmbeddingGenerator

        # Check for cached embeddings
        embedding_cache_path = (
            Path(self.config.data_dir)
            / "embeddings"
            / f"{self.config.embedding.model_name.replace('/', '_')}_{self.config.embedding.prompt_type}.npy"
        )

        if self.config.use_cache and embedding_cache_path.exists():
            self.cache_info.embedding_cache_hit = True
            embeddings = load_embeddings(embedding_cache_path)
            # Need to get pathway IDs from prompts
            valid_ids = [pid for pid, p in prompts.items() if p]
            return embeddings, valid_ids

        # Generate new embeddings
        config = EmbeddingConfig(
            model_name=self.config.embedding.model_name,
            model_type=self.config.embedding.model_type,
            batch_size=self.config.embedding.batch_size,
            normalize=self.config.embedding.normalize,
        )
        generator = EmbeddingGenerator(config)

        # Filter empty prompts
        valid_ids = [pid for pid, p in prompts.items() if p]
        texts = [prompts[pid] for pid in valid_ids]

        embeddings = generator.embed_texts(texts)

        # Cache embeddings
        if self.config.use_cache:
            save_embeddings(embeddings, embedding_cache_path)

        return embeddings, valid_ids

    def _load_cached_embeddings(
        self, entities_df: pd.DataFrame
    ) -> Tuple[np.ndarray, List[str]]:
        """Load cached embeddings matching the entities."""
        embeddings_dir = Path(self.config.data_dir) / "embeddings"

        # Try to find matching embeddings file
        for npy_file in embeddings_dir.glob("*.npy"):
            embeddings = load_embeddings(npy_file)
            # Check if dimensions match
            if embeddings.shape[0] >= len(entities_df):
                pathway_ids = entities_df["GS_ID"].tolist()
                return embeddings[: len(pathway_ids)], pathway_ids

        # Return dummy embeddings if not found
        logger.warning("No matching embeddings found, using dummy values")
        n_entities = len(entities_df)
        return np.zeros((n_entities, 768)), entities_df["GS_ID"].tolist()

    def _load_cached_relations(self) -> pd.DataFrame:
        """Load cached pathway network relations."""
        network_dir = Path(self.config.data_dir) / "pathway_networks"

        if network_dir.exists():
            csv_files = list(network_dir.glob("*.csv"))
            if csv_files:
                # Load and combine all network files
                dfs = [pd.read_csv(f) for f in csv_files]
                if dfs:
                    return pd.concat(dfs, ignore_index=True).drop_duplicates()

        # Try differential pathway analysis directory
        diff_dir = Path(self.config.data_dir) / "differential_pathway_analysis"
        if diff_dir.exists():
            return pd.DataFrame(columns=["GS_A_ID", "GS_B_ID"])

        return pd.DataFrame(columns=["GS_A_ID", "GS_B_ID"])

    def _enrich_entities(
        self, entities_df: pd.DataFrame, pathway_info: Dict
    ) -> pd.DataFrame:
        """Enrich entities with pathway metadata."""
        df = entities_df.copy()

        if "NAME" not in df.columns:
            df["NAME"] = df["GS_ID"].map(
                lambda x: pathway_info.get(x, {}).get("NAME", x)
            )

        if "Description" not in df.columns:
            df["Description"] = df["GS_ID"].map(
                lambda x: pathway_info.get(x, {}).get("Description", "")
            )

        return df

    def _save_outputs(
        self,
        entities_df: pd.DataFrame,
        relations_df: pd.DataFrame,
        output_dir: Path,
    ):
        """Save pipeline outputs to files."""
        save_entities(entities_df, output_dir / "entities.csv")
        save_relations(relations_df, output_dir / "relations.csv")

    def _generate_visualization(
        self,
        entities_df: pd.DataFrame,
        relations_df: pd.DataFrame,
        output_dir: Path,
        pathway_info: Dict,
    ) -> Tuple[Optional[Path], Optional[Path]]:
        """Generate Mondrian Map visualization."""
        try:
            from .data_processing import get_relations
            from .visualization import create_authentic_mondrian_map

            # Apply max relations limit
            if (
                self.config.visualization.max_relations_per_node is not None
                and len(relations_df) > 0
            ):
                relations_df = get_relations(
                    relations_df,
                    threshold=self.config.visualization.max_relations_per_node,
                )
            else:
                pass

            # Create figure
            fig = create_authentic_mondrian_map(
                entities_df,
                dataset_name=f"Mondrian Map - {self.config.case_study.name.upper()}",
                mem_df=relations_df if len(relations_df) > 0 else None,
                maximize=self.config.visualization.maximize,
                show_pathway_ids=self.config.visualization.show_ids,
            )

            # Save outputs
            html_path = output_dir / "mondrian_map.html"
            fig.write_html(str(html_path))
            logger.info(f"Saved HTML visualization to {html_path}")

            # Save PNG if possible
            fig_path = None
            try:
                fig_path = output_dir / "figures" / "mondrian_map.png"
                fig.write_image(str(fig_path), format="png", scale=2)
                logger.info(f"Saved PNG visualization to {fig_path}")
            except Exception as e:
                logger.warning(
                    "Could not save PNG (image export failed). "
                    "This typically means the 'kaleido' package is not installed or not available in your environment.\n"
                    "Install it with: pip install --upgrade kaleido\n"
                    "Or add 'kaleido>=0.2.1' to 'config/requirements.txt' and reinstall your dependencies.\n"
                    f"Original error: {e}"
                )

            return fig_path, html_path

        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            return None, None

    def _build_manifest(self, output_dir: Path) -> Dict[str, Any]:
        """Build and save manifest with run metadata."""
        manifest = {
            "timestamp": datetime.now().isoformat(),
            "version": self.config.version,
            "git_commit": get_git_commit_hash(),
            "config_hash": compute_config_hash(self.config),
            "random_seed": self.config.random_seed,
            "cache_info": {
                "pager_cache_hits": self.cache_info.pager_cache_hits,
                "embedding_cache_hit": self.cache_info.embedding_cache_hit,
                "tsne_cache_hit": self.cache_info.tsne_cache_hit,
                "entities_from_cache": self.cache_info.entities_from_cache,
            },
            "pager_endpoint": "https://discovery.informatics.uab.edu/PAGER",
            "embedding_model": self.config.embedding.model_name,
            "outputs": {
                "entities": "entities.csv",
                "relations": "relations.csv",
                "visualization": "mondrian_map.html",
            },
        }

        save_manifest(manifest, output_dir / "manifest.json")
        return manifest


def run_pipeline(config: PipelineConfig) -> PipelineOutputs:
    """
    Convenience function to run the pipeline.

    Args:
        config: Pipeline configuration

    Returns:
        Pipeline outputs
    """
    pipeline = MondrianMapPipeline(config)
    return pipeline.run()


def reproduce_case_study(
    case_study: str = "gbm",
    output_dir: str = "outputs",
    use_cache: bool = True,
) -> PipelineOutputs:
    """
    Reproduce a case study from the paper.

    Args:
        case_study: Case study name ("gbm")
        output_dir: Output directory
        use_cache: Whether to use cached artifacts

    Returns:
        Pipeline outputs
    """
    if case_study.lower() == "gbm":
        config = get_gbm_case_study_config()
    else:
        raise ValueError(f"Unknown case study: {case_study}")

    config.output_dir = output_dir
    config.use_cache = use_cache

    pipeline = MondrianMapPipeline(config)
    return pipeline.run()
