"""
Configuration Module for Mondrian Map Pipeline

This module defines configuration dataclasses for all pipeline components
with YAML serialization support.
"""

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class ThresholdConfig:
    """Thresholds for DEG selection and pathway classification."""

    up_regulation_threshold: float = 1.5
    down_regulation_threshold: float = 0.5
    expression_min_value: float = 0.001

    significance_threshold: float = 0.05
    fdr_threshold: float = 0.5

    color_up_threshold: float = 1.25
    color_down_threshold: float = 0.75


@dataclass
class PagerConfig:
    """Configuration for PAGER API calls."""

    source: str = "WikiPathways"
    pag_type: str = "P"
    organism: str = "All"

    min_size: int = 1
    max_size: int = 2000

    similarity: float = 0.05
    overlap: int = 1
    ncoco: float = 0

    # Significance
    pvalue: float = 0.05
    fdr: float = 0.5

    rate_limit: float = 1.0
    max_retries: int = 3
    retry_delay: float = 5.0

    use_cache: bool = True
    cache_dir: str = "cache/pager"


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""

    model_name: str = "all-mpnet-base-v2"
    model_type: str = "sentence_transformer"

    prompt_type: str = "pathway_description_summary"
    max_genes: int = 100

    batch_size: int = 32
    normalize: bool = True
    max_length: int = 512

    llm2vec_model: str = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"
    llm2vec_postfix: str = "supervised"
    pooling_mode: str = "mean"

    device: Optional[str] = None

    cache_dir: str = "cache/embeddings"
    summary_model: str = "none"


@dataclass
class TSNEConfig:
    """Configuration for t-SNE projection."""

    perplexity: float = 30.0
    learning_rate: float = 200.0
    n_iter: int = 1000
    metric: str = "euclidean"
    random_state: int = 42
    init: str = "pca"

    normalize_coords: bool = True
    canvas_size: float = 1000.0
    range_min: float = 0.05
    range_max: float = 0.95

    cache_dir: str = "cache/tsne"


@dataclass
class VisualizationConfig:
    """Configuration for Mondrian Map visualization."""

    width: int = 1000
    height: int = 1000
    block_width: int = 100
    block_height: int = 100

    show_ids: bool = False
    id_format: str = "last4"
    show_hover: bool = True
    maximize: bool = False

    max_relations_per_node: Optional[int] = 2
    show_relations: bool = True

    area_scalar: int = 4000

    line_width: int = 5
    thin_line_width: int = 1

    # Output formats
    output_format: str = "html"  # html, png, svg, pdf


@dataclass
class CaseStudyConfig:
    """Configuration for case study data."""

    name: str = "gbm"
    description: str = "GBM Temporal Gene Expression Case Study"

    # Patient groupings
    baseline_patient_ids: List[str] = field(
        default_factory=lambda: ["5965", "F922", "A7RK", "R064"]
    )
    aggressive_patient_ids: List[str] = field(default_factory=lambda: ["0279"])
    nonaggressive_patient_ids: List[str] = field(default_factory=lambda: ["0027"])

    # Timepoint comparisons
    timepoint_pairs: List[List[str]] = field(
        default_factory=lambda: [["R1", "TP"], ["R2", "TP"]]
    )

    # Top N pathways to visualize
    top_n_pathways: int = 10


@dataclass
class PipelineConfig:
    """Main configuration for the Mondrian Map pipeline."""

    # Sub-configurations
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    pager: PagerConfig = field(default_factory=PagerConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    tsne: TSNEConfig = field(default_factory=TSNEConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    case_study: CaseStudyConfig = field(default_factory=CaseStudyConfig)

    # Pipeline settings
    use_cache: bool = True
    debug: bool = False
    verbose: bool = True

    # I/O paths
    data_dir: str = "data/case_study"
    output_dir: str = "outputs"
    cache_dir: str = "cache"

    # Reproducibility
    random_seed: int = 42

    # Metadata
    version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        """Create configuration from dictionary."""
        # Handle nested dataclasses
        if "thresholds" in data and isinstance(data["thresholds"], dict):
            data["thresholds"] = ThresholdConfig(**data["thresholds"])
        if "pager" in data and isinstance(data["pager"], dict):
            data["pager"] = PagerConfig(**data["pager"])
        if "embedding" in data and isinstance(data["embedding"], dict):
            data["embedding"] = EmbeddingConfig(**data["embedding"])
        if "tsne" in data and isinstance(data["tsne"], dict):
            data["tsne"] = TSNEConfig(**data["tsne"])
        if "visualization" in data and isinstance(data["visualization"], dict):
            data["visualization"] = VisualizationConfig(**data["visualization"])
        if "case_study" in data and isinstance(data["case_study"], dict):
            data["case_study"] = CaseStudyConfig(**data["case_study"])

        return cls(**data)

    def to_yaml(self, path: Union[str, Path]) -> Path:
        """
        Save configuration to YAML file.

        Args:
            path: Output file path

        Returns:
            Path to saved file
        """
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML not installed. Install with: pip install pyyaml")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

        logger.info(f"Saved configuration to {path}")
        return path

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "PipelineConfig":
        """
        Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            PipelineConfig instance
        """
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML not installed. Install with: pip install pyyaml")

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        logger.info(f"Loaded configuration from {path}")
        return cls.from_dict(data)


def create_default_config(
    case_study: str = "gbm",
    output_dir: str = "outputs",
) -> PipelineConfig:
    """
    Create a default configuration for a case study.

    Args:
        case_study: Case study name ("gbm")
        output_dir: Output directory

    Returns:
        PipelineConfig with defaults
    """
    if case_study == "gbm":
        return PipelineConfig(
            case_study=CaseStudyConfig(
                name="gbm",
                description="GBM Temporal Gene Expression Case Study",
            ),
            output_dir=output_dir,
        )
    else:
        raise ValueError(f"Unknown case study: {case_study}")


def validate_config(config: PipelineConfig) -> List[str]:
    """
    Validate configuration for common issues.

    Args:
        config: Configuration to validate

    Returns:
        List of warning messages (empty if valid)
    """
    warnings = []

    # Check threshold consistency
    if config.thresholds.down_regulation_threshold >= 1.0:
        warnings.append(
            f"down_regulation_threshold ({config.thresholds.down_regulation_threshold}) "
            "should be < 1.0 for meaningful down-regulation detection"
        )

    # Check t-SNE perplexity
    if config.tsne.perplexity < 5 or config.tsne.perplexity > 50:
        warnings.append(
            f"t-SNE perplexity ({config.tsne.perplexity}) is outside typical range [5, 50]"
        )

    # Check paths
    data_dir = Path(config.data_dir)
    if not data_dir.exists():
        warnings.append(f"Data directory does not exist: {data_dir}")

    return warnings


def get_gbm_case_study_config() -> PipelineConfig:
    """
    Get the exact configuration used for the GBM case study in the paper.

    Returns:
        PipelineConfig matching paper methodology
    """
    return PipelineConfig(
        thresholds=ThresholdConfig(
            up_regulation_threshold=1.5,
            down_regulation_threshold=0.5,
            expression_min_value=0.001,
            significance_threshold=0.05,
            fdr_threshold=0.5,
            color_up_threshold=1.25,
            color_down_threshold=0.75,
        ),
        pager=PagerConfig(
            source="WikiPathway_2021",
            pag_type="P",
            min_size=1,
            max_size=2000,
            similarity=0.05,
            overlap=1,
            ncoco=0,
            pvalue=0.05,
            fdr=0.5,
        ),
        embedding=EmbeddingConfig(
            model_name="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
            model_type="llm2vec",
            prompt_type="pathway_description_summary",
            max_genes=100,
        ),
        tsne=TSNEConfig(
            perplexity=30.0,
            learning_rate=200.0,
            n_iter=1000,
            random_state=42,
        ),
        visualization=VisualizationConfig(
            show_ids=False,
            max_relations_per_node=2,
            area_scalar=4000,
        ),
        case_study=CaseStudyConfig(
            name="gbm",
            baseline_patient_ids=["5965", "F922", "A7RK", "R064"],
            aggressive_patient_ids=["0279"],
            nonaggressive_patient_ids=["0027"],
            top_n_pathways=10,
        ),
        random_seed=42,
        use_cache=True,
    )
