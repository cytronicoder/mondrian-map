# Mondrian Map Core Module
"""
Mondrian Map: Pathway Visualization for Biological Data

This package provides a complete pipeline for pathway enrichment analysis
and visualization using Mondrian-style treemaps.

Example:
    >>> from mondrian_map import reproduce_case_study
    >>> outputs = reproduce_case_study("gbm", "outputs/")
"""

# New pipeline modules
from .config import (CaseStudyConfig, EmbeddingConfig, PipelineConfig,
                     ThresholdConfig, TSNEConfig, VisualizationConfig,
                     get_gbm_case_study_config)
from .core import Block, Colors, Corner, GridSystem, Line, blank_canvas
from .data_processing import (get_areas, get_colors, get_IDs,
                              get_mondrian_color_description, get_points,
                              load_dataset, load_pathway_info)
from .degs import (compute_fold_change, compute_temporal_fold_change,
                   select_degs)
from .embeddings import (EmbeddingGenerator, build_pathway_description_prompts,
                         build_pathway_name_prompts)
from .io import (load_deg_table, load_embeddings, load_expression_matrix,
                 save_entities, save_manifest, save_relations)
from .pathway_stats import (build_entities_table, compute_pathway_wfc,
                            compute_wfc)
from .pipeline import (MondrianMapPipeline, PipelineOutputs,
                       reproduce_case_study, run_case_study, run_pipeline)
from .projection import (normalize_coordinates, project_tsne, tsne_project,
                         umap_project)

# Visualization module requires plotly - import only when needed
# from .visualization import create_authentic_mondrian_map, create_canvas_grid, create_color_legend

__version__ = "1.2.1"
__all__ = [
    # Core classes
    "GridSystem",
    "Block",
    "Line",
    "Corner",
    "Colors",
    "blank_canvas",
    # Data processing
    "get_points",
    "get_areas",
    "get_colors",
    "get_IDs",
    "load_pathway_info",
    "load_dataset",
    "get_mondrian_color_description",
    # Configuration
    "PipelineConfig",
    "ThresholdConfig",
    "EmbeddingConfig",
    "TSNEConfig",
    "VisualizationConfig",
    "CaseStudyConfig",
    "get_gbm_case_study_config",
    # Pipeline
    "MondrianMapPipeline",
    "PipelineOutputs",
    "run_pipeline",
    "reproduce_case_study",
    "run_case_study",
    # I/O
    "load_expression_matrix",
    "load_deg_table",
    "save_entities",
    "save_relations",
    "load_embeddings",
    "save_manifest",
    # DEG analysis
    "compute_fold_change",
    "compute_temporal_fold_change",
    "select_degs",
    # Pathway statistics
    "compute_wfc",
    "compute_pathway_wfc",
    "build_entities_table",
    # Embeddings
    "EmbeddingGenerator",
    "build_pathway_name_prompts",
    "build_pathway_description_prompts",
    # Projection
    "tsne_project",
    "umap_project",
    "normalize_coordinates",
    "project_tsne",
]
