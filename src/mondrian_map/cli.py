"""
Command Line Interface for Mondrian Map

This module provides a comprehensive CLI with subcommands for:
- pipeline: Run the full pipeline from configuration
- reproduce: Reproduce figures from the paper
- visualize: Generate visualization from entities/relations
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd


def setup_logging(verbose: bool = False, debug: bool = False):
    """Configure logging based on verbosity settings."""
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def cmd_pipeline(args):
    """Run the full pipeline from configuration."""
    from .config import PipelineConfig
    from .pipeline import MondrianMapPipeline

    setup_logging(verbose=args.verbose, debug=args.debug)
    logger = logging.getLogger(__name__)

    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        config = PipelineConfig.from_yaml(args.config)
    else:
        logger.info("Using default configuration")
        config = PipelineConfig()

    if args.output:
        config.output_dir = args.output
    if args.use_cache is not None:
        config.use_cache = args.use_cache
    if args.seed is not None:
        config.random_seed = args.seed
    config.debug = args.debug
    config.verbose = args.verbose

    logger.info("Starting Mondrian Map pipeline")
    pipeline = MondrianMapPipeline(config)

    try:
        outputs = pipeline.run()
        print("Pipeline completed successfully.")
        print(f"   Entities: {len(outputs.entities_df)} pathways")
        print(f"   Relations: {len(outputs.relations_df)} edges")
        print(f"   Output directory: {config.output_dir}")
        if outputs.html_path:
            print(f"   Visualization: {outputs.html_path}")
        return 0
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        if args.debug:
            raise
        return 1


def cmd_reproduce(args):
    """Reproduce figures from the paper."""
    from .pipeline import reproduce_case_study

    setup_logging(verbose=True, debug=args.debug)
    logger = logging.getLogger(__name__)

    case_study = args.case_study.lower()
    output_dir = args.out or f"outputs/{case_study}"

    logger.info(f"Reproducing {case_study.upper()} case study")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Use cache: {args.use_cache}")

    try:
        outputs = reproduce_case_study(
            case_study=case_study,
            output_dir=output_dir,
            use_cache=args.use_cache,
        )

        print(f"\nSuccessfully reproduced {case_study.upper()} case study.")
        print(f"   Entities: {outputs.entities_df.shape}")
        print(f"   Output directory: {output_dir}")

        if outputs.html_path and outputs.html_path.exists():
            print(f"   Visualization: {outputs.html_path}")

        print(f"\nManifest saved to: {output_dir}/manifest.json")
        return 0

    except Exception as e:
        logger.error(f"Reproduction failed: {e}")
        if args.debug:
            raise
        return 1


def cmd_visualize(args):
    """Generate visualization from entities/relations files."""
    from .data_processing import get_relations, load_pathway_info
    from .visualization import create_authentic_mondrian_map

    setup_logging(verbose=args.verbose, debug=args.debug)
    logger = logging.getLogger(__name__)

    entities_path = Path(args.entities)
    if not entities_path.exists():
        print(f"Error: Entities file not found: {args.entities}", file=sys.stderr)
        return 1

    logger.info(f"Loading entities from {args.entities}")
    df = pd.read_csv(entities_path)

    required_cols = ["GS_ID", "wFC", "pFDR", "x", "y"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}", file=sys.stderr)
        return 1

    pathway_info = {}
    if args.pathway_info:
        pathway_info_path = Path(args.pathway_info)
        if pathway_info_path.exists():
            logger.info(f"Loading pathway info from {args.pathway_info}")
            pathway_info = load_pathway_info(pathway_info_path)
        else:
            logger.warning(f"Pathway info file not found: {args.pathway_info}")

    if pathway_info:
        df["Description"] = df["GS_ID"].map(
            lambda x: pathway_info.get(x, {}).get("Description", "")
        )
        df["NAME"] = df["GS_ID"].map(lambda x: pathway_info.get(x, {}).get("NAME", x))
    else:
        if "NAME" not in df.columns:
            df["NAME"] = df["GS_ID"]
        if "Description" not in df.columns:
            df["Description"] = ""

    mem_df = None
    if args.relations:
        relations_path = Path(args.relations)
        if relations_path.exists():
            logger.info(f"Loading relations from {args.relations}")
            mem_df = pd.read_csv(relations_path)

            threshold = args.max_relations if args.max_relations else 2
            if threshold > 0:
                mem_df = get_relations(mem_df, threshold=threshold)

    logger.info(f"Creating Mondrian Map with {len(df)} pathways")
    title = args.title or "Mondrian Map"

    try:
        if mem_df is not None:
            df.attrs["relations_df"] = mem_df
        fig = create_authentic_mondrian_map(
            df,
            dataset_name=title,
            maximize=args.maximize,
            show_pathway_ids=args.show_ids,
        )

        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_format = args.format or output_path.suffix.lstrip(".") or "html"

        logger.info(f"Saving to {args.out} ({output_format} format)")

        if output_format == "html":
            fig.write_html(str(output_path))
        elif output_format == "png":
            fig.write_image(str(output_path), format="png", scale=2)
        elif output_format == "svg":
            fig.write_image(str(output_path), format="svg")
        elif output_format == "pdf":
            fig.write_image(str(output_path), format="pdf")
        else:
            print(f"Error: Unknown format: {output_format}", file=sys.stderr)
            return 1

        print(f"Mondrian Map saved to {args.out}")
        return 0

    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        if args.debug:
            raise
        return 1


def cmd_reproduce_case_study(args):
    """Reproduce the GLASS case study end-to-end."""
    from .config import PipelineConfig
    from .pipeline import run_case_study

    setup_logging(verbose=True, debug=args.debug)
    logger = logging.getLogger(__name__)

    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        config = PipelineConfig.from_yaml(args.config)
    else:
        config = PipelineConfig()

    try:
        run_case_study(
            config,
            glass_tp_path=args.tp,
            glass_r1_path=args.r1,
            glass_r2_path=args.r2,
            out_dir=args.out,
            force=args.force,
        )
        print("Case study reproduced successfully.")
        print(f"Output directory: {args.out}")
        return 0
    except Exception as e:
        logger.error(f"Case study reproduction failed: {e}")
        if args.debug:
            raise
        return 1


def cmd_config(args):
    """Generate or validate configuration files."""
    from .config import (PipelineConfig, get_gbm_case_study_config,
                         validate_config)

    setup_logging(verbose=True)

    if args.action == "generate":
        if args.case_study:
            if args.case_study.lower() == "gbm":
                config = get_gbm_case_study_config()
            else:
                print(f"Error: Unknown case study: {args.case_study}", file=sys.stderr)
                return 1
        else:
            config = PipelineConfig()

        output_path = Path(args.output or "config.yaml")
        config.to_yaml(output_path)
        print(f"Configuration saved to {output_path}")
        return 0

    elif args.action == "validate":
        if not args.config:
            print("Error: --config required for validation", file=sys.stderr)
            return 1

        config = PipelineConfig.from_yaml(args.config)
        warnings = validate_config(config)

        if warnings:
            print("Configuration warnings:")
            for warning in warnings:
                print(f"   - {warning}")
        else:
            print("Configuration is valid")

        return 0

    else:
        print(f"Error: Unknown action: {args.action}", file=sys.stderr)
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Mondrian Map - Pathway Visualization Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Reproduce the GBM case study
  mondrian-map reproduce --case-study gbm --out outputs/ --use-cache

  # Run pipeline with custom config
  mondrian-map pipeline --config configs/gbm_case_study.yaml --output outputs/

  # Generate visualization from entities file
  mondrian-map visualize --entities entities.csv --out map.html --show-ids

  # Generate default configuration
  mondrian-map config generate --case-study gbm --output config.yaml

For more information, see: https://github.com/aimed-lab/mondrian-map
        """,
    )

    try:
        # Python >=3.8
        from importlib.metadata import PackageNotFoundError
        from importlib.metadata import version as _pkg_version
    except Exception:  # pragma: no cover - very old Python
        try:
            from importlib_metadata import PackageNotFoundError
            from importlib_metadata import version as _pkg_version
        except Exception:
            _pkg_version = None
            PackageNotFoundError = Exception

    try:
        pkg_version = _pkg_version("mondrian-map") if _pkg_version else "1.2.1"
    except PackageNotFoundError:
        pkg_version = "1.2.1"

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {pkg_version}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Run the full Mondrian Map pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    pipeline_parser.add_argument(
        "--config",
        "-c",
        help="Path to YAML configuration file",
    )
    pipeline_parser.add_argument(
        "--output",
        "-o",
        help="Output directory",
    )
    pipeline_parser.add_argument(
        "--use-cache",
        action="store_true",
        default=None,
        help="Use cached artifacts",
    )
    pipeline_parser.add_argument(
        "--no-cache",
        action="store_false",
        dest="use_cache",
        help="Disable caching",
    )
    pipeline_parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility",
    )
    pipeline_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    pipeline_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output",
    )
    pipeline_parser.set_defaults(func=cmd_pipeline)

    reproduce_parser = subparsers.add_parser(
        "reproduce",
        help="Reproduce figures from the paper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    reproduce_parser.add_argument(
        "--case-study",
        required=True,
        choices=["gbm"],
        help="Case study to reproduce (gbm)",
    )
    reproduce_parser.add_argument(
        "--out",
        "-o",
        help="Output directory (default: outputs/<case-study>)",
    )
    reproduce_parser.add_argument(
        "--use-cache",
        action="store_true",
        default=True,
        help="Use cached artifacts (default: True)",
    )
    reproduce_parser.add_argument(
        "--no-cache",
        action="store_false",
        dest="use_cache",
        help="Disable caching (run from scratch)",
    )
    reproduce_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output",
    )
    reproduce_parser.set_defaults(func=cmd_reproduce)

    viz_parser = subparsers.add_parser(
        "visualize",
        help="Generate visualization from data files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    viz_parser.add_argument(
        "--entities",
        "-e",
        required=True,
        help="Path to entities CSV file (required columns: GS_ID, wFC, pFDR, x, y)",
    )
    viz_parser.add_argument(
        "--relations",
        "-r",
        help="Path to relations CSV file (optional)",
    )
    viz_parser.add_argument(
        "--out",
        "-o",
        required=True,
        help="Output file path (html, png, svg, or pdf)",
    )
    viz_parser.add_argument(
        "--format",
        "-f",
        choices=["html", "png", "svg", "pdf"],
        help="Output format (auto-detected from extension if not specified)",
    )
    viz_parser.add_argument(
        "--title",
        "-t",
        help="Title for the visualization",
    )
    viz_parser.add_argument(
        "--pathway-info",
        help="Path to pathway info JSON file",
    )
    viz_parser.add_argument(
        "--show-ids",
        action="store_true",
        help="Show pathway IDs on tiles",
    )
    viz_parser.add_argument(
        "--maximize",
        action="store_true",
        help="Create maximized (larger) visualization",
    )
    viz_parser.add_argument(
        "--max-relations",
        type=int,
        default=2,
        help="Maximum relations per node (0 = unlimited, default: 2)",
    )
    viz_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    viz_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output",
    )
    viz_parser.set_defaults(func=cmd_visualize)

    reproduce_case_parser = subparsers.add_parser(
        "reproduce-case-study",
        help="Reproduce the GLASS case study end-to-end",
    )
    reproduce_case_parser.add_argument("--tp", required=True, help="TP matrix path")
    reproduce_case_parser.add_argument("--r1", required=True, help="R1 matrix path")
    reproduce_case_parser.add_argument("--r2", required=True, help="R2 matrix path")
    reproduce_case_parser.add_argument("--out", required=True, help="Output directory")
    reproduce_case_parser.add_argument(
        "--config", help="Optional YAML configuration file"
    )
    reproduce_case_parser.add_argument(
        "--force", action="store_true", help="Overwrite existing outputs"
    )
    reproduce_case_parser.add_argument(
        "--debug", action="store_true", help="Enable debug output"
    )
    reproduce_case_parser.set_defaults(func=cmd_reproduce_case_study)

    config_parser = subparsers.add_parser(
        "config",
        help="Generate or validate configuration files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    config_parser.add_argument(
        "action",
        choices=["generate", "validate"],
        help="Action to perform",
    )
    config_parser.add_argument(
        "--config",
        "-c",
        help="Configuration file to validate",
    )
    config_parser.add_argument(
        "--output",
        "-o",
        help="Output path for generated configuration",
    )
    config_parser.add_argument(
        "--case-study",
        help="Case study to use as template",
    )
    config_parser.set_defaults(func=cmd_config)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
