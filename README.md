# Mondrian Map Explorer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![Version](https://img.shields.io/badge/version-1.2.1-blue.svg)](docs/releases/RELEASE_NOTES.md)

**Authentic implementation of Mondrian Maps for biological pathway visualization**

This repository contains a faithful implementation of the Mondrian Map algorithm described in the bioRxiv paper: [*"Mondrian Maps: A Novel Approach for Pathway Visualization"*](https://www.biorxiv.org/content/10.1101/2024.04.11.589093v2)

![Mondrian Map Banner](figures/banner.png)

## Quick Start

### Option 1: Run Locally

```bash
# Clone the repository
git clone https://github.com/aimed-lab/mondrian-map.git
cd mondrian-map

# Install dependencies
pip install -r config/requirements.txt

# Run the Streamlit app
streamlit run apps/streamlit_app.py
```

### Option 2: Try Online

**Live Demo:** [https://your-deployment-url.streamlit.app](https://your-deployment-url.streamlit.app) — Try the app without installation

### Option 3: Command Line Interface

```bash
# Install the package
pip install -e .

# Reproduce the GBM case study from the paper
mondrian-map reproduce --case-study gbm --out outputs/ --use-cache

# Generate visualization from your own data
# You can optionally supply a relations CSV and control how many relations per node are shown
mondrian-map visualize --entities my_pathways.csv --relations my_relations.csv --out my_map.html --show-ids --max-relations 2
```

## Reproduce paper figures

To reproduce Figures 1-2 from the paper, run:

```bash
# Using the CLI
mondrian-map reproduce --case-study gbm --out outputs/ --use-cache

# Or using the script
./scripts/reproduce_figures.sh
```

This will:

1. Load precomputed PAGER/GNPA results from `data/case_study/`
2. Load pathway embeddings (t-SNE coordinates)
3. Generate the Mondrian Map visualization
4. Save outputs to `outputs/gbm/`

### Reproduce from Scratch (No Cache)

To run the full pipeline without using cached artifacts:

```bash
mondrian-map reproduce --case-study gbm --out outputs/ --no-cache
```

Note: Running without cache requires PAGER API access and may take 10–30 minutes.

## Methods to Code Mapping

| Paper Section | Function/Module | File |
|---------------|-----------------|------|
| **DEG Selection** | `select_degs()`, `compute_fold_change()` | `src/mondrian_map/degs.py` |
| **GNPA Enrichment** | `PagerClient.run_gnpa()` | `src/mondrian_map/pager_client.py` |
| **wFC Computation** | `compute_wfc()`, `compute_pathway_wfc()` | `src/mondrian_map/pathway_stats.py` |
| **Embedding Generation** | `EmbeddingGenerator.encode()` | `src/mondrian_map/embeddings.py` |
| **t-SNE Projection** | `tsne_project()` | `src/mondrian_map/projection.py` |
| **Mondrian Grid** | `GridSystem`, `Block`, `Line` | `src/mondrian_map/core.py` |
| **Visualization** | `create_authentic_mondrian_map()` | `src/mondrian_map/visualization.py` |

### Key Parameters

| Parameter | Value | Location |
|-----------|-------|----------|
| Up-regulation threshold | FC ≥ 1.5 | `configs/gbm_case_study.yaml` |
| Down-regulation threshold | FC ≤ 0.5 | `configs/gbm_case_study.yaml` |
| Significance threshold | pFDR < 0.05 | `configs/gbm_case_study.yaml` |
| t-SNE perplexity | 30 | `configs/gbm_case_study.yaml` |
| t-SNE random seed | 42 | `configs/gbm_case_study.yaml` |
| Canvas size | 1001 × 1001 | `src/mondrian_map/core.py` |
| Block grid | 20 × 20 | `src/mondrian_map/core.py` |

## Repository structure

```
mondrian-map/
├── apps/                       # Streamlit applications
│   └── streamlit_app.py        # Main web application
├── mondrian_map/               # Core Python package
│   ├── core.py                 # Core algorithm classes
│   ├── data_processing.py      # Data handling utilities
│   └── visualization.py        # Plotting functions
├── data/                       # Dataset files
│   └── case_study/             # Example datasets
├── notebooks/                  # Jupyter notebooks
│   ├── visualize_mondrian_map.ipynb
│   ├── pathway_embeddings.ipynb
│   └── data_preparation.ipynb
├── config/                     # Configuration files
│   ├── requirements.txt        # Python dependencies
│   └── runtime.txt             # Python version specification
├── deployment/                 # Deployment documentation
├── docs/                       # User documentation
├── figures/                    # Images and visualizations
└── static/                     # Static assets
```

## Features

### Authentic algorithm implementation

- **3-Stage Generation Process**: Grid System → Block Placement → Line Generation
- **Exact Classes**: `GridSystem`, `Block`, `Line`, `Corner` from original research
- **Authentic Parameters**: 1001×1001 canvas, 20×20 block grid, proper adjustments

### Visual features

- **5-Color Mondrian Scheme**: White, Black, Yellow, Red, Blue
- **Smart Grid Lines**: Structural lines that avoid intersecting tiles
- **Interactive Canvas**: Click tiles for detailed pathway information
- **Multi-Dataset Support**: Compare multiple conditions side-by-side
- **Enhanced Tooltips**: Improved hover and click interactions
- **Session State Management**: Persistent user interactions

### Data processing

- **Flexible Input**: CSV files with pathway data
- **Rich Annotations**: Pathway descriptions, ontologies, disease associations
- **Network Analysis**: Pathway crosstalk visualization
- **Statistical Summaries**: Regulation statistics and significance testing
- **Input Validation**: Secure data processing

## Algorithm details

The implementation follows the exact 3-stage process from the research paper:

### Stage 1: Grid System Initialization

```python
grid_system = GridSystem(1001, 1001, 20, 20)  # Canvas: 1001×1001, Blocks: 20×20
```

### Stage 2: Block Placement

- **Area Calculation**: `abs(log2(wFC)) * 4000`
- **Color Mapping**: Based on fold-change and p-value thresholds
- **Position Optimization**: Centered around pathway coordinates

### Stage 3: Line Generation

- **Smart Grid Lines**: Avoid tile intersections, maintain structure
- **Manhattan Connections**: Pathway relationship visualization
- **Authentic Styling**: Proper line widths and adjustments

## Data format

Your CSV files should contain these columns:

| Column | Description | Example |
|--------|-------------|---------|
| `GS_ID` | Pathway identifier | `WAG002659` |
| `wFC` | Weighted fold change | `1.1057` |
| `pFDR` | Adjusted p-value | `3.5e-17` |
| `x` | X-coordinate | `381.9` |
| `y` | Y-coordinate | `468.9` |
| `NAME` | Pathway name | `Glycolysis` |

## Color scheme

| Color | Meaning | Criteria |
|-------|---------|----------|
| Red | Up-regulated | FC ≥ 1.0, p < 0.05 |
| Blue | Down-regulated | FC ≤ -1.0, p < 0.05 |
| Yellow | Moderate change | 0.5 ≤ \|FC\| < 1.0, p < 0.05 |
| Black | Neutral | \|FC\| < 0.5, p < 0.05 |
| White | Non-significant | p ≥ 0.05 |

## Development

### Setting up Development Environment

```bash
# Clone and enter directory
git clone https://github.com/aimed-lab/mondrian-map.git
cd mondrian-map

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r config/requirements.txt

# Install package in development mode
pip install -e .
```

### Running Tests

```bash
# Run the Streamlit app
streamlit run apps/streamlit_app.py

# Test with example data
python -c "from mondrian_map.core import GridSystem; print('Core module imported successfully.')"
```

### Project Structure Philosophy

- **`src/`**: Core reusable modules following Python packaging standards
- **`apps/`**: User-facing applications (Streamlit, CLI tools, etc.)
- **`config/`**: All configuration and deployment files
- **`docs/`**: Documentation and guides
- **`deployment/`**: Deployment-specific documentation

## Documentation

- **[Script Usage Guide](docs/SCRIPT_USAGE.md)** - Comprehensive documentation for execution scripts
- **[Security Features](docs/SECURITY_FEATURES.md)** - Detailed security implementation
- **[Algorithm Details](docs/FIGURE_NOTE.md)** - Technical implementation specifications
- **[Deployment Guide](deployment/DEPLOYMENT_GUIDE.md)** - Application deployment procedures
- **[Troubleshooting](deployment/DEPLOYMENT_TROUBLESHOOTING.md)** - Issue resolution and common problems
- **[Release Notes](docs/releases/RELEASE_NOTES.md)** - Version history and change log

## Recent updates

### Version 1.2.1 (2026-01-17)

- Final cleanup of remaining modules; removed backward compatibility shims; updated release notes and packaging metadata (2026-01-17)

### Version 1.1.0 (2024-06-17)

- Enhanced interactive visualization with improved tooltip handling
- Added session state management for better user experience
- Implemented click interactions for pathway information display
- Fixed various bugs and security vulnerabilities
- Improved project structure and documentation

For a complete list of changes, see [RELEASE_NOTES.md](docs/releases/RELEASE_NOTES.md).

## Contributing

We welcome contributions to this project. Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Adhere to established code style conventions
4. Add appropriate test coverage
5. Update documentation accordingly
6. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this tool in your research, please cite:

```bibtex
@article{mondrian_maps_2024,
  title={Mondrian Maps: A Novel Approach for Pathway Visualization},
  author={[Authors]},
  journal={bioRxiv},
  year={2024},
  doi={10.1101/2024.04.11.589093v2}
}
```

## Acknowledgments

- Original research paper authors for the methodological innovation in pathway visualization
- Streamlit framework contributors for the interactive application platform
- Plotly framework contributors for visualization capabilities
- Bioinformatics community for pathway data resources and annotations

## Support

- **GitHub Issues**: [Issue tracker](https://github.com/aimed-lab/mondrian-map/issues)
- **GitHub Discussions**: [Discussion forum](https://github.com/aimed-lab/mondrian-map/discussions)
- **Contact**: [jakechen@uab.edu](mailto:jakechen@uab.edu) or [fuad021@uab.edu](mailto:fuad021@uab.edu)

## Running the Application

### Unix and macOS

Execute the application with automatic port management and error handling:

```bash
./scripts/run_streamlit.sh
```

Features:

- Automatic discovery of available port
- Cleanup of existing Streamlit processes
- Verification of Streamlit installation

### Windows

Execute the application on Windows:

```bat
scripts\run_streamlit_win.bat
```

Features:

- Automatic discovery of available port
- Cleanup of existing Streamlit processes
- Verification of Streamlit installation

### Troubleshooting

- Port conflicts are automatically resolved by the script attempting subsequent available ports
- Clear error messages indicate missing dependencies with installation instructions
- Uploaded files undergo validation for filename safety, file type, and required column structure

## Security Implementation

- File uploads are sanitized and validated (CSV format only, safe filenames, required column verification)
- All uploaded data undergoes validation
- Comprehensive error handling for missing dependencies and invalid data formats
