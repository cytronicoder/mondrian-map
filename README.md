# Mondrian Map Explorer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![Version](https://img.shields.io/badge/version-1.1.0-blue.svg)](docs/releases/RELEASE_NOTES.md)

**Authentic implementation of Mondrian Maps for biological pathway visualization**

This repository contains a faithful implementation of the Mondrian Map algorithm described in the bioRxiv paper: [*"Mondrian Maps: A Novel Approach for Pathway Visualization"*](https://www.biorxiv.org/content/10.1101/2024.04.11.589093v2)

![Mondrian Map Banner](figures/banner.png)

## Quick Start

### Option 1: Run Locally

```bash
# Clone the repository
git clone https://github.com/your-username/mondrian-map.git
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
mondrian-map visualize --entities my_pathways.csv --out my_map.html --show-ids
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

## 🗺️ Methods → Code Mapping

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
├── 📱 apps/                    # Streamlit applications
│   └── streamlit_app.py        # Main web application
├── src/                     # Core Python modules
│   └── mondrian_map/           # Main package
│       ├── core.py             # Core algorithm classes
│       ├── data_processing.py  # Data handling utilities
│       └── visualization.py    # Plotting functions
├── data/                    # Dataset files
│   └── case_study/             # Example datasets
├── 📓 notebooks/               # Jupyter notebooks
│   ├── visualize_mondrian_map.ipynb
│   ├── pathway_embeddings.ipynb
│   └── data_preperation.ipynb
├── ⚙️ config/                  # Configuration files
│   ├── requirements.txt        # Python dependencies
│   └── runtime.txt            # Python version
├── 🚢 deployment/             # Deployment guides
├── 📚 docs/                   # Documentation
├── figures/                # Images and plots
└── static/                 # Static assets
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
git clone https://github.com/your-username/mondrian-map.git
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
python -c "from src.mondrian_map.core import GridSystem; print('Core module imported successfully.')"
```

### Project Structure Philosophy

- **`src/`**: Core reusable modules following Python packaging standards
- **`apps/`**: User-facing applications (Streamlit, CLI tools, etc.)
- **`config/`**: All configuration and deployment files
- **`docs/`**: Documentation and guides
- **`deployment/`**: Deployment-specific documentation

## 📖 Documentation

- **[Script Usage Guide](docs/SCRIPT_USAGE.md)** - Comprehensive guide for using the run scripts
- **[Security Features](docs/SECURITY_FEATURES.md)** - Detailed security documentation
- **[Algorithm Details](docs/FIGURE_NOTE.md)** - Technical implementation details
- **[Deployment Guide](deployment/DEPLOYMENT_GUIDE.md)** - How to deploy the app
- **[Troubleshooting](deployment/DEPLOYMENT_TROUBLESHOOTING.md)** - Common issues and solutions
- **[Release Notes](docs/releases/RELEASE_NOTES.md)** - Version history and changes

## Recent updates

### Version 1.1.0 (2024-06-17)

- Enhanced interactive visualization with improved tooltip handling
- Added session state management for better user experience
- Implemented click interactions for pathway information display
- Fixed various bugs and security vulnerabilities
- Improved project structure and documentation

For a complete list of changes, see [RELEASE_NOTES.md](docs/releases/RELEASE_NOTES.md).

## Contributing

We welcome contributions! Please see our contributing guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📖 Citation

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

- Original research paper authors for the innovative Mondrian Map concept
- Streamlit team for the excellent web app framework
- Plotly team for powerful visualization capabilities
- The bioinformatics community for pathway data and annotations

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-username/mondrian-map/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/mondrian-map/discussions)
- **Email**: <your-email@example.com>

## Running the app

### Unix/macOS

To run the app with automatic port management and error handling:

```bash
./scripts/run_streamlit.sh
```

- Finds an available port
- Cleans up existing Streamlit processes
- Checks for Streamlit installation

### Windows

To run the app on Windows:

```bat
scripts\run_streamlit_win.bat
```

- Finds an available port
- Cleans up existing Streamlit processes
- Checks for Streamlit installation

### Troubleshooting

- If you see a port conflict, the script will automatically try the next available port.
- If Streamlit is not installed, you'll get a clear error message with installation instructions.
- Uploaded files are validated for name, type, and required columns for security.

## Security features

- File uploads are sanitized and validated (only .csv, safe names, required columns)
- Input validation is performed on all uploaded data
- Error handling for missing dependencies and invalid files

---

Made for the bioinformatics community.
