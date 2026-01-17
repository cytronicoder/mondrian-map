# Mondrian Abstraction and Language Model Embeddings for Differential Pathway Analysis

**Project Overview:**

This repository contains supplementary file, codebase and data generated for our paper titled "Mondrian Abstraction and Language Model Embeddings for Differential Pathway Analysis" which is currently under peer-review in a bioinformatics conference.

**Supplementary File:** The supplementary file to our paper can be found [here](https://github.com/aimed-lab/mondrian-map/blob/main/supplementary-file.pdf).

**Code:** The `notebooks` folder contains the following jupyter notebooks:

1. **clinical_data_analysis.ipynb**: Notebook for analyzing clinical data and suitable patient profile selection.
2. **data_preperation.ipynb**: Notebook for preprocessing data to make it suitable for Mondrian Map Visualization.
3. **pathway_embeddings.ipynb**: Here, we've experimented with different embedding techniques with different prompting strategies.
4. **visualize_mondrian_map.ipynb**: In this notebook, we've generated the Mondrian Maps in our Gliblastoma case study.

**Data:** All the datasets used and processed are stored in the `data` folder.

## Flow Chart

![Flow-Diagram](figures/banner.png)

## Mondrian Map Generation

![Methodology](figures/method.png)

## Results

![Results](figures/results.png)

## Web Application Canvas Grid

![Canvas Grid](figures/canvas_grid_example.png)

## Cite Us

If you find out tool useful, cite our [latest preprint](https://www.biorxiv.org/content/10.1101/2024.04.11.589093v2).

```
@article {AlAbir_MondrianMap,
 author = {Al Abir, Fuad and Chen, Jake Y.},
 title = {Mondrian Abstraction and Language Model Embeddings for Differential Pathway Analysis},
 elocation-id = {2024.04.11.589093},
 year = {2024},
 doi = {10.1101/2024.04.11.589093},
 publisher = {Cold Spring Harbor Laboratory},
 URL = {https://www.biorxiv.org/content/early/2024/08/19/2024.04.11.589093},
 eprint = {https://www.biorxiv.org/content/early/2024/08/19/2024.04.11.589093.full.pdf},
 journal = {bioRxiv}
}
```

## Release Notes

### Version 1.12 (June 14, 2025)

**Security Updates & Bug Fixes:**

- Critical security updates: Updated urllib3, requests, certifi, Jinja2, and pillow to secure versions
- Bug fixes: Resolved UnboundLocalError in the canvas grid function and pandas KeyError in the detailed popup
- Minimal requirements: Added `requirements_minimal.txt` for secure production deployments
- Documentation: Added comprehensive security update documentation
- Performance: Updated pandas and plotly to improve performance

**Technical Improvements:**

- Fixed function parameter bug causing canvas grid rendering issues
- Resolved pandas DataFrame column access errors in detailed statistics
- Enhanced dependency management with minimal requirements file
- Added comprehensive security update tracking and documentation

### Version 1.1 (June 14, 2025)

**Major Features Added:**

- Complete authentic Mondrian algorithm: Faithful implementation of the algorithm from the bioRxiv paper
- Interactive web application: Full-featured Streamlit app with Mondrian map generation
- Canvas grid system: Multi-dataset comparison with customizable grid layouts
- Enhanced user interface: Detailed popup views, statistics, and color legends

**Technical Improvements:**

- Implemented exact `GridSystem`, `Block`, `Line`, `Corner` classes from research notebooks
- Added 3-stage generation process: block placement → Manhattan relationship lines → line extensions
- Integrated authentic color scheme with proper thresholds and area scaling
- Added light gray grid lines for authentic Mondrian appearance matching the paper
- Implemented pathway network relationship visualization
- Added file upload capability for custom datasets
- Enhanced hover tooltips and interactive features

**Data Integration:**

- Support for all 6 pre-computed datasets from the case study
- Complete pathway annotation system with descriptions and ontology
- Network relationship data integration for Manhattan connection lines
- Comprehensive statistics and analysis features

**User Experience:**

- Multi-select dataset configuration
- Customizable canvas layouts (1×1 to 4×4 grids)
- Toggle options for different viewing modes
- Adaptive text scaling and positioning
- Professional color legend and comprehensive pathway details

### Version 1.0 (Initial Publication)

**Core Research Implementation:**

- Original Mondrian Map methodology and algorithms
- Jupyter notebooks for data preparation and visualization
- Clinical data analysis and patient profile selection
- Pathway embeddings with language model techniques
- Glioblastoma case study implementation

## Contact

Reach us at [jakechen@uab.edu](mailto:jakechen@uab.edu) or [fuad021@uab.edu](mailto:fuad021@uab.edu).

## License

Mondrian Map codebase is under MIT license.

## Web Application - Authentic Mondrian Map Explorer

An interactive Streamlit web application is provided that implements the complete authentic Mondrian Map algorithm from our bioRxiv paper. The application provides a faithful reproduction of the research methodology with enhanced interactive features.

### Features

**Authentic algorithm implementation**

- Complete `GridSystem`, `Block`, `Line`, `Corner` classes with exact parameters from the paper
- 3-stage generation process: blocks → Manhattan relationship lines → line extensions
- Authentic color scheme: Red (up-regulated), Blue (down-regulated), Yellow (moderate), Black (neutral), White (non-significant)
- Area scaling using `abs(log2(wFC)) * 4000` formula
- Light gray grid lines for authentic Mondrian appearance

**Interactive visualization**

- Canvas grid layout for comparing multiple datasets simultaneously
- Detailed popup views with comprehensive pathway analysis
- Adaptive text scaling and positioning for pathway labels
- Clickable interface with hover tooltips and detailed statistics

**Data integration**

- Support for all 6 pre-computed datasets (Aggressive R1/R2, Baseline R1/R2, Nonaggressive R1/R2)
- Pathway network relationship visualization (Manhattan connection lines)
- File upload capability for custom CSV datasets
- Complete pathway annotation system with descriptions and ontology information

**User interface**

- Multi-select dataset configuration
- Customizable canvas grid layouts (1×1 to 4×4)
- Toggle options for full-size maps, color legends, and maximized views
- Professional color legend and comprehensive statistics

### Installation & Usage

1. Install dependencies:

```bash
pip install -r requirements.txt
```

1. Run the application:

```bash
streamlit run app.py
```

1. Open your browser to `http://localhost:8501`

### Supported Data Format

The application accepts CSV files with the following required columns:

- `GS_ID`: Gene set/pathway identifier
- `wFC`: Weighted fold change
- `pFDR`: Adjusted p-value (FDR)
- `x`, `y`: Coordinates for pathway positioning

### Algorithm Parameters

The implementation uses the exact parameters from the research paper:

- Canvas: 1001×1001 pixels
- Block size: 20×20 pixels
- Line width: 5 pixels (borders), 1 pixel (grid lines)
- Area scalar: 4000
- Up-regulation threshold: ≥1.25
- Down-regulation threshold: ≤0.75
