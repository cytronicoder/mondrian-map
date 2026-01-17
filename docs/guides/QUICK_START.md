# Quick Start Guide

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/aimed-lab/mondrian-map.git
cd mondrian-map
```

### 2. Install Dependencies

```bash
pip install -r config/requirements.txt
```

### 3. Run the Application

#### Option A: New Modular Structure (Recommended)

```bash
streamlit run apps/streamlit_app.py
```

#### Option B: Backward Compatibility

```bash
streamlit run app.py
```

## Using your own data

### Required CSV Format

Your CSV file should have these columns:

- `GS_ID`: Pathway identifier
- `wFC`: Weighted fold change
- `pFDR`: Adjusted p-value  
- `x`: X coordinate for positioning
- `y`: Y coordinate for positioning
- `NAME`: Pathway name (optional, will be looked up)

### Example Data

```csv
GS_ID,wFC,pFDR,x,y,NAME
WAG002659,1.1057,3.5e-17,381.9,468.9,Glycolysis
WAG002805,1.0888,5.3e-17,971.2,573.7,TCA Cycle
```

## Visualization Specification

### Color Encoding

- Red: Up-regulated pathways (log fold-change ≥ 1.0, adjusted p-value < 0.05)
- Blue: Down-regulated pathways (log fold-change ≤ -1.0, adjusted p-value < 0.05)
- Yellow: Moderate regulation (0.5 ≤ |log fold-change| < 1.0, adjusted p-value < 0.05)
- Black: Minimal regulation (|log fold-change| < 0.5, adjusted p-value < 0.05)
- White: Non-significant pathways (adjusted p-value ≥ 0.05)

### Interactive Capabilities

- Click on tiles to display detailed pathway information
- Hover interactions for pathway identification
- Multi-dataset comparative visualization in grid layout
- Zoom and pan functionality for detailed exploration

## Development mode

### Package Installation

```bash
# Install as editable package
pip install -e .

# Import in Python
from mondrian_map.core import GridSystem, Block
from mondrian_map.visualization import create_authentic_mondrian_map
```

### Verification of Installation

```bash
python -c "from mondrian_map.core import GridSystem; print('Installation verified.')"
```

## Project structure overview

```
mondrian-map/
├── apps/streamlit_app.py      # Main web application
├── src/mondrian_map/          # Core Python package
├── data/case_study/           # Example datasets
├── config/                    # Configuration files
└── docs/                      # Documentation
```

## Common Issues and Troubleshooting

### Import Errors

If import errors are encountered, ensure the command is executed from the project root directory:

```bash
cd mondrian-map
streamlit run apps/streamlit_app.py
```

### Missing Dependencies

Install missing dependencies using:

```bash
pip install -r config/requirements.txt
```

### Data Format Issues

- Verify the CSV file contains all required columns
- Confirm that fold change values are numeric
- Ensure p-values are within the valid range [0, 1]

## Recommended Next Steps

1. Test with the example datasets provided in `data/case_study/`
2. Upload custom pathway data using the file uploader
3. Explore interactive features and visualization capabilities
4. Refer to the complete documentation in the `docs/` directory

## Tips

- Start with a small number of datasets (1–2) to validate the workflow
- Use the grid layout to compare multiple conditions side-by-side
- Enable pathway identifiers if needed to aid interpretation
- Consult the color legend to interpret visual encodings

---

You can now explore your pathway data using the Mondrian Maps interface.
