# Quick Start Guide

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/mondrian-map.git
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

## Understanding the visualization

### Color Scheme

- Red: Up-regulated pathways (FC ≥ 1.0, p < 0.05)
- Blue: Down-regulated pathways (FC ≤ -1.0, p < 0.05)
- Yellow: Moderate change (0.5 ≤ |FC| < 1.0, p < 0.05)
- Black: Neutral (|FC| < 0.5, p < 0.05)
- White: Non-significant (p ≥ 0.05)

### Interactive Features

- **Click tiles** to see pathway details
- **Hover** for quick pathway information
- **Multi-dataset comparison** in grid layout
- **Zoom and pan** for detailed exploration

## Development mode

### Package Installation

```bash
# Install as editable package
pip install -e .

# Import in Python
from mondrian_map.core import GridSystem, Block
from mondrian_map.visualization import create_authentic_mondrian_map
```

### Testing the Installation

```bash
python -c "from src.mondrian_map.core import GridSystem; print('Installation successful.')"
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

## 🚨 Common Issues

### Import Errors

If you get import errors, make sure you're running from the project root:

```bash
cd mondrian-map  # Make sure you're in the project directory
streamlit run apps/streamlit_app.py
```

### Missing Dependencies

```bash
pip install -r config/requirements.txt
```

### Data Format Issues

- Ensure your CSV has the required columns
- Check that fold change values are numeric
- Verify p-values are between 0 and 1

## Next steps

1. **Try the example datasets** included in `data/case_study/`
2. **Upload your own pathway data** using the file uploader
3. **Explore the interactive features** - click tiles, adjust layouts
4. **Read the full documentation** in the `docs/` folder

## Tips

- Start with a small number of datasets (1–2) to validate the workflow
- Use the grid layout to compare multiple conditions side-by-side
- Enable pathway identifiers if needed to aid interpretation
- Consult the color legend to interpret visual encodings

---

You can now explore your pathway data using the Mondrian Maps interface.
