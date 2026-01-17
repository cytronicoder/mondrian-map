# Release Notes - Mondrian Map Explorer

## Version 1.11.0 (December 14, 2025)

### Enhanced grid lines & authentic Mondrian aesthetics

This release significantly improves the visual authenticity and user experience of the Mondrian Map Explorer with smart grid line management, enhanced interactivity, and refined UI elements.

---

## New Capabilities

### Intelligent grid line system

- **Authentic Mondrian Principles**: Implemented intelligent grid line management following true Mondrian aesthetics
- **Intersection Avoidance**: Grid lines automatically avoid intersecting pathway tile interiors
- **Structural Purpose Only**: Lines only exist where they serve meaningful structural purposes
- **Smart Start/End Points**: Lines can start/end at canvas boundaries OR closest tile edges
- **Light Gray Styling**: All grid lines use light gray color (`#D3D3D3`) for subtle appearance
- **Minimum Segment Length**: Only substantial segments (>40px) are kept to avoid visual clutter

### Pathway Identifier Management

- **Toggle Control**: Added sidebar checkbox to show/hide pathway ID labels
- **Default Off**: Pathway IDs are now hidden by default for cleaner small-view appearance
- **Regular Font**: Changed from bold "Arial Black" to regular "Arial" font for better readability
- **Universal Control**: Toggle works across all visualization modes (canvas, full-size, detailed views)

### Enhanced Click Functionality

- **Direct Tile Interaction**: Click directly on pathway tiles (no separate buttons needed)
- **Full-Screen Activation**: Clicking any tile automatically opens full-screen detailed view of the entire dataset
- **Simplified Hover**: Hover shows pathway name + "Click for full-screen view" hint
- **Session State Management**: Proper state tracking for detailed view navigation
- **Close Button**: Intuitive button to close detailed view and return to overview

### Improved connecting lines

- **Enhanced Visibility**: Colored Manhattan relationship lines now use 2pt thickness (vs 1pt grid lines)
- **Better Contrast**: Relationship lines stand out more prominently from background grid
- **Maintained Colors**: Red, blue, and yellow connecting lines preserve their biological meaning

## Technical Implementation Details

### Grid Line Generation Algorithm

- **`create_smart_grid_lines()`**: New intelligent grid line generation system
- **`get_meaningful_tile_edges()`**: Identifies structurally important tile boundaries
- **`find_structural_*_segments()`**: Creates line segments only where needed
- **`has_structural_purpose_*()`**: Validates that each line segment serves a real purpose
- **Intersection Detection**: Advanced logic to detect and avoid tile interior intersections

### User Interface Enhancements

- **Streamlined Navigation**: Removed example buttons in favor of direct interaction
- **Consistent Styling**: Unified toggle control across all map views
- **Responsive Design**: Better handling of different screen sizes and viewing modes
- **Clean Aesthetics**: Reduced visual clutter while maintaining functionality

### Bug Fixes

- **KeyError Resolution**: Fixed pandas `nlargest()` function error in detailed popup
- **State Management**: Improved session state handling for detailed views
- **Font Consistency**: Standardized font usage across all text elements
- **Click Detection**: Enhanced click event handling with proper rerun triggers

---

## Visual improvements

### Authentic Mondrian Appearance

- **Minimal Grid Lines**: Only essential structural lines remain visible
- **No Tile Borders**: Seamless tile appearance with matching border colors
- **Clean Composition**: Follows authentic Mondrian principles of minimal, purposeful elements
- **Professional Look**: Reduced visual noise while maintaining scientific accuracy

### Enhanced Interactivity

- **Intuitive Clicks**: Direct interaction with visualization elements
- **Clear Feedback**: Immediate visual feedback for user actions
- **Smooth Transitions**: Proper state management for view changes
- **Contextual Hints**: Helpful hover messages guide user interaction

---

## Migration from Version 1.1.0

### What's Changed

- **Grid Line Behavior**: Much cleaner, more authentic Mondrian appearance
- **Pathway ID Display**: Now hidden by default, can be toggled on/off
- **Click Interaction**: Direct tile clicking replaces separate example buttons
- **Line Thickness**: Relationship lines are now more prominent

### Backward Compatibility

- **Data Format**: Fully compatible with existing datasets
- **Core Functionality**: All analysis features preserved and enhanced
- **Configuration**: Previous settings work with new toggle controls

## Bug Fixes and Improvements

### Critical Fixes

- **Pandas KeyError**: Resolved `df.nlargest(5, df['wFC'].abs())` error by creating temporary column
- **Session State**: Fixed state management issues in detailed view navigation
- **Click Detection**: Improved click event handling and response

### Visual Fixes

- **Grid Line Positioning**: Proper alignment with tile boundaries
- **Font Rendering**: Consistent font styling across all text elements
- **Hover Tooltips**: Simplified and more informative hover messages

---

## Usage Guidelines and Best Practices

### New Interface Controls

- **Pathway Identifier Toggle**: Use sidebar checkbox "Show pathway IDs" for visibility control
- **Direct Interaction**: Click pathway tiles to open full-screen detailed analysis
- **Navigation Control**: Use close button to return to overview display

### Recommended Practices

- **Grid Canvas Views**: Hide pathway identifiers for cleaner appearance in multi-dataset grids
- **Detailed Analysis**: Display pathway identifiers when performing focused analysis
- **Efficient Navigation**: Utilize direct tile clicking for rapid access to detailed information

---

## Version 1.1.0 (December 14, 2025)

### Major Release: Authentic Mondrian Map Web Application

This release introduces a complete interactive web application that faithfully implements the authentic Mondrian Map algorithm from our bioRxiv paper "Mondrian Abstraction and Language Model Embeddings for Differential Pathway Analysis".

---

## New Capabilities

### Complete algorithm implementation

- **Complete Class System**: Implemented exact `GridSystem`, `Block`, `Line`, `Corner` classes with parameters matching the research notebooks
- **3-Stage Generation Process**:
  1. Block placement based on pathway coordinates and fold change
  2. Manhattan relationship lines connecting related pathways
  3. Line extensions for complete Mondrian aesthetic
- **Authentic Color Scheme**:
  - Red: Up-regulated pathways (FC ≥ 1.25)
  - Blue: Down-regulated pathways (FC ≤ 0.75)
  - Yellow: Moderate change pathways
  - Black: Neutral pathways
  - White: Non-significant pathways (p > 0.05)
- **Area Scaling**: Exact `abs(log2(wFC)) * 4000` formula from the paper
- **Light Gray Grid Lines**: Authentic Mondrian appearance with outer boundary and internal grid lines

### Interactive web application

- **Streamlit-based Interface**: Professional, responsive web application
- **Canvas Grid System**: Compare multiple datasets simultaneously in customizable grid layouts (1×1 to 4×4)
- **Detailed Popup Views**: Click any dataset button for comprehensive analysis with:
  - Maximized Mondrian map (1000×1000 pixels)
  - Dataset statistics (total pathways, up/down-regulated counts)
  - Color distribution analysis
  - Top pathways by fold change
- **Adaptive Text Scaling**: Pathway labels scale with tile size (8-24px range)
- **Smart Label Positioning**: Labels positioned above tiles with fallback to inside positioning

### Data integration and analysis

- **Multi-Dataset Support**: All 6 pre-computed datasets from the glioblastoma case study
  - Aggressive R1/R2
  - Baseline R1/R2  
  - Nonaggressive R1/R2
- **Pathway Network Integration**: Manhattan connection lines based on pathway relationships
- **File Upload Capability**: Support for custom CSV datasets with validation
- **Complete Annotation System**: Integration with pathway descriptions, ontology, and disease information
- **Comprehensive Statistics**: Real-time analysis of pathway distributions and significance

### User Interface and Experience Enhancements

- **Multi-Select Configuration**: Choose any combination of datasets for comparison
- **Viewing Modes**:
  - Canvas Grid Overview: Side-by-side dataset comparison
  - Full-Size Individual Maps: Detailed single-dataset views
  - Maximized Mode: Enhanced detail for analysis
- **Interactive Controls**:
  - Toggle color legend display
  - Toggle full-size map views
  - Maximize individual maps option
  - Customizable canvas grid dimensions
- **Professional Tooltips**: Rich hover information with pathway details
- **Color Legend**: Visual guide to the Mondrian color scheme

---

## Technical Implementation Improvements

### Algorithm Fidelity

- **Exact Parameter Matching**: All constants match the research implementation
  - Canvas: 1001×1001 pixels
  - Block size: 20×20 pixels
  - Line width: 5px (borders), 1px (grid lines), minimum 2px for visibility
  - Area scalar: 4000
- **Proper Line Rendering**: Ensured grid lines are visible with appropriate width
- **Border Handling**: Zero-width white borders for clean tile appearance
- **Color Enum Integration**: Proper conversion between Python enums and Plotly requirements

### Performance & Reliability

- **Efficient Data Loading**: Cached pathway information loading
- **Error Handling**: Robust file upload validation and error messages
- **Memory Management**: Proper canvas clearing between generations
- **Responsive Design**: Optimized for different screen sizes and viewing modes

### Code Architecture

- **Modular Design**: Separated concerns for algorithm, visualization, and UI
- **Helper Functions**: Complete set of utility functions from research notebooks
- **Type Safety**: Proper type hints and enum usage
- **Documentation**: Comprehensive inline documentation and help text

---

## Data Processing and Integration

### Pathway Data

- **Complete Dataset Coverage**: All pathways from the glioblastoma case study
- **Rich Annotations**: NAME, Description, Ontology, Disease information
- **Network Relationships**: Pathway-pathway connection data for Manhattan lines
- **Coordinate System**: Exact x,y positioning from the research

### File Format Support

- **CSV Upload**: Support for custom datasets with required columns:
  - `GS_ID`: Gene set/pathway identifier
  - `wFC`: Weighted fold change
  - `pFDR`: Adjusted p-value (FDR)
  - `x`, `y`: Coordinates for positioning
- **Validation**: Automatic checking for required columns and data types
- **Error Reporting**: Clear messages for data format issues

---

## User experience enhancements

### Navigation & Interaction

- **Intuitive Interface**: Clear section organization and navigation
- **Contextual Help**: Tooltips and help text throughout the application
- **Responsive Feedback**: Real-time updates and loading indicators
- **Professional Styling**: Modern UI design with consistent theming

### Analysis Features

- **Statistical Overview**: Comprehensive dataset statistics
- **Comparative Analysis**: Side-by-side dataset comparison
- **Detailed Exploration**: Deep-dive analysis with maximized views
- **Export-Ready Visualizations**: High-quality plots suitable for presentations

---

## Migration from Version 1.0

### What's Changed

- **Comprehensive Application Rewrite**: Transformation from basic scatter plot visualization to authentic Mondrian map implementation
- **Enhanced Data Integration**: From basic CSV loading to comprehensive pathway analysis
- **Professional UI**: From basic Streamlit interface to feature-rich application

### Backward Compatibility

- **Data Format**: Maintains compatibility with existing CSV data files
- **Core Functionality**: All original research capabilities preserved and enhanced
- **Notebook Integration**: Research notebooks remain functional alongside the web app

---

## 🐛 Bug Fixes & Improvements

### Visual Fixes

- **Grid Line Visibility**: Ensured all grid lines are properly visible
- **Text Positioning**: Fixed label positioning issues with proper centering and fallback
- **Color Consistency**: Resolved color enum conversion issues
- **Border Rendering**: Achieved clean borderless tile appearance

### Performance Improvements

- **Faster Rendering**: Optimized Plotly trace generation
- **Memory Usage**: Improved canvas clearing and object management
- **Loading Times**: Cached data loading for better responsiveness

---

## 📋 Requirements

### System Requirements

- Python 3.7+
- Modern web browser (Chrome, Firefox, Safari, Edge)
- 4GB+ RAM recommended for large datasets

### Dependencies

- streamlit
- plotly
- pandas
- numpy
- pathlib

### Installation

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🔮 Future Roadmap

### Planned Features

- **Export Functionality**: Save high-resolution images and data
- **Advanced Filtering**: Filter pathways by various criteria
- **Batch Processing**: Process multiple datasets simultaneously
- **API Integration**: REST API for programmatic access
- **Enhanced Analytics**: Statistical testing and comparison tools

### Community Contributions

We welcome contributions! Please see our contribution guidelines and open issues on GitHub.

---

## Support and Contact Information

For questions, bug reports, or feature requests:

- **Email**: [jakechen@uab.edu](mailto:jakechen@uab.edu) or [fuad021@uab.edu](mailto:fuad021@uab.edu)
- **GitHub Issues**: [https://github.com/aimed-lab/mondrian-map/issues](https://github.com/aimed-lab/mondrian-map/issues)
- **Documentation**: See README.md for detailed usage instructions

---

## 📄 Citation

If you use this software in your research, please cite our paper:

```bibtex
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

---

*Thank you for using Mondrian Map Explorer! We hope this tool enhances your pathway analysis research.*
