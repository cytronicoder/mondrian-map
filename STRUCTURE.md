# Mondrian Map - Project Reorganization Complete

This project has been reorganized to follow professional software development standards.

## Documentation Location

All documentation has been moved to the `docs/` directory for better organization:

### Quick Start

- See [docs/guides/QUICK_START.md](docs/guides/QUICK_START.md)

### Development

- See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)

### Deployment

- See [docs/guides/DEPLOYMENT_GUIDE.md](docs/guides/DEPLOYMENT_GUIDE.md)
- Troubleshooting: [docs/guides/DEPLOYMENT_TROUBLESHOOTING.md](docs/guides/DEPLOYMENT_TROUBLESHOOTING.md)

### Project Structure

- See [docs/architecture/ORGANIZATION_SUMMARY.md](docs/architecture/ORGANIZATION_SUMMARY.md)

### Complete Documentation

For all documentation, visit [docs/](docs/)

## Project Structure

```
mondrian-map/
├── src/
│   └── mondrian_map/          # Main package (src/ layout per PEP 517/518)
├── docs/                       # Organized documentation
│   ├── guides/                # User guides and how-tos
│   ├── architecture/          # System design and organization
│   ├── api/                   # API reference (generated)
│   └── releases/              # Release notes and history
├── config/                     # Centralized configuration
├── deployment/                 # Deployment configuration
├── apps/                       # Streamlit web application
├── tests/                      # Test suite
├── notebooks/                  # Jupyter notebooks
└── data/                       # Research data
```

## Installation

```bash
# Clone the repository
git clone https://github.com/aimed-lab/mondrian-map.git
cd mondrian-map

# Install in editable mode
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

## Verification

```python
from mondrian_map.core import GridSystem
print("Installation successful!")
```

## Key Changes

This reorganization implements:

- ✓ Professional src/ layout for Python packaging
- ✓ Organized documentation by category
- ✓ Centralized configuration management
- ✓ Scalable deployment structure
- ✓ Modern Python packaging standards (PEP 517/518)

See [REORGANIZATION_SUMMARY.md](REORGANIZATION_SUMMARY.md) for complete details.

## Links

- **Main README**: [README.md](README.md)
- **Organization Summary**: [ORGANIZATION_SUMMARY.md](ORGANIZATION_SUMMARY.md)
- **Reorganization Details**: [REORGANIZATION_SUMMARY.md](REORGANIZATION_SUMMARY.md)
- **Documentation**: [docs/README.md](docs/README.md)
