# Deployment Troubleshooting Guide

## Dependency Installation Error

### Problem Description

Streamlit Cloud reports "Error installing requirements" with indication of dependency conflicts.

### Root Cause Analysis

The original `requirements.txt` contained over 130 packages with version conflicts and unnecessary dependencies.

### Resolution Strategy Applied

1. **Requirements Optimization**: Reduced from 130+ packages to 4 essential dependencies
2. **Version Compatibility**: Applied broader version ranges instead of exact version pinning
3. **Dependency Resolution**: Eliminated packages causing dependency conflicts

### Simplified Requirements

```
streamlit>=1.28.0
pandas>=2.0.0  
numpy>=1.24.0
plotly>=5.15.0
```

## Deployment Procedures

### Option 1: Streamlit Community Cloud (Recommended)

1. Navigate to [share.streamlit.io](https://share.streamlit.io)
2. Authenticate using GitHub credentials
3. Select "New app"
4. Specify Repository: `aimed-lab/mondrian-map`
5. Specify Branch: `main`
6. Specify Main File Path: `app.py`
7. Configure Advanced Settings (if necessary):
   - Python version: 3.9
   - Requirements file: `requirements.txt` (default)
8. Initiate deployment

### Option 2: Alternative Resolution Steps

If deployment issues persist, perform the following:

1. **In Streamlit Cloud Dashboard**:
   - Navigate to "Manage app"
   - Select "Reboot app"
   - Alternatively, select "Delete app" and redeploy

2. **Examine Deployment Logs**:
   - Select "Manage app" → View logs
   - Identify specific error messages

## Issue Resolution Guide

### Package Not Found Error

**Resolution**: Verify all packages in requirements.txt are available on PyPI

### Memory Limit Exceeded

**Resolution**: Use the simplified requirements configuration

### Build Timeout

**Resolution**: The simplified requirements should provide faster build times

### Import Errors

**Fix**: All imports in app.py should work with the 4 core packages

## Files Created for Deployment

- `requirements.txt` - Minimal, clean dependencies
- `.streamlit/config.toml` - Streamlit configuration
- `requirements_original_backup.txt` - Backup of original
- `requirements_streamlit_cloud.txt` - Alternative version

## Verification Steps

After deployment:

1. **Check app loads**: Should show the Mondrian Map interface
2. **Test dataset selection**: Try different datasets
3. **Test canvas grid**: Should show multiple maps
4. **Test interactivity**: Click on tiles, toggle options

## Alternative Deployment (If Streamlit Cloud Fails)

### Railway (Backup Option)

1. Go to [railway.app](https://railway.app)
2. Connect GitHub repository
3. Deploy automatically
4. Uses the same clean requirements.txt

### Render (Backup Option)

1. Go to [render.com](https://render.com)  
2. Create new Web Service
3. Connect GitHub repository
4. Build command: `pip install -r requirements.txt`
5. Start command: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`

## Success indicators

- Build completes without errors
- App starts and shows the interface
- Data loads correctly
- Visualizations render properly
- Interactive features operate as expected

## Need Help?

If you're still having issues:

1. **Check the specific error** in deployment logs
2. **Try the alternative platforms** (Railway, Render)
3. **Use the backup requirements** files if needed
4. **Verify all data files** are in the repository

The simplified approach should resolve the dependency conflicts you were experiencing!

## New Script-Based Launch (v1.1.1)

### Unix/macOS

- Use `./scripts/run_streamlit.sh` to launch the app
- Script finds an available port, cleans up old processes, and checks for Streamlit

### Windows

- Use `scripts\run_streamlit_win.bat` to launch the app
- Script finds an available port, cleans up old processes, and checks for Streamlit

### Common Issues

- **Streamlit not installed**: Script will print an error and exit. Run `pip install -r config/requirements.txt`.
- **Port in use**: Script will try the next available port automatically.
- **File upload rejected**: Only .csv files with safe names and required columns are accepted. Check your file format.

## Security Improvements

- File uploads are sanitized and validated for type and columns
