# Script Usage Guide

## Overview

The Mondrian Map project includes cross-platform scripts to launch the Streamlit application with automatic port management and error handling.

## Unix/macOS Script (`scripts/run_streamlit.sh`)

### Features

- Automatic port detection and conflict resolution
- Cleanup of existing Streamlit processes
- Dependency checking (Streamlit installation)
- Informative error messages

### Usage

```bash
# Make sure the script is executable (should already be set)
chmod +x scripts/run_streamlit.sh

# Run the application
./scripts/run_streamlit.sh
```

### Example Output

```
Cleaning up any existing Streamlit processes...
Port 8501 is in use, trying next port...
Port 8502 is in use, trying next port...
Starting Streamlit on port 8503

  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8503
  Network URL: http://192.168.1.100:8503
  External URL: http://your-ip:8503
```

## Windows Script (`scripts/run_streamlit_win.bat`)

### Features

- Automatic port detection and conflict resolution
- Cleanup of existing Streamlit processes
- Dependency checking (Streamlit installation)
- Informative error messages

### Usage

```bat
REM Run from the project root directory
scripts\run_streamlit_win.bat
```

### Example Output

```
Cleaning up any existing Streamlit processes...
Port 8501 is in use, trying next port...
Starting Streamlit on port 8502

  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8502
```

## Error Handling

### Streamlit Not Installed

If Streamlit is not installed, you'll see:

```
[ERROR] Streamlit is not installed. Please run: pip install -r config/requirements.txt
```

**Solution**: Install dependencies

```bash
pip install -r config/requirements.txt
```

### Port Conflicts

The scripts automatically find the next available port starting from 8501.

### Permission Issues (Unix/macOS)

If you get permission denied:

```bash
chmod +x scripts/run_streamlit.sh
```

## Advanced Usage

### Custom Port Range

To modify the starting port, edit the script:

**Unix/macOS** (`scripts/run_streamlit.sh`):

```bash
# Change this line:
local port=8501
# To your preferred starting port:
local port=9000
```

**Windows** (`scripts/run_streamlit_win.bat`):

```bat
REM Change this line:
set PORT=8501
REM To your preferred starting port:
set PORT=9000
```

### Network Access

Both scripts bind to `0.0.0.0` to allow network access. To restrict to localhost only, change:

**Unix/macOS**:

```bash
streamlit run apps/streamlit_app.py --server.port $PORT --server.address 127.0.0.1
```

**Windows**:

```bat
streamlit run apps/streamlit_app.py --server.port %PORT% --server.address 127.0.0.1
```

## Troubleshooting

### Script Doesn't Start

1. **Check permissions** (Unix/macOS): `ls -la scripts/`
2. **Check Streamlit installation**: `streamlit --version`
3. **Check Python environment**: `python --version`

### App Crashes on Startup

1. **Check dependencies**: `pip install -r config/requirements.txt`
2. **Check data files**: Ensure `data/` directory exists
3. **Check logs**: Look for error messages in the terminal

### Port Still in Use

If ports are still in use after cleanup:

```bash
# Unix/macOS - Find and kill processes
lsof -ti:8501 | xargs kill -9

# Windows - Find and kill processes
netstat -ano | findstr :8501
taskkill /PID <PID_NUMBER> /F
```

### SSL/OpenSSL Warnings

The urllib3 warning about LibreSSL is expected on macOS and doesn't affect functionality:

```
NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'
```

This is resolved by using urllib3 v1.26.18 in our requirements.

## Security Notes

### File Upload Validation

The application includes security features:

- Only `.csv` files with safe names are accepted
- Required columns are validated: `GS_ID`, `wFC`, `pFDR`, `x`, `y`
- Invalid files are rejected with user-friendly warnings

### Network Security

- Scripts bind to all interfaces (`0.0.0.0`) by default
- For production, consider restricting to localhost (`127.0.0.1`)
- Use a reverse proxy (nginx, Apache) for production deployments

## Integration with IDEs

### VS Code

Add to `.vscode/tasks.json`:

```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Run Mondrian Map",
            "type": "shell",
            "command": "./scripts/run_streamlit.sh",
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "new"
            },
            "problemMatcher": []
        }
    ]
}
```

### PyCharm

1. Go to **Run** → **Edit Configurations**
2. Add **Shell Script** configuration
3. Set script path to `scripts/run_streamlit.sh`
4. Set working directory to project root

## Deployment Considerations

### Development

Use the scripts as-is for local development.

### Production

For production deployment:

1. Use a process manager (systemd, supervisor)
2. Set up a reverse proxy
3. Use environment variables for configuration
4. Consider using Docker for containerization

## Support

If you encounter issues with the scripts:

1. Check this documentation
2. Look at the error messages
3. Verify your Python environment
4. Check GitHub Issues for similar problems
