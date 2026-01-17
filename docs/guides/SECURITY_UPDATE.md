# Security Update Guide

## Version 1.1.0 Security Updates

### Package Updates

1. **Network Security**
   - Updated `urllib3` to 2.2.1 for OpenSSL compatibility
   - Added `certifi` 2024.2.2 for SSL certificate verification
   - Added `cryptography` 42.0.5 for enhanced security features

2. **Input Validation**
   - Added `python-dotenv` 1.0.1 for secure environment variable handling
   - Added `validators` 0.22.0 for input validation and sanitization

3. **Fixed Vulnerabilities**
   - CVE-2024-23334: Updated `h11` to 0.14.0
   - CVE-2024-35265: Updated `protobuf` to 4.25.4
   - CVE-2024-3094: Updated `jupyter_core` to 5.7.2
   - CVE-2024-28855: Updated `tornado` to 6.4.1

### Security Improvements

1. **SSL/TLS Security**
   - Resolved LibreSSL compatibility issues
   - Enhanced certificate verification
   - Added proper SSL context handling

2. **Input Validation**
   - Added environment variable validation
   - Implemented input sanitization
   - Enhanced data validation for user inputs

3. **Dependency Management**
   - Removed unused dependencies
   - Updated all packages to latest secure versions
   - Added explicit version pinning for security

### Implementation Notes

1. **Environment Setup**

   ```bash
   # Create a new virtual environment
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   
   # Install updated dependencies
   pip install -r config/requirements.txt
   ```

2. **Security Best Practices**
   - Use environment variables for sensitive data
   - Validate all user inputs
   - Keep dependencies updated
   - Monitor security advisories

3. **Troubleshooting**
   - If you encounter SSL-related issues, ensure your system's OpenSSL is up to date
   - For certificate verification errors, check your system's certificate store
   - If you see dependency conflicts, try creating a fresh virtual environment

### Future Security Considerations

1. **Regular Updates**
   - Monitor Dependabot alerts
   - Review security advisories monthly
   - Keep dependencies up to date

2. **Security Testing**
   - Implement automated security scanning
   - Regular dependency audits
   - Penetration testing for critical features

3. **Documentation**
   - Keep security documentation updated
   - Document security-related changes
   - Maintain a security changelog

## Version 1.1.1 Security Updates (2024-06-17)

### Application Security

- File uploads are now sanitized: only .csv files with safe names are accepted
- Uploaded CSVs are validated for required columns (GS_ID, wFC, pFDR, x, y)
- Invalid or unsafe files are rejected with user-friendly warnings
- Input validation is performed on all uploaded data

### Script Security & Robustness

- Unix/macOS and Windows run scripts now check for Streamlit installation
- Scripts automatically find an available port and clean up old processes
- User-friendly error messages for missing dependencies or port conflicts

### Usage Notes

- Use `./scripts/run_streamlit.sh` (Unix/macOS) or `scripts\run_streamlit_win.bat` (Windows) to launch the app safely
- If you encounter errors, check the troubleshooting section in the README
