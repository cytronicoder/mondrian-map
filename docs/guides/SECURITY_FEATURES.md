# Security Features Documentation

## Overview

The Mondrian Map application implements multiple layers of security to protect against common web application vulnerabilities and ensure safe file handling.

## File Upload Security

### File Type Validation

- **Allowed Extensions**: Only `.csv` files are accepted
- **MIME Type Checking**: Files are validated beyond file extension alone
- **File Name Sanitization**: Only safe characters allowed in file names

```python
def is_valid_csv_file(filename):
    # Only allow .csv extension and safe characters
    return bool(re.match(r'^[\w,\s-]+\.csv$', filename))
```

### Content Validation

- **Required Columns**: Uploaded CSVs must contain: `GS_ID`, `wFC`, `pFDR`, `x`, `y`
- **Data Type Validation**: Column contents are validated for expected data types
- **Size Limits**: Streamlit's built-in file size limits are enforced

```python
def validate_csv_columns(df):
    required_columns = {'GS_ID', 'wFC', 'pFDR', 'x', 'y'}
    return required_columns.issubset(set(df.columns))
```

### Error Handling

- **User-Friendly Messages**: Clear error messages for invalid files
- **Graceful Degradation**: App continues to function even with invalid uploads
- **No Sensitive Information Exposure**: Error messages do not reveal system details

## Input Sanitization

### Regular Expression Validation

- **File Names**: Restricted to alphanumeric, spaces, commas, and hyphens
- **Path Traversal Prevention**: No directory traversal characters allowed
- **Special Character Filtering**: Prevents injection attacks

### Data Processing Security

- **Pandas Safe Loading**: Using pandas' built-in CSV validation
- **Memory Management**: Proper cleanup of uploaded file data
- **Exception Handling**: Comprehensive try-catch blocks for file operations

## Network Security

### Server Configuration

- **Configurable Binding**: Scripts can bind to localhost or all interfaces
- **Port Management**: Automatic port detection prevents conflicts
- **Process Isolation**: Clean process management and cleanup

### SSL/TLS Considerations

- **urllib3 Compatibility**: Downgraded to version compatible with LibreSSL
- **Certificate Validation**: Proper SSL certificate handling
- **Secure Connections**: HTTPS support for production deployments

## Session Management

### Streamlit Session State

- **Proper Initialization**: All session variables are initialized at startup
- **State Validation**: Session state keys are checked before access
- **Memory Management**: Session data is properly managed and cleaned up

```python
# Secure session state initialization
if 'clicked_pathway_info' not in st.session_state:
    st.session_state.clicked_pathway_info = None
```

## Error Handling & Logging

### Secure Error Messages

- **No Stack Traces**: Production errors do not expose internal system details
- **User-Friendly**: Clear, actionable error messages for users
- **Logging**: Errors are logged securely without exposing sensitive data

### Exception Management

- **Comprehensive Coverage**: All file operations are wrapped in try-catch
- **Graceful Degradation**: App continues functioning despite errors
- **Resource Cleanup**: Proper cleanup of resources on error

## Dependency Security

### Package Management

- **Version Pinning**: Exact versions specified to prevent supply chain attacks
- **Vulnerability Scanning**: Regular updates to address known CVEs
- **Minimal Dependencies**: Only necessary packages are included

### Security Updates

- **CVE Tracking**: Known vulnerabilities are addressed promptly
- **Update Documentation**: All security updates are documented
- **Testing**: Security updates are tested before deployment

## Production Security Recommendations

### Deployment Security

```bash
# Use environment variables for sensitive configuration
export STREAMLIT_SERVER_ADDRESS=127.0.0.1
export STREAMLIT_SERVER_PORT=8501

# Run with restricted permissions
sudo -u streamlit-user ./scripts/run_streamlit.sh
```

### Reverse Proxy Configuration

```nginx
# Nginx configuration for production
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Firewall Configuration

```bash
# UFW firewall rules
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw deny 8501/tcp  # Block direct access to Streamlit
sudo ufw enable
```

## Security Monitoring

### Log Monitoring

- **Access Logs**: Monitor for unusual access patterns
- **Error Logs**: Track application errors and potential attacks
- **Upload Logs**: Monitor file upload attempts and failures

### Alerting

- **Failed Upload Attempts**: Alert on repeated invalid file uploads
- **Error Spikes**: Monitor for unusual error rates
- **Resource Usage**: Track memory and CPU usage for DoS detection

## Vulnerability Assessment

### Regular Security Checks

1. **Dependency Scanning**: Use tools like `safety` or `pip-audit`
2. **Code Analysis**: Static analysis for security issues
3. **Penetration Testing**: Regular security assessments

### Security Testing Commands

```bash
# Check for known vulnerabilities
pip install safety
safety check -r config/requirements.txt

# Audit dependencies
pip install pip-audit
pip-audit -r config/requirements.txt

# Code security analysis
pip install bandit
bandit -r apps/ src/
```

## Compliance Considerations

### Data Privacy

- **No Persistent Storage**: Uploaded files are not stored permanently
- **Session Isolation**: User sessions are isolated from each other
- **Data Minimization**: Only necessary data is processed

### GDPR Compliance

- **Data Processing**: Clear purpose for data processing
- **User Consent**: Implicit consent through file upload action
- **Data Retention**: No long-term data retention

## Security Incident Response

### Incident Handling

1. **Immediate Response**: Stop the application if security breach detected
2. **Assessment**: Evaluate the scope and impact of the incident
3. **Containment**: Isolate affected systems
4. **Recovery**: Restore from clean backups if necessary
5. **Documentation**: Document the incident and response

### Contact Information

- **Security Team**: <security@your-domain.com>
- **Emergency Contact**: +1-XXX-XXX-XXXX
- **Incident Reporting**: <incidents@your-domain.com>

## Security Best Practices for Users

### File Upload Guidelines

1. **Trusted Sources**: Only upload files from trusted sources
2. **File Validation**: Verify file contents before upload
3. **Size Limits**: Keep file sizes reasonable to prevent DoS
4. **Data Validation**: Ensure uploaded data does not contain sensitive information

### Browser Security

1. **Updated Browser**: Use updated browsers with security patches
2. **HTTPS**: Always use HTTPS in production
3. **Private Browsing**: Consider using private/incognito mode for sensitive data

## Future Security Enhancements

### Planned Improvements

1. **Rate Limiting**: Implement upload rate limiting
2. **User Authentication**: Add optional user authentication
3. **Audit Logging**: Comprehensive audit trail
4. **Content Security Policy**: Implement CSP headers
5. **File Scanning**: Integrate malware scanning for uploads

### Security Roadmap

- **Q1 2024**: Implement rate limiting and enhanced logging
- **Q2 2024**: Add user authentication and authorization
- **Q3 2024**: Implement comprehensive audit logging
- **Q4 2024**: Add advanced threat detection and response
