# Deployment Guide: Streamlit Application Hosting Options

## Option 1: Streamlit Community Cloud (Recommended)

**Optimal for**: Official Streamlit hosting with seamless GitHub integration

### Deployment Steps

1. **Ensure your code is on GitHub** (already done)
2. **Visit** [share.streamlit.io](https://share.streamlit.io)
3. **Sign in** with your GitHub account
4. **Click "New app"**
5. **Repository**: Select `aimed-lab/mondrian-map`
6. **Branch**: `main`
7. **Main file path**: `app.py`
8. **Click "Deploy!"**

### Advantages

- Free tier available
- Automatic deployments from GitHub commits
- Built-in secrets management
- Support for custom domains
- SSL certificates provided
- No configuration files required for basic use

### Requirements

- Public GitHub repository (you have this)
- `requirements.txt` file (available)

---

## Option 2: Heroku

**Optimal for**: Traditional Platform-as-a-Service with extensive add-ons

### Deployment Steps

1. **Install Heroku CLI**: [Download here](https://devcenter.heroku.com/articles/heroku-cli)
2. **Login to Heroku**:

   ```bash
   heroku login
   ```

3. **Create Heroku app**:

   ```bash
   heroku create your-mondrian-map-app
   ```

4. **Deploy**:

   ```bash
   git push heroku main
   ```

### Files Created

- `Procfile` - Heroku process definition
- `setup.sh` - Streamlit configuration script
- `runtime.txt` - Python version specification

### Advantages

- Free tier available (550 hours/month)
- Extensive add-ons ecosystem
- Support for custom domains
- Environment variables are supported

---

## Option 3: Railway

**Optimal for**: Contemporary deployment platform with excellent developer experience

### Deployment Steps

1. **Visit** [railway.app](https://railway.app)
2. **Sign in** with GitHub
3. **Click "New Project"**
4. **Select "Deploy from GitHub repo"**
5. **Choose** `aimed-lab/mondrian-map`
6. **Deploy automatically**

### Files Created

- `railway.toml` - Railway configuration

### Advantages

- $5 free credit monthly
- Automatic deployments
- Modern dashboard
- Streamlined environment variable management

---

## Option 4: Render

**Optimal for**: Streamlined deployment with substantial free tier

### Deployment Steps

1. **Visit** [render.com](https://render.com)
2. **Sign up** with GitHub
3. **Click "New +" → "Web Service"**
4. **Connect** your GitHub repository
5. **Configure**:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
6. **Deploy**

### Advantages

- Free tier with 750 hours/month
- Automatic SSL
- Support for custom domains
- Straightforward database integration

---

## Option 5: Hugging Face Spaces

**Optimal for**: Machine learning and AI applications with active community features

### Deployment Steps

1. **Visit** [huggingface.co/spaces](https://huggingface.co/spaces)
2. **Create new Space**
3. **Choose Streamlit SDK**
4. **Upload your files** or connect GitHub
5. **Deploy**

### Requirements

- Rename `requirements.txt` to `requirements.txt`
- Add `app.py` as main file

### Advantages

- Free tier available
- Suitable for ML applications
- Community features and sharing options

---

## Recommended Deployment Strategy

### For Mondrian Map Application

**Primary Recommendation**: Streamlit Community Cloud

- Specifically designed for Streamlit applications
- Minimal configuration overhead
- Automatic deployment on repository updates
- Official platform support

**Secondary Recommendation**: Railway

- Contemporary platform infrastructure
- Competitive free tier offerings
- Streamlined configuration process

### Quick Start (Streamlit Community Cloud)

1. Navigate to [share.streamlit.io](https://share.streamlit.io)
2. Authenticate using GitHub credentials
3. Select "New app"
4. Choose `aimed-lab/mondrian-map` repository
5. Specify main file: `app.py`
6. Initiate deployment

Your application will be accessible at: `https://your-app-name.streamlit.app`

## Configuration Options

### Environment Variable Management

- **Streamlit Community Cloud**: Use the application dashboard
- **Heroku**: `heroku config:set VAR_NAME=value`
- **Railway**: Use the application dashboard interface
- **Render**: Use the application dashboard interface

### Custom Domain Configuration

- Most platforms support custom domain assignment
- Custom domain functionality typically requires premium tier
- Free platform-provided subdomains are available

### Performance Optimization Strategies

- Utilize `requirements_minimal.txt` for accelerated builds
- Implement Streamlit caching with `@st.cache_data`
- Optimize data loading processes and workflows

## Critical Deployment Considerations

1. **Data Directory**: Ensure `data/` directory is included in repository
2. **Path Configuration**: Use relative paths (implementation complete)
3. **Resource Constraints**: Free tiers have memory and processing limitations
4. **Idle Management**: Free-tier applications may suspend after inactivity periods

---

## Deployment Comparison Table

| Platform | Free Tier | Custom Domain | Auto Deploy | Best For |
|----------|-----------|---------------|-------------|----------|
| Streamlit Cloud | Unlimited | Yes | Yes | Streamlit apps |
| Heroku | 550h/month | Yes | Yes | General apps |
| Railway | $5 credit | Yes | Yes | Modern apps |
| Render | 750h/month | Yes | Yes | Web services |
| HF Spaces | Unlimited | No | Yes | ML apps |

**Recommended: Streamlit Community Cloud** for this use case.
