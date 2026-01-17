# Deployment guide: Free Streamlit hosting

## Option 1: Streamlit Community Cloud (Recommended)

**Best for:** Official Streamlit hosting with seamless GitHub integration

### Steps

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

**Best for:** Traditional PaaS with extensive add-ons

### Steps

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

**Best for:** Modern deployment platform with great developer experience

### Steps

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
- Easy environment variables management

---

## Option 4: Render

**Best for:** Simple deployment with good free tier

### Steps

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
- Simple database integration

---

## Option 5: Hugging Face Spaces

**Best for:** ML/AI applications with great community

### Steps

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

## Recommended approach

### **For Your Mondrian Map App:**

**1st Choice: Streamlit Community Cloud**

- Perfect for Streamlit apps
- Zero configuration needed
- Automatic deployments
- Official support

**2nd Choice: Railway**

- Modern platform
- Great free tier
- Easy setup

### **Quick Start (Streamlit Community Cloud):**

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select `aimed-lab/mondrian-map`
5. Set main file: `app.py`
6. Deploy!

Your app will be live at: `https://your-app-name.streamlit.app`

---

## Configuration tips

### **Environment Variables (if needed):**

- Streamlit Cloud: Use the dashboard
- Heroku: `heroku config:set VAR_NAME=value`
- Railway: Use the dashboard
- Render: Use the dashboard

### **Custom Domain:**

- Most platforms support custom domains
- Usually requires upgrading to paid tier
- Free subdomains are provided

### **Performance Optimization:**

- Use `requirements_minimal.txt` for faster builds
- Enable caching with `@st.cache_data`
- Optimize data loading

---

## 🚨 **Important Notes**

1. **Data Files**: Ensure your `data/` directory is included in the repository
2. **File Paths**: Use relative paths (already implemented)
3. **Memory Limits**: Free tiers have memory restrictions
4. **Sleep Mode**: Free apps may sleep after inactivity

---

## 📊 **Comparison Table**

| Platform | Free Tier | Custom Domain | Auto Deploy | Best For |
|----------|-----------|---------------|-------------|----------|
| Streamlit Cloud | Unlimited | Yes | Yes | Streamlit apps |
| Heroku | 550h/month | Yes | Yes | General apps |
| Railway | $5 credit | Yes | Yes | Modern apps |
| Render | 750h/month | Yes | Yes | Web services |
| HF Spaces | Unlimited | No | Yes | ML apps |

**Recommended: Streamlit Community Cloud** for this use case.
