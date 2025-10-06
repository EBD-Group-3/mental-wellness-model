# 🐳 Docker Compose - Secure Setup Guide

## 🔐 **Secure Credential Management Integrated**

The main `docker-compose.yml` now includes secure credential management for production deployments with Google Cloud Storage integration.

## 🚀 **Quick Start**

### **Option 1: Using Environment File (.env)**
```bash
# 1. Copy the template
cp .env.template .env

# 2. Edit .env and add your credentials (choose one method):
#    - GOOGLE_CREDENTIALS_JSON (full JSON - recommended)
#    - Or individual fields (GCS_PROJECT_ID, GCS_CLIENT_EMAIL, etc.)

# 3. Start the services
docker-compose up
```

### **Option 2: Direct Environment Variables**
```bash
# Set credentials in your shell
export GOOGLE_CREDENTIALS_JSON='{"type":"service_account",...}'

# Or set individual fields
export GCS_PROJECT_ID="mentalwellness-473814"
export GCS_CLIENT_EMAIL="google-cloud-storage-writer@mentalwellness-473814.iam.gserviceaccount.com"
# ... etc

# Start services
docker-compose up
```

### **Option 3: Docker Secrets (Highest Security)**
```bash
# Uncomment the secrets section in docker-compose.yml
# Then run:
docker-compose up
```

## 📋 **Available Services**

### **1. mental-wellness-api (Production)**
- **Port**: 8000
- **Features**: 
  - Secure GCS integration
  - Model persistence in cloud storage
  - Production-ready with Gunicorn
  - Health checks included
- **Environment**: Production optimized
- **Credentials**: Uses secure credential manager

### **2. mental-wellness-api-dev (Development)**
- **Port**: 8001  
- **Features**:
  - Auto-reload on code changes
  - Full project mounted for development
  - Uvicorn with reload
- **Environment**: Development mode

### **3. mental-wellness-cli (Command Line)**
- **Usage**: `docker-compose run mental-wellness-cli python cli.py --help`
- **Features**: Interactive CLI access
- **Environment**: Standard Python environment

### **4. mental-wellness-dev (Interactive Development)**
- **Usage**: `docker-compose exec mental-wellness-dev bash`
- **Features**: Interactive shell access
- **Environment**: Full development environment

## 🔐 **Credential Setup Options**

### **Method 1: Full JSON (Recommended)**
```bash
# In .env file:
GOOGLE_CREDENTIALS_JSON='{"type":"service_account","project_id":"mentalwellness-473814",...}'
```

### **Method 2: Individual Fields**
```bash
# In .env file:
GCS_PROJECT_ID=mentalwellness-473814
GCS_CLIENT_EMAIL=google-cloud-storage-writer@mentalwellness-473814.iam.gserviceaccount.com
GCS_PRIVATE_KEY_ID=your-private-key-id
GCS_PRIVATE_KEY=-----BEGIN PRIVATE KEY-----\nyour-key\n-----END PRIVATE KEY-----\n
```

### **Method 3: Docker Secrets**
```yaml
# Uncomment in docker-compose.yml:
secrets:
  gcs_credentials:
    file: ./credentials/mentalwellness-473814-key.json
```

## ✅ **Security Features**

- ✅ **No credentials in Git**: All sensitive data via environment variables
- ✅ **Multiple credential sources**: JSON, individual fields, secrets, defaults
- ✅ **Automatic fallbacks**: Local development → Environment → Cloud defaults
- ✅ **Production ready**: Secure for all deployment environments

## 🎯 **Common Commands**

```bash
# Start production API with secure credentials
docker-compose up mental-wellness-api

# Start development API with auto-reload
docker-compose up mental-wellness-api-dev

# Run CLI commands
docker-compose run mental-wellness-cli python cli.py train

# Interactive development shell
docker-compose run mental-wellness-dev bash

# View logs
docker-compose logs mental-wellness-api

# Stop all services
docker-compose down
```

## 🔧 **Environment Variables Reference**

| Variable | Description | Required | Example |
|----------|-------------|----------|---------|
| `GOOGLE_CREDENTIALS_JSON` | Full service account JSON | Option 1 | `{"type":"service_account",...}` |
| `GCS_PROJECT_ID` | Google Cloud project ID | Option 2 | `mentalwellness-473814` |
| `GCS_CLIENT_EMAIL` | Service account email | Option 2 | `...@mentalwellness-473814.iam.gserviceaccount.com` |
| `GCS_PRIVATE_KEY_ID` | Private key ID | Option 2 | `526ba6a9c8b5a0a2f50c27eb0920fc700ac85531` |
| `GCS_PRIVATE_KEY` | Private key content | Option 2 | `-----BEGIN PRIVATE KEY-----\n...` |
| `ENVIRONMENT` | Runtime environment | No | `production` |

## 🚨 **Important Security Notes**

1. **Never commit `.env` files** - They're automatically ignored by Git
2. **Rotate credentials regularly** - Generate new service account keys periodically  
3. **Use environment-specific configs** - Different credentials for dev/staging/prod
4. **Monitor credential usage** - Check Google Cloud Console for key usage
5. **Use least privilege** - Service account has minimal required permissions

## 🎉 **Benefits of Merged Setup**

- **Single configuration file** - No need for separate secure version
- **Environment-based security** - Credentials never stored in repository
- **Multiple deployment options** - Works locally, Docker, Kubernetes, Cloud Run
- **Developer friendly** - Easy setup with clear documentation
- **Production ready** - Enterprise-grade security practices