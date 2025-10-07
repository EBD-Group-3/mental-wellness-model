# 🚀 Render Deployment Guide for Mental Wellness Model

This guide explains how to deploy your Mental Wellness Model API to Render using environment variables for secure GCP authentication.

## 📋 Prerequisites

1. **Render Account**: Sign up at [render.com](https://render.com)
2. **GitHub Repository**: Your code should be in a GitHub repository
3. **GCP Service Account**: You need a Google Cloud service account with Storage permissions

## 🔐 Environment Variables Setup

### Required Environment Variables for Render

Set these environment variables in your Render service dashboard:

#### Option 1: Single JSON Credential (Recommended)
```
GOOGLE_CREDENTIALS_JSON={"type":"service_account","project_id":"mentalwellness-473814","private_key_id":"your-key-id","private_key":"-----BEGIN PRIVATE KEY-----\n...your-private-key...\n-----END PRIVATE KEY-----\n","client_email":"your-service-account@mentalwellness-473814.iam.gserviceaccount.com","client_id":"your-client-id","auth_uri":"https://accounts.google.com/o/oauth2/auth","token_uri":"https://oauth2.googleapis.com/token","auth_provider_x509_cert_url":"https://www.googleapis.com/oauth2/v1/certs"}
```

#### Option 2: Individual Fields
```
GCS_PROJECT_ID=mentalwellness-473814
GCS_PRIVATE_KEY=-----BEGIN PRIVATE KEY-----\n...your-private-key...\n-----END PRIVATE KEY-----\n
GCS_CLIENT_EMAIL=your-service-account@mentalwellness-473814.iam.gserviceaccount.com
GCS_PRIVATE_KEY_ID=your-private-key-id
```

#### Additional Environment Variables
```
ENVIRONMENT=production
PORT=8000
```

## 🚀 Deployment Steps

### Step 1: Create New Web Service on Render

1. Go to your Render dashboard
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Select the mental-wellness-model repository

### Step 2: Configure Build Settings

**Build Command:**
```bash
pip install -r requirements.txt && pip install -e .
```

**Start Command:**
```bash
gunicorn --config gunicorn.conf.py app:app
```

### Step 3: Set Environment Variables

In the Render dashboard:
1. Go to your service → "Environment"
2. Add all the environment variables listed above
3. **Important**: When adding `GOOGLE_CREDENTIALS_JSON`, make sure to:
   - Copy the entire JSON as one line
   - Escape any quotes if necessary
   - Ensure no line breaks in the middle of the JSON

### Step 4: Deploy

1. Click "Create Web Service"
2. Render will automatically build and deploy your application
3. Monitor the build logs for any issues

## 🔍 Verification

### Check Deployment Status

1. **Health Check**: Visit `https://your-app-name.onrender.com/health`
2. **API Documentation**: Visit `https://your-app-name.onrender.com/docs`
3. **Model Training**: Test the `/train` endpoint
4. **Predictions**: Test the `/predict` endpoint

### Debug Common Issues

#### Issue: GCS Authentication Failed
```bash
# Check logs for credential loading messages
# Should see: "✅ Loaded credentials using: Environment JSON"
```

**Solutions:**
- Verify `GOOGLE_CREDENTIALS_JSON` is properly formatted
- Check that the service account has Storage permissions
- Ensure the GCS bucket exists and is accessible

#### Issue: 502 Bad Gateway / Port Binding Error
**Problem:** Render automatically sets the `PORT` environment variable, but the app was hardcoded to port 8000.
**Solution:** The `gunicorn.conf.py` has been updated to use the dynamic PORT:

```python
# In gunicorn.conf.py
import os
port = os.getenv('PORT', '8000')
bind = f"0.0.0.0:{port}"
```

**Additional fixes made:**
- Reduced worker count for cloud deployment to avoid memory issues
- Fixed health endpoint validation errors
- Updated Dockerfile to use environment variables properly

## 🔧 Configuration Files Updated

The following files have been updated for Render deployment:

### 1. `Dockerfile`
- Removed hardcoded credential file copying
- Uses environment variables instead of mounted credential files
- Sets `ENVIRONMENT=production` for proper credential loading priority

### 2. `secure_credential_manager.py`
- Prioritizes environment variables over file-based credentials in production
- Detects Render environment via `RENDER` environment variable
- Falls back gracefully to file-based credentials in development

### 3. `mental_wellness_model/config/default.yaml`
- Added cloud deployment configuration section
- Documents required environment variables
- Includes Render-specific settings

### 4. `gcs_model_storage.py`
- Deprecated hardcoded credential file paths
- Now relies entirely on secure credential manager with environment variables

## 🏗️ Build Process Overview

1. **Render clones your repository**
2. **Dependencies installed** via `pip install -r requirements.txt`
3. **Package installed** via `pip install -e .`
4. **Environment variables loaded** from Render dashboard
5. **Application starts** using Gunicorn
6. **GCS authentication** happens automatically using environment variables

## 🔐 Security Best Practices

1. **Never commit credentials** to your repository
2. **Use environment variables** for all secrets
3. **Rotate service account keys** regularly
4. **Monitor access logs** in Google Cloud Console
5. **Use least privilege** - only grant necessary GCS permissions

## 📊 Monitoring

### Application Logs
Monitor your application through Render's dashboard:
- Real-time logs
- Error tracking
- Performance metrics

### GCS Usage
Monitor GCS usage in Google Cloud Console:
- Storage usage
- API requests
- Access patterns

## 🆘 Troubleshooting

### Common Error Messages

**"❌ No valid credentials found"**
- Check environment variable formatting
- Verify service account permissions
- Ensure correct variable names

**"Failed to create working GCS client"**
- Verify bucket exists
- Check network connectivity
- Validate service account has Storage Admin role

**"Port already in use"**
- Render handles port assignment automatically
- Don't hardcode port 8000 in production

**"Missing optional dependency 'pyarrow'"**
- This has been fixed by adding pyarrow to requirements.txt
- Render will install pyarrow on next deployment
- Alternatively, use CSV format instead of Parquet
- Check `/system/dependencies` endpoint to verify installation

### Getting Help

1. Check Render build logs
2. Review application logs in Render dashboard
3. Test GCS connection locally with same environment variables
4. Verify service account permissions in GCP Console
5. Use `/system/dependencies` endpoint to check library availability

## 🎉 Success!

Once deployed successfully, your Mental Wellness Model API will be:
- ✅ Running on Render's infrastructure
- ✅ Using secure environment variable authentication
- ✅ Storing models persistently in Google Cloud Storage
- ✅ Loading training data directly from GCS bucket
- ✅ Supporting both CSV and Parquet data formats
- ✅ Automatically handling scaling and health checks

## 🗃️ GCS Data Integration

Your deployment now supports loading training data from Google Cloud Storage:

### GCS Bucket Structure:
```
mental_wellness_data_lake/
├── Model/                    # Trained models
└── RawData/                  # Training data
    └── wellness_sample.parquet
```

### New Endpoints:
- **List Data Files**: `GET /gcs/data`
- **Preview Data**: `GET /gcs/data/preview?filename=wellness_sample.parquet`
- **Train with GCS Data**: `POST /train` with `"use_gcs_data": true`

### Usage Example:
```bash
curl -X POST "https://mental-wellness-model.onrender.com/train" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "random_forest",
    "use_sample_data": false,
    "use_gcs_data": true,
    "gcs_data_folder": "RawData",
    "gcs_data_filename": "wellness_sample.parquet"
  }'
```

📖 **For detailed GCS data integration guide, see:** [GCS_DATA_INTEGRATION.md](GCS_DATA_INTEGRATION.md)

Your API endpoints will be available at:
- `https://your-app-name.onrender.com/health` - Health check
- `https://your-app-name.onrender.com/docs` - API documentation
- `https://your-app-name.onrender.com/train` - Model training
- `https://your-app-name.onrender.com/predict` - Predictions
- `https://your-app-name.onrender.com/gcs/data` - List GCS data files
- `https://your-app-name.onrender.com/gcs/data/preview` - Preview GCS data

## 🔄 Updates and Maintenance

To update your deployment:
1. Push changes to your GitHub repository
2. Render will automatically redeploy
3. Monitor the build process
4. Verify functionality after deployment

---

**Need help?** Check the [Render documentation](https://render.com/docs) or review the application logs for specific error messages.