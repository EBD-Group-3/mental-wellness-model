@echo off
REM Environment Setup Script for Secure GCS Authentication (Windows)
REM Use this to set up credentials for your deployment environment

echo 🔧 Setting up Secure GCS Authentication
echo ======================================

REM Check if credentials file exists (for development)
if exist "credentials\mentalwellness-473814-key.json" (
    echo 📁 Found credentials file
    echo.
    echo 📋 Windows Environment Variable Setup:
    echo ====================================
    echo.
    echo For Command Prompt:
    echo -------------------
    echo set GCS_PROJECT_ID=mentalwellness-473814
    echo set GCS_CLIENT_EMAIL=google-cloud-storage-writer@mentalwellness-473814.iam.gserviceaccount.com
    echo set GCS_PRIVATE_KEY_ID=your-private-key-id
    echo set GCS_PRIVATE_KEY=-----BEGIN PRIVATE KEY----- your-key -----END PRIVATE KEY-----
    echo.
    echo For PowerShell:
    echo ---------------
    echo $env:GCS_PROJECT_ID="mentalwellness-473814"
    echo $env:GCS_CLIENT_EMAIL="google-cloud-storage-writer@mentalwellness-473814.iam.gserviceaccount.com"
    echo $env:GCS_PRIVATE_KEY_ID="your-private-key-id"
    echo $env:GCS_PRIVATE_KEY="-----BEGIN PRIVATE KEY----- your-key -----END PRIVATE KEY-----"
    echo.
    echo JSON Method ^(PowerShell^):
    echo ---------------------------
    echo $env:GOOGLE_CREDENTIALS_JSON=^(Get-Content credentials\mentalwellness-473814-key.json -Raw^)
    
) else (
    echo ❌ Credentials file not found
    echo 💡 Please obtain new service account key from Google Cloud Console
    echo.
    echo 📋 Manual Environment Variable Setup:
    echo ===================================
    echo set GCS_PROJECT_ID=mentalwellness-473814
    echo set GCS_CLIENT_EMAIL=google-cloud-storage-writer@mentalwellness-473814.iam.gserviceaccount.com
    echo set GCS_PRIVATE_KEY_ID=your-private-key-id
    echo set GCS_PRIVATE_KEY=-----BEGIN PRIVATE KEY----- your-key -----END PRIVATE KEY-----
)

echo.
echo 🐳 Docker Deployment:
echo ====================
echo docker run -e GOOGLE_CREDENTIALS_JSON="$(Get-Content credentials\mentalwellness-473814-key.json -Raw)" your-app
echo.
echo ☁️  Cloud Deployment:
echo ====================
echo In Cloud Run/GKE, default service account is used automatically
echo.
echo ✅ Next Steps:
echo =============
echo 1. Set environment variables using the commands above
echo 2. Test with: python secure_credential_manager.py
echo 3. Run your application

pause