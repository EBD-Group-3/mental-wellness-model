#!/bin/bash

# Environment Setup Script for Secure GCS Authentication
# Use this to set up credentials for your deployment environment

echo "🔧 Setting up Secure GCS Authentication"
echo "======================================"

# Check if credentials file exists (for development)
if [ -f "credentials/mentalwellness-473814-key.json" ]; then
    echo "📁 Found credentials file - extracting values..."
    
    # Extract values from JSON for environment variables
    PROJECT_ID=$(cat credentials/mentalwellness-473814-key.json | jq -r '.project_id')
    PRIVATE_KEY_ID=$(cat credentials/mentalwellness-473814-key.json | jq -r '.private_key_id')
    PRIVATE_KEY=$(cat credentials/mentalwellness-473814-key.json | jq -r '.private_key')
    CLIENT_EMAIL=$(cat credentials/mentalwellness-473814-key.json | jq -r '.client_email')
    
    echo "📋 Environment Variable Setup:"
    echo "=============================="
    echo "export GCS_PROJECT_ID=\"$PROJECT_ID\""
    echo "export GCS_PRIVATE_KEY_ID=\"$PRIVATE_KEY_ID\""
    echo "export GCS_CLIENT_EMAIL=\"$CLIENT_EMAIL\""
    echo "export GCS_PRIVATE_KEY=\"$PRIVATE_KEY\""
    echo ""
    echo "💡 Copy and run these commands in your shell"
    
    # Option for full JSON
    echo ""
    echo "📋 Alternative - Full JSON Setup:"
    echo "================================="
    echo "export GOOGLE_CREDENTIALS_JSON='$(cat credentials/mentalwellness-473814-key.json | tr -d '\n')'"
    
else
    echo "❌ Credentials file not found"
    echo "💡 Please obtain new service account key from Google Cloud Console"
    echo ""
    echo "📋 Manual Environment Variable Setup:"
    echo "====================================="
    echo "export GCS_PROJECT_ID=\"mentalwellness-473814\""
    echo "export GCS_PRIVATE_KEY_ID=\"your-private-key-id\""
    echo "export GCS_CLIENT_EMAIL=\"google-cloud-storage-writer@mentalwellness-473814.iam.gserviceaccount.com\""
    echo "export GCS_PRIVATE_KEY=\"-----BEGIN PRIVATE KEY-----"
    echo "your-private-key-content"
    echo "-----END PRIVATE KEY-----\""
fi

echo ""
echo "🐳 Docker Deployment:"
echo "===================="
echo "docker run -e GOOGLE_CREDENTIALS_JSON='\$(cat credentials/mentalwellness-473814-key.json)' your-app"
echo ""
echo "☁️  Cloud Deployment:"
echo "===================="
echo "In Cloud Run/GKE, default service account is used automatically"
echo ""
echo "✅ Next Steps:"
echo "============="
echo "1. Set environment variables using the commands above"
echo "2. Test with: python secure_credential_manager.py"
echo "3. Run your application"