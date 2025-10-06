# Credentials Directory

🔐 **This directory contains sensitive authentication files and should NEVER be committed to Git.**

## Purpose
Store Google Cloud service account keys and other authentication credentials here for local development.

## Security Notice
- ✅ This folder is excluded from Git via `.gitignore`
- ✅ Files here are used only for local development
- ✅ Production deployments use environment variables instead

## Expected Files
- `mentalwellness-473814-key.json` - Google Cloud service account key (for local development)
- Other credential files as needed

## Production Deployment
In production, credentials are loaded via environment variables:
- `GOOGLE_CREDENTIALS_JSON` - Full JSON credentials as string
- Or individual fields: `GCS_PROJECT_ID`, `GCS_CLIENT_EMAIL`, etc.

## Getting Credentials
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Navigate to IAM & Admin → Service Accounts
3. Find `google-cloud-storage-writer@mentalwellness-473814.iam.gserviceaccount.com`
4. Create and download a new key
5. Save as `mentalwellness-473814-key.json` in this directory

## Important
- Never commit credential files to Git
- Rotate keys regularly
- Use environment variables for deployment