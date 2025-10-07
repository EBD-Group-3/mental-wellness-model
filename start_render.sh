#!/bin/bash
# Render startup script for Mental Wellness API

echo "🚀 Starting Mental Wellness API on Render..."

# Print environment info
echo "Environment: ${ENVIRONMENT:-development}"
echo "Port: ${PORT:-8000}"
echo "Render: ${RENDER:-not-set}"

# Check if GCS credentials are available
if [ -n "$GOOGLE_CREDENTIALS_JSON" ]; then
    echo "✅ GCS credentials available (JSON format)"
elif [ -n "$GCS_PROJECT_ID" ]; then
    echo "✅ GCS credentials available (individual fields)"
else
    echo "⚠️  GCS credentials not found - model persistence will be limited"
fi

# Start the application
echo "Starting Gunicorn server..."
exec gunicorn --config gunicorn.conf.py app:app