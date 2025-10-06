# Google Cloud Storage Model Persistence

This document explains how the Mental Wellness API integrates with Google Cloud Storage (GCS) for persistent model storage across Docker deployments.

## Overview

The API now automatically saves trained models to your Google Cloud Storage bucket `mental_wellness_data_lake` in the `Model` folder, ensuring model persistence across container restarts and deployments.

## Features

### ✅ **Automatic Model Persistence**
- Models are automatically uploaded to GCS after training
- Latest models are downloaded from GCS on container startup
- Models persist across Docker builds and deployments

### ✅ **Seamless Integration**
- No code changes required for basic usage
- Models load automatically from GCS if local storage is empty
- Falls back to local models if GCS is unavailable

### ✅ **API Endpoints for GCS Management**
- `/models/gcs` - List all models in GCS
- `/models/{model_name}/upload-to-gcs` - Upload specific model to GCS
- `/models/{model_name}/download-from-gcs` - Download model from GCS
- `/models/sync-production-to-gcs` - Sync production model to GCS

## Configuration

### **1. Google Cloud Storage Setup**

Your GCS configuration:
- **Bucket**: `mental_wellness_data_lake`
- **Model Folder**: `Model`
- **Service Account**: `google-cloud-storage-writer@mentalwellness-473814.iam.gserviceaccount.com`

### **2. Credentials**

The service account credentials are included in `credentials/mentalwellness-473814-key.json` and automatically configured in the Docker container.

### **3. Environment Variables**

```bash
GOOGLE_APPLICATION_CREDENTIALS="/app/credentials/mentalwellness-473814-key.json"
```

## Workflow

### **Training Process**
1. User calls `/train` endpoint
2. Model is trained locally
3. Model is saved locally in `/app/models/`
4. **Model is automatically uploaded to GCS**
5. Training response includes GCS upload status

### **Startup Process**
1. Container starts
2. **System checks GCS for latest models**
3. If GCS models exist and local storage is empty:
   - Downloads latest model from GCS
   - Sets it as production model
4. Falls back to local models (like `basic_trained_model.joblib`)
5. API becomes ready for predictions

### **Prediction Process**
1. User calls `/predict` endpoint
2. System uses loaded model (from GCS or local)
3. If no model is loaded, attempts to load from GCS first
4. Falls back to local models if needed

## Docker Integration

### **Dockerfile Changes**
```dockerfile
# Copy GCS credentials
COPY credentials/ /app/credentials/

# Set environment variable for GCS credentials
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/credentials/mentalwellness-473814-key.json"
```

### **Requirements**
```
google-cloud-storage>=2.10.0
```

## API Usage Examples

### **1. Check GCS Models**
```bash
curl http://localhost:8000/models/gcs
```

### **2. Train and Auto-Upload Model**
```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "random_forest",
    "test_size": 0.2,
    "use_sample_data": true,
    "sample_size": 1000
  }'
```

### **3. Upload Existing Model to GCS**
```bash
curl -X POST http://localhost:8000/models/my_model/upload-to-gcs
```

### **4. Download Model from GCS**
```bash
curl -X POST http://localhost:8000/models/gcs_model_name/download-from-gcs
```

### **5. Sync Production Model**
```bash
curl -X POST http://localhost:8000/models/sync-production-to-gcs
```

## File Structure in GCS

```
mental_wellness_data_lake/
└── Model/
    ├── api_trained_model_20251006_120000.joblib
    ├── api_trained_model_20251006_120000_metadata.json
    ├── production_model.joblib
    ├── production_model_metadata.json
    └── ...
```

Each model includes:
- **Model file**: `.joblib` binary file
- **Metadata file**: `.json` with training information, performance metrics, etc.

## Benefits

### **🚀 Persistent Models**
- Models survive container restarts
- No need to retrain after deployment
- Consistent model versions across environments

### **🔄 Automatic Sync**
- Latest models are always available
- No manual upload/download required
- Seamless model updates

### **🛡️ Reliability**
- Falls back to local models if GCS unavailable
- Multiple model loading strategies
- Robust error handling

### **📊 Model Management**
- Track model versions and metadata
- Easy model rollback and deployment
- Centralized model storage

## Monitoring

### **Health Check**
```bash
curl http://localhost:8000/health
```

Response includes:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_info": {
    "source": "gcs_download",
    "gcs_model_name": "api_trained_model_20251006_120000",
    "loaded_at": "2025-10-06T12:00:00"
  }
}
```

### **Model Info**
```bash
curl http://localhost:8000/model/info
```

Response includes:
```json
{
  "model_metadata": {...},
  "is_trained": true,
  "feature_columns": [...],
  "components_initialized": true,
  "gcs_available": true
}
```

## Troubleshooting

### **GCS Not Available**
- Check credentials file exists: `/app/credentials/mentalwellness-473814-key.json`
- Verify service account permissions
- Check network connectivity to GCS
- System falls back to local models automatically

### **Model Upload Fails**
- Verify bucket permissions
- Check disk space
- Ensure model file exists locally
- Check GCS API quotas

### **Model Download Fails**
- Verify model exists in GCS
- Check local disk space
- Verify read permissions on bucket
- System falls back to local models

## Testing

Run the GCS integration test:
```bash
python test_gcs_integration.py
```

This verifies:
- GCS connection
- Model upload/download
- Model manager integration
- End-to-end workflow

## Security

- Service account has minimal required permissions
- Credentials are containerized securely
- No credentials in environment variables in production
- Access limited to specific bucket and folder

---

**Result**: Your Mental Wellness API now has full Google Cloud Storage integration, ensuring model persistence across all Docker deployments while maintaining backward compatibility with local model storage.