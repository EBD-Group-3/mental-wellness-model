# 🗃️ GCS Data Integration Guide

This g### Supported formats in the `RawData` folder:
- **CSV files** (`.csv`) - Standard comma-separated values (always supported)
- **Parquet files** (`.parquet`) - Optimized columnar format (requires pyarrow)

**Note**: After the latest update, pyarrow is included in requirements.txt and will be available after deployment. You can check availability using the `/system/dependencies` endpoint.e explains how to use Google Cloud Storage (GCS) for training data in the Mental Wellness Model API.

## 🎯 Overview

The Mental Wellness Model now supports loading training data directly from Google Cloud Storage, eliminating the need to upload files locally. This enables:

- **Cloud-native data storage** - Store large datasets in GCS
- **Seamless integration** - Direct data loading from cloud storage
- **Multiple formats** - Support for CSV and Parquet files
- **Data versioning** - Organize datasets in structured folders

## 📁 GCS Bucket Structure

Your GCS bucket should be organized as follows:

```
mental_wellness_data_lake/
├── Model/                    # Trained models (existing)
│   ├── basic_trained_model.joblib
│   └── model_metadata.json
└── RawData/                  # Training data (new)
    ├── wellness_sample.parquet
    ├── custom_training_data.csv
    └── other_datasets.parquet
```

## 🔧 Setup Requirements

### 1. Environment Variables (Already configured for Render)
```
GOOGLE_CREDENTIALS_JSON={"type":"service_account",...}
ENVIRONMENT=production
```

### 2. GCS Bucket Permissions
Your service account needs the following permissions:
- `Storage Object Viewer` - to read data files
- `Storage Object Creator` - to upload trained models

### 3. Data File Formats
Supported formats in the `RawData` folder:
- **CSV files** (`.csv`) - Standard comma-separated values
- **Parquet files** (`.parquet`) - Optimized columnar format (recommended)

## 📡 API Endpoints

### 1. List Available Data Files
```http
GET /gcs/data
```

**Response:**
```json
{
  "gcs_data_files": [
    {
      "name": "wellness_sample.parquet",
      "full_path": "RawData/wellness_sample.parquet",
      "size_bytes": 245760,
      "updated": "2025-10-07T12:00:00Z",
      "content_type": "application/octet-stream"
    }
  ],
  "bucket": "mental_wellness_data_lake",
  "folder": "RawData",
  "total_files": 1
}
```

### 2. Preview Data File
```http
GET /gcs/data/preview?filename=wellness_sample.parquet&folder=RawData&rows=5
```

**Parameters:**
- `filename` - Name of the data file (default: "wellness_sample.parquet")
- `folder` - GCS folder name (default: "RawData")
- `rows` - Number of preview rows (default: 5)

**Response:**
```json
{
  "filename": "wellness_sample.parquet",
  "folder": "RawData",
  "total_rows": 1000,
  "total_columns": 15,
  "columns": ["age", "sleep_hours", "exercise_minutes", ...],
  "data_types": {"age": "int64", "sleep_hours": "float64", ...},
  "preview_rows": [
    {"age": 25, "sleep_hours": 7.5, "exercise_minutes": 30, ...},
    {"age": 34, "sleep_hours": 6.2, "exercise_minutes": 45, ...}
  ],
  "sample_statistics": {"age": {"mean": 35.2, "std": 12.1, ...}}
}
```

### 3. Train Model with GCS Data
```http
POST /train
```

**Request Body:**
```json
{
  "model_type": "random_forest",
  "test_size": 0.2,
  "use_sample_data": false,
  "use_gcs_data": true,
  "gcs_data_folder": "RawData",
  "gcs_data_filename": "wellness_sample.parquet"
}
```

**Parameters:**
- `use_gcs_data`: `true` - Enable GCS data loading
- `gcs_data_folder`: GCS folder containing the data file (default: "RawData")
- `gcs_data_filename`: Data file name with extension (supports .csv, .parquet)

## 🛠️ Usage Examples

### Example 1: Train with Default GCS Data
```bash
curl -X POST "https://mental-wellness-model.onrender.com/train" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "random_forest",
    "use_sample_data": false,
    "use_gcs_data": true
  }'
```

### Example 2: Train with Custom CSV Data
```bash
curl -X POST "https://mental-wellness-model.onrender.com/train" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "logistic_regression",
    "test_size": 0.3,
    "use_sample_data": false,
    "use_gcs_data": true,
    "gcs_data_folder": "RawData",
    "gcs_data_filename": "custom_training_data.csv"
  }'
```

### Example 3: Preview Data Before Training
```bash
# First, preview the data
curl "https://mental-wellness-model.onrender.com/gcs/data/preview?filename=wellness_sample.parquet&rows=3"

# Then train with the data
curl -X POST "https://mental-wellness-model.onrender.com/train" \
  -H "Content-Type: application/json" \
  -d '{
    "use_gcs_data": true,
    "gcs_data_filename": "wellness_sample.parquet"
  }'
```

### Example 4: PowerShell (Windows)
```powershell
# List available data files
Invoke-RestMethod -Uri "https://mental-wellness-model.onrender.com/gcs/data"

# Train with GCS data
$body = @{
    model_type = "random_forest"
    use_sample_data = $false
    use_gcs_data = $true
    gcs_data_folder = "RawData"
    gcs_data_filename = "wellness_sample.parquet"
} | ConvertTo-Json

Invoke-RestMethod -Uri "https://mental-wellness-model.onrender.com/train" `
  -Method POST `
  -ContentType "application/json" `
  -Body $body
```

## 🎯 Data Format Requirements

Your data files must contain the following columns for successful training:

### Required Columns:
- `age` - Age of the individual
- `sleep_hours` - Hours of sleep per night
- `exercise_minutes` - Minutes of exercise per day
- `work_stress_level` - Work stress level (1-10)
- `mood_rating` - Daily mood rating (1-10)
- `energy_level` - Energy level (1-10)

### Target Columns (at least one required):
- `depression_risk` - Depression risk score (0-1)
- `anxiety_risk` - Anxiety risk score (0-1)

### Example Data Structure:
```csv
age,sleep_hours,exercise_minutes,work_stress_level,mood_rating,energy_level,depression_risk,anxiety_risk
25,7.5,30,4,7,8,0.2,0.1
34,6.2,45,7,5,6,0.4,0.3
41,8.1,20,3,8,9,0.1,0.1
```

## 🔧 Testing

Use the provided test script to verify GCS integration:

```bash
python test_gcs_data.py
```

This script will:
1. Test health check
2. List available GCS data files
3. Preview data content
4. Train a model using GCS data

## 🐛 Troubleshooting

### Common Issues:

**1. "GCS storage not available"**
- Check that `GOOGLE_CREDENTIALS_JSON` environment variable is set
- Verify GCS bucket exists and is accessible
- Ensure service account has proper permissions

**2. "Data file not found in GCS"**
- Verify the file exists in the specified folder
- Check file name spelling and case sensitivity
- Use `/gcs/data` endpoint to list available files

**3. "Unsupported file format"**
- Ensure file extension is `.csv` or `.parquet`
- Check file is not corrupted
- Verify file contains valid data

**4. "Data validation failed"**
- Ensure all required columns are present
- Check data types match expected formats
- Remove any null values in required columns

**5. "Missing optional dependency 'pyarrow'"**
- This indicates Parquet support is not available
- **Solution**: pyarrow has been added to requirements.txt and will be installed on next deployment
- **Workaround**: Convert your Parquet file to CSV format and use that instead
- **Check status**: Use `/system/dependencies` endpoint to verify pyarrow installation

### Debug Steps:
1. Test health endpoint: `GET /health`
2. Check dependencies: `GET /system/dependencies`
3. List data files: `GET /gcs/data`
4. Preview data: `GET /gcs/data/preview`
5. Check logs in Render dashboard for detailed error messages

## 🚀 Benefits

- **Scalability** - Handle large datasets without local storage limits
- **Reliability** - Cloud storage with built-in redundancy
- **Performance** - Optimized data loading with Parquet format
- **Flexibility** - Support for multiple data formats and structures
- **Integration** - Seamless workflow with existing model training

---

🎉 **You're now ready to train models using cloud-stored data!**

For more information, see the [Render Deployment Guide](RENDER_DEPLOYMENT_GUIDE.md) and [API Documentation](http://mental-wellness-model.onrender.com/docs).