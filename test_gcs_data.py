"""
Test script for GCS data training functionality.
Demonstrates how to train a model using data from GCS bucket.
"""

import requests
import json

# Base URL for the API
BASE_URL = "http://localhost:8000"  # Change to your Render URL when deployed

def test_gcs_data_endpoints():
    """Test the new GCS data endpoints."""
    
    print("🧪 Testing GCS Data Integration")
    print("=" * 50)
    
    # 1. Test health check
    print("\n1. Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            health_data = response.json()
            print(f"   Model loaded: {health_data.get('model_loaded', 'Unknown')}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")
    
    # 2. Test listing GCS data files
    print("\n2. Testing GCS data file listing...")
    try:
        response = requests.get(f"{BASE_URL}/gcs/data")
        if response.status_code == 200:
            data = response.json()
            print("✅ GCS data files listed successfully")
            print(f"   Total files: {data.get('total_files', 0)}")
            for file_info in data.get('gcs_data_files', []):
                print(f"   - {file_info['name']} ({file_info['size_bytes']} bytes)")
        else:
            print(f"❌ GCS data listing failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ GCS data listing error: {e}")
    
    # 3. Test data preview
    print("\n3. Testing GCS data preview...")
    try:
        response = requests.get(f"{BASE_URL}/gcs/data/preview?filename=wellness_sample.parquet&rows=3")
        if response.status_code == 200:
            data = response.json()
            print("✅ GCS data preview successful")
            print(f"   Filename: {data.get('filename')}")
            print(f"   Total rows: {data.get('total_rows')}")
            print(f"   Total columns: {data.get('total_columns')}")
            print(f"   Columns: {', '.join(data.get('columns', []))}")
            print("   Preview data:")
            for i, row in enumerate(data.get('preview_rows', [])):
                print(f"     Row {i+1}: {dict(list(row.items())[:3])}...")  # Show first 3 columns
        else:
            print(f"❌ GCS data preview failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ GCS data preview error: {e}")
    
    # 4. Test training with GCS data
    print("\n4. Testing model training with GCS data...")
    try:
        training_request = {
            "model_type": "random_forest",
            "test_size": 0.2,
            "use_sample_data": False,
            "use_gcs_data": True,
            "gcs_data_folder": "RawData",
            "gcs_data_filename": "wellness_sample.parquet"
        }
        
        response = requests.post(
            f"{BASE_URL}/train",
            json=training_request,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Model training with GCS data successful")
            print(f"   Model type: {data.get('model_type')}")
            print(f"   Training samples: {data.get('training_samples')}")
            print(f"   Test samples: {data.get('test_samples')}")
            print(f"   Accuracy: {data.get('performance_metrics', {}).get('accuracy', 'N/A')}")
            print(f"   GCS upload: {'✅' if data.get('gcs_upload_success') else '❌'}")
        else:
            print(f"❌ Model training failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Model training error: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 GCS Data Integration Test Complete!")

def show_usage_examples():
    """Show usage examples for the new GCS data functionality."""
    
    print("\n📖 Usage Examples")
    print("=" * 50)
    
    print("\n1. List available data files in GCS:")
    print("   GET /gcs/data")
    
    print("\n2. Preview data from GCS:")
    print("   GET /gcs/data/preview?filename=wellness_sample.parquet&rows=5")
    
    print("\n3. Train model using GCS data:")
    print("   POST /train")
    print("   Body:")
    example_request = {
        "model_type": "random_forest",
        "test_size": 0.2,
        "use_sample_data": False,
        "use_gcs_data": True,
        "gcs_data_folder": "RawData",
        "gcs_data_filename": "wellness_sample.parquet"
    }
    print(f"   {json.dumps(example_request, indent=2)}")
    
    print("\n4. Train model using custom GCS data:")
    print("   POST /train")
    print("   Body:")
    custom_request = {
        "model_type": "logistic_regression",
        "test_size": 0.3,
        "use_sample_data": False,
        "use_gcs_data": True,
        "gcs_data_folder": "RawData",
        "gcs_data_filename": "your_custom_data.csv"
    }
    print(f"   {json.dumps(custom_request, indent=2)}")

if __name__ == "__main__":
    print("🧠 Mental Wellness Model - GCS Data Integration Test")
    
    # Show usage examples
    show_usage_examples()
    
    # Ask if user wants to run tests
    print("\n" + "=" * 50)
    run_tests = input("Do you want to run the tests? (y/n): ").lower().strip()
    
    if run_tests == 'y':
        # Run the tests
        test_gcs_data_endpoints()
    else:
        print("Tests skipped. Use the examples above to test manually.")