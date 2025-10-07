"""
Debug script to test the GCS training request and identify the validation issue.
"""

import json
from pydantic import ValidationError
from app import TrainingRequest

def test_training_request_validation():
    """Test different TrainingRequest configurations to identify the issue."""
    
    print("🧪 Testing TrainingRequest Validation")
    print("=" * 50)
    
    # Test 1: GCS data request (this should work)
    print("\n1. Testing GCS data request...")
    try:
        gcs_request = {
            "model_type": "random_forest",
            "test_size": 0.2,
            "use_sample_data": False,
            "use_gcs_data": True,
            "gcs_data_folder": "RawData",
            "gcs_data_filename": "wellness_sample.parquet"
        }
        
        request = TrainingRequest(**gcs_request)
        print("✅ GCS request validation passed")
        print(f"   use_sample_data: {request.use_sample_data}")
        print(f"   use_gcs_data: {request.use_gcs_data}")
        print(f"   data_file: {request.data_file}")
        print(f"   gcs_data_filename: {request.gcs_data_filename}")
        
    except ValidationError as e:
        print("❌ GCS request validation failed:")
        print(f"   Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    # Test 2: Sample data request
    print("\n2. Testing sample data request...")
    try:
        sample_request = {
            "model_type": "random_forest",
            "use_sample_data": True,
            "sample_size": 1000
        }
        
        request = TrainingRequest(**sample_request)
        print("✅ Sample data request validation passed")
        print(f"   use_sample_data: {request.use_sample_data}")
        print(f"   use_gcs_data: {request.use_gcs_data}")
        print(f"   data_file: {request.data_file}")
        
    except ValidationError as e:
        print("❌ Sample data request validation failed:")
        print(f"   Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    # Test 3: Local file request
    print("\n3. Testing local file request...")
    try:
        local_request = {
            "model_type": "random_forest",
            "use_sample_data": False,
            "use_gcs_data": False,
            "data_file": "custom_training_data.csv"
        }
        
        request = TrainingRequest(**local_request)
        print("✅ Local file request validation passed")
        print(f"   use_sample_data: {request.use_sample_data}")
        print(f"   use_gcs_data: {request.use_gcs_data}")
        print(f"   data_file: {request.data_file}")
        
    except ValidationError as e:
        print("❌ Local file request validation failed:")
        print(f"   Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    # Test 4: Invalid request (should fail)
    print("\n4. Testing invalid request (should fail)...")
    try:
        invalid_request = {
            "model_type": "random_forest",
            "use_sample_data": False,
            "use_gcs_data": False
            # Missing data_file
        }
        
        request = TrainingRequest(**invalid_request)
        print("❌ Invalid request validation passed (this should have failed!)")
        
    except ValidationError as e:
        print("✅ Invalid request correctly rejected:")
        print(f"   Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    # Test 5: Your exact request
    print("\n5. Testing your exact request...")
    try:
        your_request = {
            "model_type": "random_forest",
            "test_size": 0.2,
            "use_sample_data": False,
            "use_gcs_data": True,
            "gcs_data_folder": "RawData", 
            "gcs_data_filename": "wellness_sample.parquet"
        }
        
        request = TrainingRequest(**your_request)
        print("✅ Your request validation passed")
        print(f"   Request object: {request}")
        
        # Test the logic flow
        print("\n   Testing logic flow:")
        if request.use_sample_data:
            print("   → Would use sample data")
        elif request.use_gcs_data:
            print("   → Would use GCS data ✅")
            print(f"     Folder: {request.gcs_data_folder}")
            print(f"     Filename: {request.gcs_data_filename}")
        else:
            print(f"   → Would use local file: {request.data_file}")
            if not request.data_file:
                print("   ❌ This would cause the error you're seeing!")
        
    except ValidationError as e:
        print("❌ Your request validation failed:")
        print(f"   Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def show_request_example():
    """Show the correct request format."""
    print("\n" + "=" * 50)
    print("📖 Correct Request Format for GCS Training:")
    print("=" * 50)
    
    correct_request = {
        "model_type": "random_forest",
        "test_size": 0.2,
        "use_sample_data": False,
        "use_gcs_data": True,
        "gcs_data_folder": "RawData",
        "gcs_data_filename": "wellness_sample.parquet"
    }
    
    print("JSON Request Body:")
    print(json.dumps(correct_request, indent=2))
    
    print("\nCURL Command:")
    print('curl -X POST "https://mental-wellness-model.onrender.com/train" \\')
    print('  -H "Content-Type: application/json" \\')
    print(f'  -d \'{json.dumps(correct_request)}\'')

if __name__ == "__main__":
    test_training_request_validation()
    show_request_example()