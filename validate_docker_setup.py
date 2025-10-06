"""
Quick Docker validation script.
Verifies that the container will have a working model.
"""

import os
import sys
from pathlib import Path


def validate_docker_setup():
    """Validate Docker setup for model persistence."""
    
    print("=== Docker Setup Validation ===\n")
    
    issues = []
    
    # Check 1: Basic model exists
    basic_model_path = "./models/basic_trained_model.joblib"
    if os.path.exists(basic_model_path):
        file_size = os.path.getsize(basic_model_path) / 1024 / 1024
        print(f"✅ Basic model exists: {basic_model_path} ({file_size:.2f} MB)")
    else:
        print(f"❌ Basic model missing: {basic_model_path}")
        issues.append("Basic model not found")
    
    # Check 2: GCS credentials
    creds_path = "./credentials/mentalwellness-473814-key.json"
    if os.path.exists(creds_path):
        print(f"✅ GCS credentials exist: {creds_path}")
    else:
        print(f"⚠️  GCS credentials missing: {creds_path}")
        print("   (Non-blocking - will use local models only)")
    
    # Check 3: Dockerfile
    dockerfile_path = "./Dockerfile"
    if os.path.exists(dockerfile_path):
        print(f"✅ Dockerfile exists: {dockerfile_path}")
        
        # Check if Dockerfile mentions basic model
        with open(dockerfile_path, 'r') as f:
            dockerfile_content = f.read()
            if "basic_trained_model" in dockerfile_content:
                print("   ✅ Dockerfile includes basic model handling")
            else:
                print("   ⚠️  Dockerfile may not handle basic model properly")
    else:
        print(f"❌ Dockerfile missing: {dockerfile_path}")
        issues.append("Dockerfile not found")
    
    # Check 4: Docker entrypoint
    entrypoint_path = "./docker-entrypoint.sh"
    if os.path.exists(entrypoint_path):
        print(f"✅ Docker entrypoint exists: {entrypoint_path}")
    else:
        print(f"⚠️  Docker entrypoint missing: {entrypoint_path}")
        print("   (Will use default Docker CMD)")
    
    # Check 5: Requirements include GCS
    requirements_path = "./requirements.txt"
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r') as f:
            requirements = f.read()
            if "google-cloud-storage" in requirements:
                print("✅ Requirements include Google Cloud Storage")
            else:
                print("⚠️  Requirements missing Google Cloud Storage")
    else:
        print(f"❌ Requirements file missing: {requirements_path}")
        issues.append("Requirements file not found")
    
    # Check 6: App.py model loading
    app_path = "./app.py"
    if os.path.exists(app_path):
        with open(app_path, 'r') as f:
            app_content = f.read()
            if "/app/models/basic_trained_model.joblib" in app_content:
                print("✅ App includes Docker model paths")
            else:
                print("⚠️  App may not check Docker model paths")
    else:
        print(f"❌ App file missing: {app_path}")
        issues.append("App file not found")
    
    # Summary
    print("\n=== Validation Summary ===")
    
    if not issues:
        print("🎉 All validations passed!")
        print("\n✅ Your Docker container should:")
        print("  - Build successfully")
        print("  - Find the basic model on startup")
        print("  - Load the model automatically")
        print("  - Be ready for predictions immediately")
        print("  - Not show 'Model not loaded' errors")
        
        print("\n🐳 Ready to build:")
        print("  docker build -t mental-wellness-api .")
        print("  docker run -p 8000:8000 mental-wellness-api")
        
        return True
    else:
        print(f"❌ Found {len(issues)} critical issues:")
        for issue in issues:
            print(f"  - {issue}")
        
        print("\n🔧 To fix:")
        if "Basic model not found" in issues:
            print("  Run: python create_basic_model.py")
        if "Dockerfile not found" in issues:
            print("  Ensure Dockerfile exists in project root")
        if "Requirements file not found" in issues:
            print("  Ensure requirements.txt exists")
        if "App file not found" in issues:
            print("  Ensure app.py exists")
        
        return False


def main():
    """Main validation function."""
    success = validate_docker_setup()
    
    if success:
        print("\n🚀 Docker setup is ready for deployment!")
    else:
        print("\n⚠️  Please fix the issues above before building Docker container")


if __name__ == "__main__":
    main()