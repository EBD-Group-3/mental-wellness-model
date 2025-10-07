#!/usr/bin/env python3
"""
Simple startup test for Render deployment.
This script tests if the application can start and respond to basic requests.
"""

import os
import sys
import time
import requests
import subprocess
from pathlib import Path

# Add the package to the path
sys.path.insert(0, str(Path(__file__).parent))

def test_startup():
    """Test application startup and basic endpoints."""
    print("🚀 Testing Mental Wellness API startup...")
    
    # Get port from environment or use default
    port = os.environ.get('PORT', '8000')
    base_url = f"http://localhost:{port}"
    
    print(f"Using port: {port}")
    print(f"Base URL: {base_url}")
    
    # Start the application in background
    print("\n1. Starting application...")
    try:
        # Start with gunicorn
        cmd = ["gunicorn", "--config", "gunicorn.conf.py", "app:app"]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Give it time to start
        print("Waiting for application to start...")
        for i in range(30):  # Wait up to 30 seconds
            time.sleep(1)
            try:
                response = requests.get(f"{base_url}/health", timeout=5)
                if response.status_code == 200:
                    print(f"✅ Application started successfully after {i+1} seconds")
                    break
            except requests.exceptions.RequestException:
                if i == 29:
                    print("❌ Application failed to start within 30 seconds")
                    return False
                continue
        
        # Test health endpoint
        print("\n2. Testing /health endpoint...")
        try:
            response = requests.get(f"{base_url}/health", timeout=10)
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.json()}")
            
            if response.status_code == 200:
                print("✅ Health check passed")
            else:
                print(f"❌ Health check failed with status {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False
        
        # Test root endpoint
        print("\n3. Testing root endpoint...")
        try:
            response = requests.get(base_url, timeout=10)
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.json()}")
            
            if response.status_code == 200:
                print("✅ Root endpoint test passed")
            else:
                print(f"❌ Root endpoint failed with status {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Root endpoint test failed: {e}")
            return False
        
        print("\n🎉 All tests passed! Application is ready for deployment.")
        return True
        
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        return False
    
    finally:
        # Clean up
        if 'process' in locals():
            process.terminate()
            process.wait()

def check_environment():
    """Check environment setup for deployment."""
    print("🔍 Checking environment setup...")
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check required packages
    required_packages = ['fastapi', 'uvicorn', 'gunicorn', 'pandas', 'scikit-learn']
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            print(f"❌ {package} is missing")
            return False
    
    # Check environment variables
    env_vars = {
        'PORT': os.environ.get('PORT', 'Not set (will use default 8000)'),
        'ENVIRONMENT': os.environ.get('ENVIRONMENT', 'Not set'),
        'RENDER': os.environ.get('RENDER', 'Not set'),
        'GOOGLE_CREDENTIALS_JSON': 'Set' if os.environ.get('GOOGLE_CREDENTIALS_JSON') else 'Not set'
    }
    
    print("\nEnvironment variables:")
    for var, value in env_vars.items():
        print(f"  {var}: {value}")
    
    return True

if __name__ == "__main__":
    print("🧠 Mental Wellness API - Startup Test")
    print("=" * 50)
    
    # Check environment first
    if not check_environment():
        print("❌ Environment check failed")
        sys.exit(1)
    
    # Test startup
    if test_startup():
        print("\n✅ Startup test completed successfully")
        sys.exit(0)
    else:
        print("\n❌ Startup test failed")
        sys.exit(1)