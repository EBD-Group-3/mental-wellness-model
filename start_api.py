"""
Quick startup and test script for the Mental Wellness API.
Run this to start the API and test the combined functionality.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_requirements():
    """Check if required packages are installed."""
    required_packages = ['fastapi', 'uvicorn', 'pandas', 'scikit-learn', 'joblib']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    return True

def start_api_server():
    """Start the FastAPI server."""
    print("Starting Mental Wellness API server...")
    
    # Change to the correct directory
    os.chdir(Path(__file__).parent)
    
    try:
        # Try to start with uvicorn
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "app:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\nServer stopped.")
    except Exception as e:
        print(f"Failed to start server: {e}")
        print("Trying alternative method...")
        try:
            # Fallback to running app.py directly
            subprocess.run([sys.executable, "app.py"])
        except Exception as e2:
            print(f"Alternative method also failed: {e2}")

def main():
    """Main function."""
    print("=== Mental Wellness API Startup ===\n")
    
    # Check requirements
    if not check_requirements():
        return
    
    print("✅ All required packages are available")
    print("\nStarting API server...")
    print("Once the server starts, you can:")
    print("1. Visit http://localhost:8000/docs for interactive API documentation")
    print("2. Visit http://localhost:8000/health to check the health status")
    print("3. Use the /train endpoint to train a model")
    print("4. Use the /predict endpoint to make predictions")
    print("\nPress Ctrl+C to stop the server\n")
    
    # Start the server
    start_api_server()

if __name__ == "__main__":
    main()