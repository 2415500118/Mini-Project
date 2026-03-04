#!/usr/bin/env python3
import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    required = ['fastapi', 'uvicorn', 'torch', 'librosa', 'numpy']
    missing = []
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    return missing

def check_files():
    required_files = ['model_best.pth', 'voice_db.pkl']
    missing = []
    for file in required_files:
        if not Path(file).exists():
            missing.append(file)
    return missing

def main():
    print("\nVoxGuard - Speaker Verification System\n")
    
    missing_files = check_files()
    if missing_files:
        print(f"Error: Missing files - {', '.join(missing_files)}")
        sys.exit(1)
    
    missing_deps = check_dependencies()
    if missing_deps:
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install',
                'fastapi', 'uvicorn', 'torch', 'librosa', 'soundfile',
                'python-multipart', 'numpy', 'scikit-learn'
            ])
        except subprocess.CalledProcessError:
            print("Failed to install dependencies")
            sys.exit(1)
    
    print("Starting server at http://localhost:8081\n")
    
    try:
        import uvicorn
        uvicorn.run(
            "api:app",
            host="0.0.0.0",
            port=8081,
            reload=False,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\nServer stopped")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
