#!/usr/bin/env python3
"""
VoxGuard Backend Launcher
Starts the FastAPI server for speaker verification
"""

import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    required = ['fastapi', 'uvicorn', 'torch', 'librosa', 'numpy']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    return missing

def check_files():
    """Check if required model and database files exist"""
    required_files = ['model.pth', 'voice_db.pkl']
    missing = []
    
    for file in required_files:
        if not Path(file).exists():
            missing.append(file)
    
    return missing

def main():
    print("\n" + "="*60)
    print(" VoxGuard - AI Speaker Verification System")
    print(" Backend API Server Launcher")
    print("="*60 + "\n")
    
    # Check files
    print("[INFO] Checking required files...")
    missing_files = check_files()
    if missing_files:
        print(f"[ERROR] Missing files: {', '.join(missing_files)}")
        sys.exit(1)
    print("[OK] All required files found\n")
    
    # Check dependencies
    print("[INFO] Checking dependencies...")
    missing_deps = check_dependencies()
    if missing_deps:
        print(f"[WARNING] Missing packages: {', '.join(missing_deps)}")
        print("[INFO] Installing missing dependencies...")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install',
                'fastapi', 'uvicorn', 'torch', 'librosa', 'soundfile',
                'python-multipart', 'numpy', 'scikit-learn'
            ])
        except subprocess.CalledProcessError:
            print("[ERROR] Failed to install dependencies")
            sys.exit(1)
    print("[OK] All dependencies installed\n")
    
    # Start server
    print("[INFO] Starting FastAPI server...")
    print("[INFO] API available at: http://localhost:8000")
    print("[INFO] API docs at: http://localhost:8000/docs")
    print("[INFO] Frontend: Open speaker_verification.html in browser\n")
    print("Press Ctrl+C to stop the server\n")
    
    try:
        # Import after checking dependencies
        import uvicorn
        uvicorn.run(
            "api:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n[INFO] Server stopped")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
