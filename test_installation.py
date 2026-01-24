#!/usr/bin/env python3
"""Test script to verify SceneScriber AI installation."""

import sys
import subprocess
from pathlib import Path

def check_python():
    """Check Python version and packages."""
    print("🔍 Checking Python installation...")
    
    # Check Python version
    python_version = sys.version_info
    print(f"  Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 9):
        print("  ❌ Python 3.9+ required")
        return False
    
    print("  ✅ Python version OK")
    return True

def check_ffmpeg():
    """Check if FFmpeg is installed."""
    print("\n🔍 Checking FFmpeg installation...")
    
    try:
        # Check ffmpeg
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            # Extract version from first line
            lines = result.stdout.split('\n')
            if lines:
                print(f"  ✅ FFmpeg found: {lines[0].split()[2]}")
        else:
            print("  ❌ FFmpeg not found or not working")
            return False
        
        # Check ffprobe
        result = subprocess.run(['ffprobe', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            if lines:
                print(f"  ✅ FFprobe found: {lines[0].split()[2]}")
        else:
            print("  ❌ FFprobe not found or not working")
            return False
        
        return True
        
    except FileNotFoundError:
        print("  ❌ FFmpeg/FFprobe not found in PATH")
        return False

def check_backend_dependencies():
    """Check if backend dependencies can be imported."""
    print("\n🔍 Checking backend dependencies...")
    
    # Try to import core modules
    modules_to_test = [
        ("fastapi", "FastAPI"),
        ("uvicorn", "run"),
        ("PIL", "Image"),
        ("numpy", "array"),
    ]
    
    all_ok = True
    for module_name, attribute in modules_to_test:
        try:
            __import__(module_name)
            print(f"  ✅ {module_name} OK")
        except ImportError as e:
            print(f"  ❌ {module_name}: {e}")
            all_ok = False
    
    return all_ok

def check_project_structure():
    """Check if project files exist."""
    print("\n🔍 Checking project structure...")
    
    required_files = [
        "backend/src/main.py",
        "backend/src/models.py",
        "backend/src/srt_exporter.py",
        "backend/src/ai_describer.py",
        "backend/src/scene_detector_simple.py",
        "frontend/package.json",
        "frontend/src/App.tsx",
        "README.md",
    ]
    
    all_ok = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} (missing)")
            all_ok = False
    
    return all_ok

def check_frontend():
    """Check frontend setup."""
    print("\n🔍 Checking frontend setup...")
    
    package_json = Path("frontend/package.json")
    if not package_json.exists():
        print("  ❌ frontend/package.json not found")
        return False
    
    print("  ✅ Frontend package.json exists")
    
    # Check if node_modules exists
    node_modules = Path("frontend/node_modules")
    if node_modules.exists():
        print("  ✅ node_modules directory exists")
    else:
        print("  ⚠️  node_modules not found (run 'npm install' in frontend/)")
    
    return True

def main():
    """Run all checks."""
    print("🎬 SceneScriber AI Installation Test")
    print("=" * 50)
    
    checks = [
        ("Python", check_python),
        ("FFmpeg", check_ffmpeg),
        ("Backend Dependencies", check_backend_dependencies),
        ("Project Structure", check_project_structure),
        ("Frontend", check_frontend),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"  ❌ Error during {name} check: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {name}")
    
    print(f"\n🎯 Score: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n✨ All checks passed! You're ready to run SceneScriber AI.")
        print("\nTo start the application:")
        print("1. Backend: cd backend && source venv/bin/activate && uvicorn src.main:app --reload --port 8000")
        print("2. Frontend: cd frontend && npm run dev")
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        print("\nQuick fixes:")
        print("- Install FFmpeg: sudo apt install ffmpeg (Ubuntu) or brew install ffmpeg (macOS)")
        print("- Install Python packages: cd backend && pip install -r requirements-minimal.txt")
        print("- Setup frontend: cd frontend && npm install")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)