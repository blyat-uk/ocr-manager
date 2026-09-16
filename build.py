#!/usr/bin/env python3
"""
Cross-platform build script for OCR Manager.

Usage:
    .venv/bin/python build.py

This script:
1. Creates a virtual environment
2. Installs dependencies
3. Runs PyInstaller to create a single executable

Requirements:
    - Python 3.11+
    - pip
"""

import subprocess
import sys
import shutil
from pathlib import Path

def main():
    project_dir = Path(__file__).parent
    venv_dir = project_dir / ".build-venv"
    dist_dir = project_dir / "dist"

    # Determine platform-specific paths
    if sys.platform == "win32":
        python_exe = venv_dir / "Scripts" / "python.exe"
        pip_exe = venv_dir / "Scripts" / "pip.exe"
    else:
        python_exe = venv_dir / "bin" / "python"
        pip_exe = venv_dir / "bin" / "pip"

    print("=" * 60)
    print("OCR Manager Build Script")
    print("=" * 60)

    # Step 1: Create virtual environment
    print("\n[1/4] Creating virtual environment...")
    if venv_dir.exists():
        shutil.rmtree(venv_dir)
    subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)

    # Step 2: Upgrade pip
    print("\n[2/4] Upgrading pip...")
    subprocess.run([str(pip_exe), "install", "--upgrade", "pip"], check=True)

    # Step 3: Install dependencies
    print("\n[3/4] Installing dependencies...")
    subprocess.run([
        str(pip_exe), "install",
        "PyQt6>=6.5.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "scipy>=1.14.0",
        "pyinstaller>=6.0.0",
    ], check=True)

    # Step 4: Build with PyInstaller
    print("\n[4/4] Building executable with PyInstaller...")
    subprocess.run([
        str(python_exe), "-m", "PyInstaller",
        "--clean",
        str(project_dir / "ocr-manager.spec"),
    ], check=True, cwd=project_dir)

    # Report result
    print("\n" + "=" * 60)
    print("BUILD COMPLETE")
    print("=" * 60)

    if sys.platform == "win32":
        exe_path = dist_dir / "ocr-manager.exe"
    elif sys.platform == "darwin":
        exe_path = dist_dir / "OCR Manager.app"
    else:
        exe_path = dist_dir / "ocr-manager"

    if exe_path.exists():
        if exe_path.is_dir():
            # macOS .app bundle
            size = sum(f.stat().st_size for f in exe_path.rglob('*') if f.is_file())
        else:
            size = exe_path.stat().st_size
        size_mb = size / (1024 * 1024)
        print(f"\nOutput: {exe_path}")
        print(f"Size: {size_mb:.1f} MB")
    else:
        print(f"\nWarning: Expected output not found at {exe_path}")
        print(f"Check the dist/ directory for output files.")

    # Cleanup option
    print(f"\nTo clean up build files, delete:")
    print(f"  - {venv_dir}")
    print(f"  - {project_dir / 'build'}")

if __name__ == "__main__":
    main()
