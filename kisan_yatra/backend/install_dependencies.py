#!/usr/bin/env python
"""
KisanYatra - Complete Installation Script
Installs all dependencies for crop recommendation and disease detection
"""

import subprocess
import sys

def run_command(command, description):
    """Run a shell command and report status"""
    print(f"\n{'='*60}")
    print(f"📦 {description}")
    print(f"{'='*60}")
    print(f"Running: {command}\n")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=False, text=True)
        if result.returncode == 0:
            print(f"✅ {description} - SUCCESS")
            return True
        else:
            print(f"❌ {description} - FAILED")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║     🌾 KisanYatra - Crop Recommendation System 🚜         ║
    ║               Installation Script v1.0                      ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    # List of packages to install
    packages = [
        ("pip install --upgrade pip", "Upgrading pip"),
        ("pip install streamlit", "Installing Streamlit"),
        ("pip install pandas numpy", "Installing Data Science packages"),
        ("pip install scikit-learn lightgbm", "Installing ML frameworks"),
        ("pip install torch torchvision", "Installing PyTorch"),
        ("pip install pillow matplotlib seaborn", "Installing Image/Plotting libraries"),
        ("pip install plotly", "Installing Plotly"),
        ("pip install opencv-python", "Installing OpenCV"),
        ("pip install requests", "Installing requests library"),
        ("pip install gTTS", "Installing text-to-speech support"),
        ("pip install streamlit-geolocation", "Installing location access support"),
    ]
    
    print("\n📋 INSTALLATION PLAN:")
    print("=" * 60)
    for i, (cmd, desc) in enumerate(packages, 1):
        print(f"{i}. {desc}")
    print("=" * 60)
    
    response = input("\n✅ Do you want to proceed? (yes/no): ").strip().lower()
    
    if response not in ['yes', 'y']:
        print("❌ Installation cancelled.")
        sys.exit(0)
    
    print("\n🚀 Starting installation...\n")
    
    failed = []
    for cmd, desc in packages:
        if not run_command(cmd, desc):
            failed.append(desc)
    
    # Final summary
    print(f"\n{'='*60}")
    print("📊 INSTALLATION SUMMARY")
    print(f"{'='*60}")
    
    successful = len(packages) - len(failed)
    print(f"✅ Successful: {successful}/{len(packages)}")
    
    if failed:
        print(f"❌ Failed: {len(failed)}/{len(packages)}")
        print("\nFailed packages:")
        for f in failed:
            print(f"  - {f}")
    else:
        print("\n✨ All packages installed successfully!")
    
    # Additional information
    print(f"\n{'='*60}")
    print("🎯 NEXT STEPS:")
    print(f"{'='*60}")
    print("""
1. Verify PyTorch CPU installation:
   python -c "import torch; print(torch.__version__)"

2. Start the KisanYatra app:
   streamlit run app.py

3. Open your browser:
   http://localhost:8502

4. Sign up and start using:
   - Crop Recommendation
   - Disease Detection
   - Data Analysis
    """)
    
    print(f"\n{'='*60}")
    print("📚 DOCUMENTATION:")
    print(f"{'='*60}")
    print("""
- Main app: app.py
- Disease detection: disease_detection.py
- Guide: DISEASE_DETECTION_GUIDE.md
    """)
    
    if failed:
        print(f"\n⚠️  Some packages failed to install. Try installing manually:")
        for f in failed:
            print(f"   pip install {f.lower().replace(' ', '-')}")
    else:
        print("\n🎉 Installation complete! Happy farming! 🌾")

if __name__ == "__main__":
    main()
