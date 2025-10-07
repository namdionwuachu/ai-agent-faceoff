# diagnose_frameworks.py
import sys
import subprocess
import os

def diagnose_environment():
    print("🔍 DIAGNOSING PYTHON ENVIRONMENT")
    print("=" * 50)
    
    # Check Python executable
    python_exe = sys.executable
    print(f"Python executable: {python_exe}")
    
    # Check Python path
    print(f"Python path: {sys.path[0]}")
    
    # Check virtual environment
    venv = os.getenv('VIRTUAL_ENV', 'Not in virtual environment')
    print(f"Virtual environment: {venv}")
    
    # Check installed packages
    print("\n📦 CHECKING INSTALLED PACKAGES:")
    try:
        result = subprocess.run([python_exe, '-m', 'pip', 'list'], 
                              capture_output=True, text=True)
        packages = ['autogen', 'langgraph', 'crewai']
        for package in packages:
            if package in result.stdout.lower():
                print(f"✅ {package}: INSTALLED")
            else:
                print(f"❌ {package}: NOT FOUND IN PIP LIST")
    except Exception as e:
        print(f"Error checking packages: {e}")
    
    # Test imports
    print("\n🔧 TESTING IMPORTS:")
    frameworks = {
        "AutoGen": "autogen",
        "LangGraph": "langgraph",
        "CrewAI": "crewai"
    }
    
    for name, package in frameworks.items():
        try:
            __import__(package)
            print(f"✅ {name}: CAN IMPORT")
        except ImportError as e:
            print(f"❌ {name}: IMPORT FAILED - {e}")

def fix_autogen():
    """Try to fix AutoGen installation"""
    print("\n🛠️ ATTEMPTING TO FIX AUTOGEN")
    print("=" * 40)
    
    python_exe = sys.executable
    
    # Method 1: Install with specific Python
    print("1. Installing with specific Python executable...")
    try:
        subprocess.run([python_exe, '-m', 'pip', 'install', 'pyautogen'], check=True)
        print("   ✅ Installation attempted")
    except Exception as e:
        print(f"   ❌ Installation failed: {e}")
    
    # Method 2: Force reinstall
    print("2. Forcing reinstall...")
    try:
        subprocess.run([python_exe, '-m', 'pip', 'install', '--force-reinstall', 'pyautogen'], check=True)
        print("   ✅ Force reinstall attempted")
    except Exception as e:
        print(f"   ❌ Force reinstall failed: {e}")
    
    # Test again
    print("3. Testing import...")
    try:
        __import__('autogen')
        print("   ✅ AutoGen now works!")
        return True
    except ImportError:
        print("   ❌ AutoGen still not working")
        return False

def main():
    print("🤖 AI FRAMEWORK INSTALLATION DIAGNOSTIC")
    print("=" * 60)
    print("Finding why AutoGen installs but won't import")
    print("=" * 60)
    
    diagnose_environment()
    
    print("\n" + "=" * 60)
    print("💡 ANALYSIS:")
    print("=" * 60)
    
    # Common causes and solutions
    issues = [
        "Virtual environment path corruption",
        "Multiple Python installations conflicting", 
        "Package installed in wrong location",
        "Shell environment variables incorrect",
        "pip and python pointing to different locations"
    ]
    
    for i, issue in enumerate(issues, 1):
        print(f"{i}. {issue}")
    
    print("\n🚀 QUICK FIXES TO TRY:")
    print("1. deactivate && source .venv/bin/activate")
    print("2. python -m pip install --force-reinstall pyautogen") 
    print("3. Check if python and pip point to same location")
    print("4. Try a fresh virtual environment")
    
    # Ask if user wants to attempt fix
    response = input("\nAttempt to fix AutoGen? (y/n): ")
    if response.lower() == 'y':
        fix_autogen()

if __name__ == "__main__":
    main()
