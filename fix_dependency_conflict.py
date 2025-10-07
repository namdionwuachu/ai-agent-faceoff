# fix_dependency_conflict.py
import subprocess
import sys

def diagnose_dependencies():
    print("🔍 DIAGNOSING DEPENDENCY CONFLICTS")
    print("=" * 50)
    
    python_exe = sys.executable
    
    # Check current google-ai-generativelanguage version
    print("1. Checking current versions:")
    try:
        result = subprocess.run([
            python_exe, '-c', 
            "import pkg_resources; print('google-ai-generativelanguage:', [p.version for p in pkg_resources.working_set if p.key=='google-ai-generativelanguage'])"
        ], capture_output=True, text=True)
        print(f"   {result.stdout.strip()}")
    except:
        print("   Could not check version")
    
    # Check what requires it
    print("\n2. Checking what requires google-ai-generativelanguage:")
    try:
        result = subprocess.run([
            python_exe, '-m', 'pip', 'show', 'google-ai-generativelanguage'
        ], capture_output=True, text=True)
        if "Required-by" in result.stdout:
            for line in result.stdout.split('\n'):
                if "Required-by" in line:
                    print(f"   {line}")
    except:
        print("   Could not check dependencies")

def fix_conflict():
    """Fix the dependency conflict"""
    print("\n🛠️ FIXING DEPENDENCY CONFLICT")
    print("=" * 40)
    
    python_exe = sys.executable
    
    # Option 1: Upgrade google-ai-generativelanguage
    print("1. Upgrading google-ai-generativelanguage...")
    try:
        subprocess.run([
            python_exe, '-m', 'pip', 'install', '--upgrade', 'google-ai-generativelanguage'
        ], check=True)
        print("   ✅ Upgrade attempted")
    except Exception as e:
        print(f"   ❌ Upgrade failed: {e}")
    
    # Option 2: Reinstall langchain-google-genai with correct deps
    print("2. Reinstalling langchain-google-genai...")
    try:
        subprocess.run([
            python_exe, '-m', 'pip', 'uninstall', '-y', 'langchain-google-genai'
        ])
        subprocess.run([
            python_exe, '-m', 'pip', 'install', 'langchain-google-genai'
        ], check=True)
        print("   ✅ Reinstall attempted")
    except Exception as e:
        print(f"   ❌ Reinstall failed: {e}")
    
    # Test AutoGen
    print("3. Testing AutoGen import...")
    try:
        __import__('autogen')
        print("   ✅ AutoGen now works!")
        return True
    except ImportError as e:
        print(f"   ❌ AutoGen still broken: {e}")
        return False

def nuclear_option():
    """Last resort: fresh environment"""
    print("\n💣 NUCLEAR OPTION: Fresh Environment")
    print("=" * 45)
    
    print("If all else fails, create a fresh virtual environment:")
    print("""
    deactivate
    rm -rf .venv
    python -m venv .venv
    source .venv/bin/activate
    pip install pyautogen langgraph crewai
    """)

def main():
    print("🤖 DEPENDENCY CONFLICT RESOLUTION")
    print("=" * 60)
    print("Fixing the silent dependency war breaking AutoGen")
    print("=" * 60)
    
    diagnose_dependencies()
    
    print("\n💡 THE PROBLEM:")
    print("=" * 20)
    print("langchain-google-genai 2.1.12 requires google-ai-generativelanguage>=0.7")
    print("But you have google-ai-generativelanguage 0.6.15")
    print("This creates a silent conflict that breaks AutoGen!")
    
    response = input("\nAttempt to fix dependency conflict? (y/n): ")
    if response.lower() == 'y':
        success = fix_conflict()
        if not success:
            nuclear_option()
    
    print("\n🎯 FOR YOUR YOUTUBE VIDEO:")
    print("=" * 30)
    print("This shows:")
    print("• Dependency hell is real with AI frameworks")
    print("• Silent conflicts break things without clear errors") 
    print("• Framework interoperability is still fragile")
    print("• The ecosystem is complex and interdependent")

if __name__ == "__main__":
    main()
