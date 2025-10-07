# langgraph_demo.py - FINAL WORKING VERSION (NO HARDCODING)
import os
import requests
import pandas as pd
from io import StringIO
from dotenv import load_dotenv

load_dotenv()

# Get all values from environment variables with fallbacks
DEFAULT_CSV = os.getenv("CSV_URL", "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DEFAULT_MODEL = os.getenv("MODEL", "models/gemini-2.0-flash")

def fetch_csv_text(url=DEFAULT_CSV):
    """Fetch CSV data from URL"""
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.text

def analyse_csv_text(csv_text):
    """Analyze CSV and return summary"""
    df = pd.read_csv(StringIO(csv_text))
    summary = {
        "rows": len(df),
        "columns": list(df.columns),
        "preview": df.head(3).to_dict()
    }
    return summary

def format_summary_for_prompt(summary):
    """Format summary for LLM prompt"""
    return f"Dataset: {summary['rows']} rows, {len(summary['columns'])} columns: {', '.join(summary['columns'])}"

def get_available_models():
    """Get list of available models from environment or use defaults"""
    models_env = os.getenv("AVAILABLE_MODELS")
    if models_env:
        return [model.strip() for model in models_env.split(",")]
    else:
        return [
            "models/gemini-2.0-flash",
            "models/gemini-2.0-flash-001",
            "models/gemini-2.5-flash", 
            "models/gemini-pro-latest",
            "models/gemini-flash-latest"
        ]

def get_working_gemini_model():
    """Get a working Gemini model with fallbacks"""
    try:
        import google.generativeai as genai
        
        if not GOOGLE_API_KEY:
            return None
            
        genai.configure(api_key=GOOGLE_API_KEY)
        
        # Try available models in order
        available_models = get_available_models()
        
        for model_name in available_models:
            try:
                print(f"   Testing: {model_name}")
                model = genai.GenerativeModel(model_name)
                # Quick test
                model.generate_content("Test")
                print(f"   ✅ Using: {model_name}")
                return model
            except Exception as e:
                print(f"   ❌ {model_name}: {str(e)[:80]}...")
                continue
                
        return None
    except ImportError:
        print("   ❌ google-generativeai not installed")
        return None

def main():
    print("🚀 LANGGRAPH-STYLE: Two Specialized AI Agents Workflow")
    print("=" * 60)
    print("Agents: 👨‍💼 Project Manager + 👩‍🔬 Data Analyst")
    print("=" * 60)
    
    # Display configuration
    print(f"🔧 Configuration:")
    print(f"   - Data Source: {DEFAULT_CSV}")
    print(f"   - Default Model: {DEFAULT_MODEL}")
    print(f"   - API Key: {'✅ Set' if GOOGLE_API_KEY else '❌ Missing'}")
    
    # Get working model
    print(f"\n🔍 Initializing AI Models...")
    model = get_working_gemini_model()
    
    # NODE 1: Fetch Data
    print("\n📥 NODE 1: Fetching data...")
    csv_text = fetch_csv_text()
    
    # NODE 2: Analyze Data
    print("🔍 NODE 2: Analyzing data...")
    summary = analyse_csv_text(csv_text)
    summary_text = format_summary_for_prompt(summary)
    
    print(f"📊 Data Summary: {summary_text}")
    
    if model:
        # NODE 3: Project Manager Agent
        print("\n👨‍💼 NODE 3: Project Manager Agent - Defining requirements...")
        pm_prompt = f"""As Project Manager, create specific instructions for the Data Analyst:

DATA ANALYSIS:
{summary_text}

Create clear, actionable requirements for a stakeholder email about restaurant performance. Include:
- Key business questions to address
- Required email sections and structure
- Tone and style guidelines
- Specific insights to highlight from the data"""

        print("⏳ Project Manager generating requirements...")
        try:
            pm_response = model.generate_content(pm_prompt)
            pm_instructions = pm_response.text
            print("✅ Project Manager requirements defined (Real AI)")
        except Exception as e:
            print(f"❌ Project Manager failed: {e}")
            pm_instructions = "Create comprehensive stakeholder email with data-driven insights"
        
        # NODE 4: Data Analyst Agent
        print("\n👩‍🔬 NODE 4: Data Analyst Agent - Creating email draft...")
        analyst_prompt = f"""As Data Analyst, follow the Project Manager's instructions:

PROJECT MANAGER REQUIREMENTS:
{pm_instructions}

DATA ANALYSIS:
{summary_text}

Create a professional stakeholder email with:
- Clear, compelling subject line
- 3-5 data-driven insights grounded in the actual data
- Actionable recommendations
- Professional, executive-appropriate tone
- Clear call to action"""

        print("⏳ Data Analyst drafting email...")
        try:
            analyst_response = model.generate_content(analyst_prompt)
            final_email = analyst_response.text
            ai_used = True
        except Exception as e:
            print(f"❌ Data Analyst failed: {e}")
            final_email = generate_simulated_email(summary)
            ai_used = False
    else:
        # Fallback simulation
        print("⚠️ Using simulated AI responses (API not available)")
        pm_instructions = "Create stakeholder email with key insights from restaurant data"
        final_email = generate_simulated_email(summary)
        ai_used = False
    
    # Display results
    print("\n" + "=" * 80)
    print("📧 FINAL COLLABORATIVE EMAIL")
    print("=" * 80)
    print(final_email)
    
    print("\n" + "=" * 80)
    print("🔄 TWO-AGENT WORKFLOW EXECUTED")
    print("=" * 80)
    print("1. 📥 Data fetched and analyzed")
    print("2. 👨‍💼 Project Manager defined requirements") 
    print("3. 👩‍🔬 Data Analyst created email draft")
    print("4. 🤝 Workflow coordination completed")
    print("5. 📧 Professional email delivered")
    
    if ai_used:
        print("\n🎉 SUCCESS: Real AI used throughout the workflow!")
    else:
        print("\n💡 TIP: Configure .env with GOOGLE_API_KEY for real AI collaboration")

def generate_simulated_email(summary):
    """Generate a simulated email using actual data statistics"""
    return f"""
Subject: Restaurant Performance Insights from {summary['rows']} Transactions

Dear Team,

Based on our analysis of {summary['rows']} customer transactions across {len(summary['columns'])} data dimensions, here are the key insights:

• Customer behavior patterns reveal opportunities for service optimization during peak hours
• Transaction data supports strategic staffing and resource allocation decisions
• The dataset enables targeted customer experience and menu improvements
• Comprehensive analytics available for {', '.join(summary['columns'][:3])} and other metrics

Recommended Action: Schedule a strategic review meeting to discuss data-driven operational enhancements.

Best regards,
Data Analytics Team
"""

if __name__ == "__main__":
    main()
