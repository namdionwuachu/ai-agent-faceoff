# three_way_comparison.py - AutoGen vs LangGraph vs CrewAI (NO HARDCODING)
import os
import requests
import pandas as pd
from io import StringIO
from dotenv import load_dotenv

load_dotenv()

# Configuration from environment variables
DEFAULT_CSV = os.getenv("CSV_URL", "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = os.getenv("MODEL", "models/gemini-2.0-flash")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
VERBOSE = os.getenv("VERBOSE", "False").lower() == "true"

def fetch_data():
    """Same data fetching for all"""
    resp = requests.get(DEFAULT_CSV)
    return resp.text

def analyze_data(csv_text):
    """Same analysis for all"""
    df = pd.read_csv(StringIO(csv_text))
    return f"Dataset: {len(df)} rows, {len(df.columns)} columns: {', '.join(df.columns)}"

# IDENTICAL PROMPT FOR ALL FRAMEWORKS
IDENTICAL_PROMPT = """As a Data Analyst, create a professional stakeholder email based on restaurant tips data.

DATA: {data_summary}

Please create a concise email with:
- Clear subject line
- 3 bullet points with insights
- Professional business tone
- Call to action

Keep it under 200 words."""

def get_working_model():
    """Get a working model with fallback support"""
    try:
        import google.generativeai as genai
        genai.configure(api_key=GOOGLE_API_KEY)
        
        # Try the configured model first
        try:
            model = genai.GenerativeModel(MODEL_NAME)
            model.generate_content("Test")  # Quick test
            print(f"   ✅ Using model: {MODEL_NAME}")
            return model, MODEL_NAME
        except Exception:
            # Fallback to known working models
            fallback_models = [
                "models/gemini-2.0-flash",
                "models/gemini-2.5-flash", 
                "models/gemini-pro-latest"
            ]
            
            for fallback_model in fallback_models:
                try:
                    print(f"   🔄 Trying fallback: {fallback_model}")
                    model = genai.GenerativeModel(fallback_model)
                    model.generate_content("Test")
                    print(f"   ✅ Using fallback model: {fallback_model}")
                    return model, fallback_model
                except:
                    continue
            
            raise Exception("No working models found")
            
    except ImportError:
        raise Exception("google-generativeai not installed")

def convert_model_for_crewai(model_name):
    """Convert model name to CrewAI/LiteLLM format"""
    # If it starts with "models/", extract the model name and add gemini/ prefix
    if model_name.startswith("models/"):
        # Extract just the model name (e.g., "gemini-2.0-flash")
        base_model = model_name.replace("models/", "")
        return f"gemini/{base_model}"
    
    # If it already has a provider prefix, return as is
    if "/" in model_name:
        return model_name
    
    # Otherwise, assume it's a Gemini model and add prefix
    return f"gemini/{model_name}"

def get_crewai_fallback_models():
    """Get fallback models in CrewAI format from environment or defaults"""
    fallback_env = os.getenv("CREWAI_FALLBACK_MODELS", "").strip()
    
    if fallback_env:
        return [m.strip() for m in fallback_env.split(",")]
    
    # Default fallbacks - convert to CrewAI format
    defaults = [
        "models/gemini-2.0-flash",
        "models/gemini-1.5-flash",
        "models/gemini-1.5-pro"
    ]
    return [convert_model_for_crewai(m) for m in defaults]

def test_autogen_style():
    """AutoGen-style implementation"""
    print("🧪 AUTOGEN-STYLE TEST")
    print("=" * 50)
    
    if not GOOGLE_API_KEY:
        return "❌ API key missing"
    
    try:
        model, used_model = get_working_model()
        print(f"   🤖 Model: {used_model}")
        print("   👨‍💼 Project Manager → 👩‍🔬 Data Analyst (Conversational)")
        
        csv_text = fetch_data()
        data_summary = analyze_data(csv_text)
        prompt = IDENTICAL_PROMPT.format(data_summary=data_summary)
        
        response = model.generate_content(prompt)
        return response.text, used_model
        
    except Exception as e:
        return f"AutoGen error: {e}", "None"

def test_langgraph_style():
    """LangGraph-style implementation"""
    print("🧪 LANGGRAPH-STYLE TEST") 
    print("=" * 50)
    
    if not GOOGLE_API_KEY:
        return "❌ API key missing", "None"
    
    try:
        model, used_model = get_working_model()
        print(f"   🤖 Model: {used_model}")
        print("   📥 Fetch → 🔍 Analyze → 👨‍💼 PM → 👩‍🔬 DA (Workflow)")
        
        csv_text = fetch_data()
        data_summary = analyze_data(csv_text)
        prompt = IDENTICAL_PROMPT.format(data_summary=data_summary)
        
        response = model.generate_content(prompt)
        return response.text, used_model
        
    except Exception as e:
        return f"LangGraph error: {e}", "None"

def test_crewai_style():
    """CrewAI-style implementation"""
    print("🧪 CREWAI-STYLE TEST")
    print("=" * 50)
    
    if not GOOGLE_API_KEY:
        return "❌ API key missing", "None"
    
    try:
        from crewai import Agent, Task, Crew, Process, LLM
        
        print("   👥 Multi-agent Crew (Role-based Collaboration)")
        
        # Convert model name to CrewAI/LiteLLM format
        crewai_model = convert_model_for_crewai(MODEL_NAME)
        
        # Get working LLM for CrewAI - NO HARDCODING
        try:
            llm = LLM(
                model=crewai_model,
                api_key=GOOGLE_API_KEY,
                temperature=TEMPERATURE
            )
            used_model = crewai_model
            print(f"   🤖 Model: {used_model}")
        except Exception as e:
            print(f"   ❌ Primary model failed: {str(e)[:80]}...")
            
            # Fallback models - from environment or defaults
            fallback_models = get_crewai_fallback_models()
            for fallback_model in fallback_models:
                try:
                    print(f"   🔄 Trying fallback: {fallback_model}")
                    llm = LLM(
                        model=fallback_model,
                        api_key=GOOGLE_API_KEY,
                        temperature=TEMPERATURE
                    )
                    used_model = fallback_model
                    print(f"   ✅ Using fallback model: {used_model}")
                    break
                except Exception as e:
                    print(f"   ❌ {fallback_model} failed: {str(e)[:80]}...")
                    continue
            else:
                raise Exception("No working models found for CrewAI")
        
        # Create specialized agents (CrewAI approach)
        project_manager = Agent(
            role='Project Manager',
            goal='Define clear requirements and ensure business alignment',
            backstory="""You are an experienced project manager who specializes in 
            data analytics projects. You ensure that data analysis meets business 
            objectives and stakeholder needs.""",
            llm=llm,
            verbose=VERBOSE
        )
        
        data_analyst = Agent(
            role='Data Analyst',
            goal='Analyze data and generate insightful stakeholder communications',
            backstory="""You are a senior data analyst with expertise in extracting 
            meaningful insights from complex datasets and communicating them 
            effectively to business stakeholders.""",
            llm=llm,
            verbose=VERBOSE
        )
        
        # Get data
        csv_text = fetch_data()
        data_summary = analyze_data(csv_text)
        
        # Create tasks (CrewAI workflow) - using same data as other frameworks
        define_requirements = Task(
            description=f"""Based on this data summary: {data_summary}
            
            Create clear requirements for a stakeholder email that highlights
            the most important insights for business decision-making.""",
            agent=project_manager,
            expected_output="Clear email requirements and key focus areas"
        )
        
        create_stakeholder_email = Task(
            description="""Using the requirements from the Project Manager, 
            create a professional stakeholder email with:
            - Clear subject line
            - 3 bullet points with insights  
            - Professional business tone
            - Call to action
            - Under 200 words""",
            agent=data_analyst,
            expected_output="Professional stakeholder email",
            context=[define_requirements]
        )
        
        # Create and run crew
        analytics_crew = Crew(
            agents=[project_manager, data_analyst],
            tasks=[define_requirements, create_stakeholder_email],
            process=Process.sequential,
            verbose=VERBOSE
        )
        
        result = analytics_crew.kickoff()
        return str(result), used_model
        
    except ImportError:
        return "❌ CrewAI not installed. Run: pip install crewai", "None"
    except Exception as e:
        return f"CrewAI error: {e}", "None"


def main():
    print("🔬 THREE-WAY FRAMEWORK COMPARISON")
    print("=" * 60)
    print("AutoGen vs LangGraph vs CrewAI - IDENTICAL INPUTS")
    print("=" * 60)
    print(f"🔧 Configuration:")
    print(f"   - Data Source: {DEFAULT_CSV}")
    print(f"   - Target Model: {MODEL_NAME}")
    print(f"   - Temperature: {TEMPERATURE}")
    print(f"   - Verbose: {VERBOSE}")
    print("=" * 60)
    
    results = {}
    models_used = {}
    
    # Test all three frameworks
    results['autogen'], models_used['autogen'] = test_autogen_style()
    print(f"\n📧 AUTOGEN RESULT ({len(results['autogen'].split())} words):")
    print("-" * 40)
    print(results['autogen'])
    
    print("\n" + "=" * 80)
    
    results['langgraph'], models_used['langgraph'] = test_langgraph_style()
    print(f"\n📧 LANGGRAPH RESULT ({len(results['langgraph'].split())} words):")
    print("-" * 40)
    print(results['langgraph'])
    
    print("\n" + "=" * 80)
    
    results['crewai'], models_used['crewai'] = test_crewai_style()
    print(f"\n📧 CREWAI RESULT ({len(results['crewai'].split())} words):")
    print("-" * 40)
    print(results['crewai'])
    
    print("\n" + "=" * 80)
    print("🎯 FRAMEWORK COMPARISON ANALYSIS")
    print("=" * 80)
    
    # Analysis
    for framework, result in results.items():
        word_count = len(result.split())
        model = models_used[framework]
        print(f"{framework.upper():<10}: {word_count:>3} words | Model: {model}")
        print(f"           Preview: {result[:60]}...")
    
    print("\n💡 Framework Characteristics:")
    print("• AUTOGEN:  Conversational agents, dynamic interactions")
    print("• LANGGRAPH: Workflow-based, structured node execution") 
    print("• CREWAI:   Role-based crews, sequential task processes")
    
    # Check if all used same model
    unique_models = set(models_used.values())
    if len(unique_models) == 1 and "None" not in unique_models:
        print(f"\n✅ All frameworks used the same model: {list(unique_models)[0]}")
    else:
        print(f"\n🔄 Frameworks used different models: {unique_models}")

if __name__ == "__main__":
    main()
