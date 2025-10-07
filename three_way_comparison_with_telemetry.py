# three_way_comparison_with_telemetry.py
import os
import time
import requests
import pandas as pd
from io import StringIO
from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import List, Dict
from datetime import datetime

load_dotenv()

# Configuration
DEFAULT_CSV = os.getenv("CSV_URL", "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = os.getenv("MODEL", "models/gemini-2.0-flash")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
VERBOSE = os.getenv("VERBOSE", "False").lower() == "true"

# Telemetry Data Classes
@dataclass
class LLMCall:
    timestamp: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    caller: str

@dataclass
class FrameworkTelemetry:
    framework: str
    total_duration_ms: float
    llm_calls: List[LLMCall] = field(default_factory=list)
    steps: List[Dict] = field(default_factory=list)
    
    @property
    def total_llm_calls(self):
        return len(self.llm_calls)
    
    @property
    def total_tokens(self):
        return sum(c.input_tokens + c.output_tokens for c in self.llm_calls)
    
    @property
    def total_llm_time_ms(self):
        return sum(c.latency_ms for c in self.llm_calls)

# Simple token counter (rough estimate)
def count_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token"""
    return len(text) // 4

def fetch_data():
    """Same data fetching for all"""
    resp = requests.get(DEFAULT_CSV)
    return resp.text

def analyze_data(csv_text):
    """Same analysis for all"""
    df = pd.read_csv(StringIO(csv_text))
    return f"Dataset: {len(df)} rows, {len(df.columns)} columns: {', '.join(df.columns)}"

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
        
        try:
            model = genai.GenerativeModel(MODEL_NAME)
            model.generate_content("Test")
            print(f"   ✅ Using model: {MODEL_NAME}")
            return model, MODEL_NAME
        except Exception:
            fallback_models = ["models/gemini-2.0-flash", "models/gemini-1.5-flash"]
            for fallback_model in fallback_models:
                try:
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
    if model_name.startswith("models/"):
        base_model = model_name.replace("models/", "")
        return f"gemini/{base_model}"
    if "/" in model_name:
        return model_name
    return f"gemini/{model_name}"

def get_crewai_fallback_models():
    """Get fallback models in CrewAI format"""
    fallback_env = os.getenv("CREWAI_FALLBACK_MODELS", "").strip()
    if fallback_env:
        return [m.strip() for m in fallback_env.split(",")]
    defaults = ["models/gemini-1.5-flash", "models/gemini-1.5-pro"]
    return [convert_model_for_crewai(m) for m in defaults]

def test_autogen_style():
    """AutoGen-style with telemetry"""
    print("🧪 AUTOGEN-STYLE TEST")
    print("=" * 50)
    
    telemetry = FrameworkTelemetry(framework="AutoGen", total_duration_ms=0)
    start_time = time.time()
    
    if not GOOGLE_API_KEY:
        return "❌ API key missing", "None", telemetry
    
    try:
        model, used_model = get_working_model()
        print(f"   🤖 Model: {used_model}")
        print("   👨‍💼 Project Manager → 👩‍🔬 Data Analyst (Conversational)")
        
        # Step 1: Fetch data
        step_start = time.time()
        csv_text = fetch_data()
        telemetry.steps.append({
            "step": "fetch_data",
            "duration_ms": (time.time() - step_start) * 1000,
            "llm_call": False
        })
        
        # Step 2: Analyze data
        step_start = time.time()
        data_summary = analyze_data(csv_text)
        telemetry.steps.append({
            "step": "analyze_data", 
            "duration_ms": (time.time() - step_start) * 1000,
            "llm_call": False
        })
        
        # Step 3: Generate response (LLM call)
        step_start = time.time()
        prompt = IDENTICAL_PROMPT.format(data_summary=data_summary)
        response = model.generate_content(prompt)
        llm_duration = (time.time() - step_start) * 1000
        
        # Record LLM call
        telemetry.llm_calls.append(LLMCall(
            timestamp=datetime.now().isoformat(),
            model=used_model,
            input_tokens=count_tokens(prompt),
            output_tokens=count_tokens(response.text),
            latency_ms=llm_duration,
            caller="DataAnalyst"
        ))
        
        telemetry.steps.append({
            "step": "generate_email",
            "duration_ms": llm_duration,
            "llm_call": True
        })
        
        telemetry.total_duration_ms = (time.time() - start_time) * 1000
        return response.text, used_model, telemetry
        
    except Exception as e:
        telemetry.total_duration_ms = (time.time() - start_time) * 1000
        return f"AutoGen error: {e}", "None", telemetry

def test_langgraph_style():
    """LangGraph-style with telemetry"""
    print("🧪 LANGGRAPH-STYLE TEST")
    print("=" * 50)
    
    telemetry = FrameworkTelemetry(framework="LangGraph", total_duration_ms=0)
    start_time = time.time()
    
    if not GOOGLE_API_KEY:
        return "❌ API key missing", "None", telemetry
    
    try:
        model, used_model = get_working_model()
        print(f"   🤖 Model: {used_model}")
        print("   📥 Fetch → 🔍 Analyze → 👨‍💼 PM → 👩‍🔬 DA (Workflow)")
        
        # Node 1: Fetch
        step_start = time.time()
        csv_text = fetch_data()
        telemetry.steps.append({
            "node": "fetch_data",
            "duration_ms": (time.time() - step_start) * 1000,
            "llm_call": False,
            "state_keys": ["csv_text"]
        })
        
        # Node 2: Analyze
        step_start = time.time()
        data_summary = analyze_data(csv_text)
        telemetry.steps.append({
            "node": "analyze_data",
            "duration_ms": (time.time() - step_start) * 1000,
            "llm_call": False,
            "state_keys": ["data_summary"]
        })
        
        # Node 3: Generate (LLM call)
        step_start = time.time()
        prompt = IDENTICAL_PROMPT.format(data_summary=data_summary)
        response = model.generate_content(prompt)
        llm_duration = (time.time() - step_start) * 1000
        
        telemetry.llm_calls.append(LLMCall(
            timestamp=datetime.now().isoformat(),
            model=used_model,
            input_tokens=count_tokens(prompt),
            output_tokens=count_tokens(response.text),
            latency_ms=llm_duration,
            caller="analyst_node"
        ))
        
        telemetry.steps.append({
            "node": "generate_email",
            "duration_ms": llm_duration,
            "llm_call": True,
            "state_keys": ["email_output"]
        })
        
        telemetry.total_duration_ms = (time.time() - start_time) * 1000
        return response.text, used_model, telemetry
        
    except Exception as e:
        telemetry.total_duration_ms = (time.time() - start_time) * 1000
        return f"LangGraph error: {e}", "None", telemetry

def test_crewai_style():
    """CrewAI-style with telemetry"""
    print("🧪 CREWAI-STYLE TEST")
    print("=" * 50)
    
    telemetry = FrameworkTelemetry(framework="CrewAI", total_duration_ms=0)
    start_time = time.time()
    
    if not GOOGLE_API_KEY:
        return "❌ API key missing", "None", telemetry
    
    try:
        from crewai import Agent, Task, Crew, Process, LLM
        
        print("   👥 Multi-agent Crew (Role-based Collaboration)")
        
        crewai_model = convert_model_for_crewai(MODEL_NAME)
        
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
        
        project_manager = Agent(
            role='Project Manager',
            goal='Define clear requirements and ensure business alignment',
            backstory="""You are an experienced project manager who specializes in 
            data analytics projects.""",
            llm=llm,
            verbose=VERBOSE
        )
        
        data_analyst = Agent(
            role='Data Analyst',
            goal='Analyze data and generate insightful stakeholder communications',
            backstory="""You are a senior data analyst with expertise in extracting 
            meaningful insights from complex datasets.""",
            llm=llm,
            verbose=VERBOSE
        )
        
        csv_text = fetch_data()
        data_summary = analyze_data(csv_text)
        
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
        
        analytics_crew = Crew(
            agents=[project_manager, data_analyst],
            tasks=[define_requirements, create_stakeholder_email],
            process=Process.sequential,
            verbose=VERBOSE
        )
        
        # CrewAI execution (likely multiple LLM calls internally)
        crew_start = time.time()
        result = analytics_crew.kickoff()
        crew_duration = (time.time() - crew_start) * 1000
        
        # Estimate LLM calls (CrewAI makes multiple calls internally)
        # Typically: 2-3 calls per agent per task
        telemetry.steps.append({
            "task": "ProjectManager - Define requirements",
            "duration_ms": crew_duration * 0.4,  # Rough estimate
            "llm_calls_estimated": 2
        })
        
        telemetry.steps.append({
            "task": "DataAnalyst - Create email",
            "duration_ms": crew_duration * 0.6,  # Rough estimate  
            "llm_calls_estimated": 3
        })
        
        # Note: CrewAI doesn't expose individual LLM call metrics easily
        # so we estimate based on typical patterns
        print("   ℹ️  Note: CrewAI makes multiple internal LLM calls not shown here")
        print("   ℹ️  Check the CrewAI trace URL above for detailed breakdown")
        
        telemetry.total_duration_ms = (time.time() - start_time) * 1000
        return str(result), used_model, telemetry
        
    except ImportError:
        return "❌ CrewAI not installed", "None", telemetry
    except Exception as e:
        telemetry.total_duration_ms = (time.time() - start_time) * 1000
        return f"CrewAI error: {e}", "None", telemetry

def print_telemetry_comparison(telemetries: Dict[str, FrameworkTelemetry]):
    """Print beautiful telemetry comparison"""
    print("\n" + "=" * 80)
    print("📊 TELEMETRY COMPARISON")
    print("=" * 80)
    
    for name, telem in telemetries.items():
        print(f"\n🔍 {telem.framework.upper()}")
        print("-" * 40)
        print(f"   Total Duration:    {telem.total_duration_ms:>8.0f}ms")
        print(f"   LLM Calls:         {telem.total_llm_calls:>8}")
        print(f"   Total Tokens:      {telem.total_tokens:>8}")
        print(f"   LLM Time:          {telem.total_llm_time_ms:>8.0f}ms")
        print(f"   Non-LLM Time:      {telem.total_duration_ms - telem.total_llm_time_ms:>8.0f}ms")
        
        if telem.llm_calls:
            print(f"\n   LLM Call Breakdown:")
            for i, call in enumerate(telem.llm_calls, 1):
                print(f"      {i}. {call.caller}")
                print(f"         Input:  {call.input_tokens:>5} tokens")
                print(f"         Output: {call.output_tokens:>5} tokens")
                print(f"         Time:   {call.latency_ms:>5.0f}ms")
        
        print(f"\n   Execution Steps:")
        for i, step in enumerate(telem.steps, 1):
            step_name = step.get('step') or step.get('node') or step.get('task', 'Unknown')
            duration = step.get('duration_ms', 0)
            is_llm = step.get('llm_call', False)
            marker = "🤖" if is_llm else "⚙️ "
            print(f"      {i}. {marker} {step_name}: {duration:.0f}ms")
    
    # Comparison summary
    print("\n" + "=" * 80)
    print("💡 KEY DIFFERENCES")
    print("=" * 80)
    
    for name, telem in telemetries.items():
        overhead_pct = ((telem.total_duration_ms - telem.total_llm_time_ms) / 
                       telem.total_duration_ms * 100) if telem.total_duration_ms > 0 else 0
        print(f"{telem.framework:<12}: {telem.total_llm_calls} LLM calls, "
              f"{overhead_pct:.1f}% framework overhead")

def main():
    print("🔬 THREE-WAY FRAMEWORK COMPARISON WITH TELEMETRY")
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
    telemetries = {}
    
    # Test all three frameworks
    results['autogen'], models_used['autogen'], telemetries['autogen'] = test_autogen_style()
    print(f"\n📧 AUTOGEN RESULT ({len(results['autogen'].split())} words):")
    print("-" * 40)
    print(results['autogen'][:200] + "..." if len(results['autogen']) > 200 else results['autogen'])
    
    print("\n" + "=" * 80)
    
    results['langgraph'], models_used['langgraph'], telemetries['langgraph'] = test_langgraph_style()
    print(f"\n📧 LANGGRAPH RESULT ({len(results['langgraph'].split())} words):")
    print("-" * 40)
    print(results['langgraph'][:200] + "..." if len(results['langgraph']) > 200 else results['langgraph'])
    
    print("\n" + "=" * 80)
    
    results['crewai'], models_used['crewai'], telemetries['crewai'] = test_crewai_style()
    print(f"\n📧 CREWAI RESULT ({len(results['crewai'].split())} words):")
    print("-" * 40)
    print(results['crewai'][:200] + "..." if len(results['crewai']) > 200 else results['crewai'])
    
    # Print telemetry comparison
    print_telemetry_comparison(telemetries)
    
    print("\n" + "=" * 80)
    print("🎯 FRAMEWORK COMPARISON ANALYSIS")
    print("=" * 80)
    
    for framework, result in results.items():
        word_count = len(result.split())
        model = models_used[framework]
        print(f"{framework.upper():<10}: {word_count:>3} words | Model: {model}")
        print(f"           Preview: {result[:60]}...")

if __name__ == "__main__":
    main()
