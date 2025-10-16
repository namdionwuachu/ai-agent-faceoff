# ================================================================================
# THREE-WAY FRAMEWORK COMPARISON: SAME PROMPT • SAME DATA • SAME MODEL
# ================================================================================
# Filename: three_way_framework_comparison.py
# Purpose:
#   Conduct a controlled comparison of AutoGen, LangGraph, and CrewAI by enforcing
#   identical data inputs, the same prompt text, and the exact same Gemini model.
#
# Summary:
#   • All frameworks receive the same restaurant "tips" dataset from Seaborn.
#   • The identical analytical prompt is passed verbatim to each framework.
#   • The same Gemini model (e.g., models/gemini-2.0-flash) is used across all runs.
#   • Outputs and telemetry (duration, tokens, call counts) are compared for parity.
#
# Outcome:
#   Produces a fair, one-to-one benchmark showing how each framework processes
#   identical LLM tasks under identical conditions.
# ================================================================================

import os
import time
import requests
import pandas as pd
import re
from io import StringIO
from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import List, Dict, TypedDict, Tuple, Set
from datetime import datetime

# =======================
# Unified ENV + Config (Gemini + CrewAI)
# =======================
load_dotenv()

# --- Required ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()

# --- Optional ---
CSV_URL = os.getenv(
    "CSV_URL",
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
).strip()

MODEL_ENV = os.getenv("MODEL", "models/gemini-2.0-flash").strip()

# --- Optional fallback models (from .env or defaults) ---
AVAILABLE_MODELS = [
    m.strip() for m in os.getenv(
        "AVAILABLE_MODELS",
        "models/gemini-2.0-flash,models/gemini-2.5-flash,models/gemini-pro-latest"
    ).split(",") if m.strip()
]

CREWAI_FALLBACK_MODELS = [
    m.strip() for m in os.getenv(
        "CREWAI_FALLBACK_MODELS",
        "gemini/gemini-1.5-flash,gemini/gemini-1.5-pro"
    ).split(",") if m.strip()
]

TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
VERBOSE = os.getenv("VERBOSE", "False").lower() == "true"
CREWAI_TRACING_ENABLED = os.getenv("CREWAI_TRACING_ENABLED", "false").lower() == "true"

# --- Model name conversions ---
def to_gemini_native(name: str) -> str:
    """Return native Gemini format (e.g., models/gemini-2.0-flash)."""
    if not name.startswith("models/"):
        return f"models/{name}"
    return name

def to_crewai_format(name: str) -> str:
    """Convert Gemini model name to CrewAI/LiteLLM format (e.g., gemini/gemini-2.0-flash)."""
    base = name.replace("models/", "")
    if "/" not in base:
        base = f"gemini/{base}"
    elif not base.startswith("gemini/"):
        base = f"gemini/{base}"
    return base

def to_openai_format(name: str) -> str:
    """Convert to OpenAI-compatible format (e.g., gemini-2.0-flash)."""
    return name.replace("models/", "")

# Unified accessors - DEFINE THESE CRITICAL VARIABLES
MODEL_NAME_GEMINI = to_gemini_native(MODEL_ENV)  # For google-generativeai
MODEL_NAME_CREWAI = to_crewai_format(MODEL_ENV)  # For CrewAI LLM()
MODEL_NAME_OPENAI = to_openai_format(MODEL_ENV)  # For OpenAI-compatible endpoints

# Use consistent naming throughout the code
MODEL_NAME = MODEL_NAME_GEMINI
DEFAULT_CSV = CSV_URL

# DEBUG: Check AutoGen import
print("=== DEBUG AUTOGen IMPORT ===")
try:
    import autogen
    from autogen import AssistantAgent, UserProxyAgent
    print(f"✅ AutoGen SUCCESS: {autogen.__version__}")
    print(f"✅ All components imported")
except ImportError as e:
    print(f"❌ Import failed: {e}")
print("=== END DEBUG ===")

# Rest of your existing functions continue below...

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

def count_tokens(text: str) -> int:
    return len(text) // 4

def fetch_data():
    resp = requests.get(CSV_URL)
    return resp.text

def analyze_data(csv_text):
    df = pd.read_csv(StringIO(csv_text))
    return f"Dataset: {len(df)} rows, {len(df.columns)} columns: {', '.join(df.columns)}"

def calculate_business_intelligence(csv_text):
    """Calculate actual business metrics for hybrid approach"""
    df = pd.read_csv(StringIO(csv_text))
    
    # Calculate comprehensive business intelligence
    daily_stats = df.groupby('day').agg({
        'total_bill': ['mean', 'sum', 'count'],
        'tip': ['mean', 'sum'],
        'size': 'mean'
    }).round(2)
    
    time_stats = df.groupby('time').agg({
        'total_bill': ['mean', 'sum'],
        'tip': ['mean', 'sum'],
        'size': 'mean'
    }).round(2)
    
    # Calculate actual percentages and business metrics
    df['tip_percentage'] = (df['tip'] / df['total_bill'] * 100)
    overall_tip_rate = df['tip_percentage'].mean()
    
    # Find best performing segments
    best_tip_day = df.groupby('day')['tip_percentage'].mean().idxmax()
    best_revenue_time = df.groupby('time')['total_bill'].sum().idxmax()
    
    return f"""
📊 CALCULATED BUSINESS INTELLIGENCE REPORT:

💰 FINANCIAL PERFORMANCE:
- Total Revenue: ${df['total_bill'].sum():.2f}
- Total Tips: ${df['tip'].sum():.2f}
- Overall Tip Rate: {overall_tip_rate:.1f}%
- Average Transaction: ${df['total_bill'].mean():.2f}

📅 DAILY BUSINESS INTELLIGENCE:
{chr(10).join([f"- {day}: ${daily_stats['total_bill']['mean'][day]:.2f} avg bill, {daily_stats['total_bill']['count'][day]} transactions, ${daily_stats['total_bill']['sum'][day]:.2f} total" for day in daily_stats.index])}

⏰ TIME-BASED PERFORMANCE:
{chr(10).join([f"- {time}: ${time_stats['total_bill']['mean'][time]:.2f} avg bill, ${time_stats['total_bill']['sum'][time]:.2f} total revenue" for time in time_stats.index])}

🎯 KEY BUSINESS INSIGHTS:
- Most Profitable Day: {best_tip_day} (highest tip percentage)
- Highest Revenue Time: {best_revenue_time}
- Customer Party Size: {df['size'].mean():.1f} average
- Busiest Day: {daily_stats['total_bill']['count'].idxmax()} ({daily_stats['total_bill']['count'].max()} transactions)

📈 PERFORMANCE METRICS:
- Weekend vs Weekday Revenue: ${df[df['day'].isin(['Sat','Sun'])]['total_bill'].sum():.2f} vs ${df[~df['day'].isin(['Sat','Sun'])]['total_bill'].sum():.2f}
- Dinner Premium: {((df[df['time']=='Dinner']['total_bill'].mean() - df[df['time']=='Lunch']['total_bill'].mean()) / df[df['time']=='Lunch']['total_bill'].mean() * 100):.1f}% higher bills
"""
# =========================
# SHARED PROMPT TEMPLATES
# =========================

ANALYST_CORE = """You are a senior data analyst. Analyze the PRE-CALCULATED METRICS below and produce exactly 3 actionable insights.
Rules:
- Use only numbers that appear verbatim in the BUSINESS INTELLIGENCE block below.
- Do NOT calculate, round, or derive new figures. For comparisons, write "$X vs $Y" or reuse a provided percentage.
- Output exactly 3 one-line bullets.
Inputs:
- BUSINESS INTELLIGENCE:
{BUSINESS_INTEL}
- OPTIONAL REQUIREMENTS (may be empty):
{PM_REQUIREMENTS}
Output:
- 3 bullet points (one line each), using concrete figures where relevant
"""

PM_REQUIREMENTS_CORE = """As Project Manager, define concise email requirements for executives.
Rules:
- Reference only the numbers already present in the BUSINESS INTELLIGENCE block.
- Do NOT ask anyone to compute or derive new numbers; reuse figures exactly as given.
Inputs:
- DATA SUMMARY: {DATA_SUMMARY}
- BUSINESS INTELLIGENCE: {BUSINESS_INTEL}
Requirements must ensure:
- 3 key insights with revenue impact, using ACTUAL metrics already provided (no new calculations)
- Actionable recommendations
- Executive tone and clear structure
- Final email under 200 words, plain text only
Output:
- A short, numbered list of specific requirements (5–7 items)
"""

EXEC_EMAIL_CORE = """Create a PLAIN TEXT executive email using the inputs below.
PROTOCOL:
- Return your final email body ONLY between the exact lines:
---BEGIN EMAIL---
...email body...
---END EMAIL---
- Do not add any text before or after those markers.
STRICT RULES:
- Use only numbers that appear verbatim in the BUSINESS INTELLIGENCE block.
- Do NOT calculate, round, or derive new figures.
- Keep the entire email under 200 words.
Inputs:
- ANALYSIS (3 insights): {ANALYSIS}
- BUSINESS INTELLIGENCE: {BUSINESS_INTEL}
- OPTIONAL REQUIREMENTS (may be empty): {PM_REQUIREMENTS}
Constraints:
- Subject line (business value)
- 3 bullets using ACTUAL metrics (no invented numbers)
- One-line call to action
- Entire email < 200 words
Return ONLY the email body (no markdown, no code fences).
"""

# =========================
# EMAIL PROCESSING FUNCTIONS
# =========================

def extract_email_body(text: str) -> str:
    """Extract email body from between markers, or return full text"""
    START, END = "---BEGIN EMAIL---", "---END EMAIL---"
    if START in text and END in text:
        return text.split(START, 1)[1].split(END, 1)[0].strip()
    return text.strip()

def _extract_numbers(text: str) -> List[str]:
    return re.findall(r"-?\d+(?:\.\d+)?", text)

def _build_allowed_numbers(bi_text: str) -> Set[str]:
    return set(_extract_numbers(bi_text))

def validate_email_numbers(business_intel_text: str, email_text: str) -> Tuple[bool, List[str]]:
    allowed = _build_allowed_numbers(business_intel_text)
    mentioned = _extract_numbers(email_text)
    bad = [n for n in mentioned if n not in allowed]
    return (len(bad) == 0), bad

# =========================
# FRAMEWORK HELPER FUNCTIONS  
# =========================

def extract_last_message(result, preferred_name: str | None = None) -> str:
    """Return the most recent assistant reply (works for AutoGen/LangGraph/CrewAI)."""
    hist = getattr(result, "chat_history", None)
    if not hist:
        return ""

    # Prefer assistant reply from the specific agent
    if preferred_name:
        for msg in reversed(hist):
            if msg.get("role") == "assistant" and (
                (msg.get("name") or msg.get("sender")) == preferred_name
            ):
                return msg.get("content") or msg.get("text") or ""

    # Otherwise take the most recent assistant reply
    for msg in reversed(hist):
        if msg.get("role") == "assistant":
            return msg.get("content") or msg.get("text") or ""

    # Fallback: last non-user message
    for msg in reversed(hist):
        name = msg.get("name") or msg.get("sender")
        if name and name not in ("User_Proxy", "user"):
            return msg.get("content") or msg.get("text") or ""

    return ""

def timed_chat(user_proxy, recipient, message, caller_label, telemetry):
    t0 = time.time()
    try:
        res = user_proxy.initiate_chat(
            recipient=recipient,
            message=message,
            clear_history=True,
            max_turns=1
        )
        latency = (time.time() - t0) * 1000
        
        # Extract assistant text reliably
        out_text = extract_last_message(res, recipient.name) or ""
        in_tokens = count_tokens(message)
        out_tokens = count_tokens(out_text)
        
        telemetry.llm_calls.append(LLMCall(
            timestamp=datetime.now().isoformat(),
            model=MODEL_NAME_GEMINI,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            latency_ms=latency,
            caller=caller_label
        ))
        return res
    except Exception as e:
        telemetry.llm_calls.append(LLMCall(
            timestamp=datetime.now().isoformat(),
            model=MODEL_NAME_GEMINI,
            input_tokens=count_tokens(message),
            output_tokens=0,
            latency_ms=(time.time() - t0) * 1000,
            caller=f"ERROR_{caller_label}"
        ))
        raise e

def test_real_autogen():
    """REAL AutoGen with actual multi-agent conversation"""
    print("🧪 REAL AUTOGEN TEST")
    print("=" * 50)
    
    import google.generativeai as genai # sanity import
    genai.configure(api_key=GOOGLE_API_KEY)
    _ = genai.GenerativeModel(MODEL_NAME_GEMINI).generate_content("ping")
    
    telemetry = FrameworkTelemetry(framework="AutoGen", total_duration_ms=0)
    start_time = time.time()
    
    if not GOOGLE_API_KEY:
        return "❌ Google API key missing", "None", telemetry
    
    try:
        import autogen
        from autogen import AssistantAgent, UserProxyAgent
        
        print("   🤖 Using REAL AutoGen Multi-Agent System")
        print("   👨‍💼 Data Analyst ↔ 👔 Project Manager ↔ 📧 Comms Specialist (Conversation)")  # ← ADDED EMOJIS
        
        # Step 1: Fetch data
        step_start = time.time()
        csv_text = fetch_data()
        
        # 🚀 HYBRID APPROACH: Pre-calculate business intelligence
        pre_calculated_metrics = calculate_business_intelligence(csv_text)
        data_summary = analyze_data(csv_text)
        
        telemetry.steps.append({
            "step": "fetch_analyze_data",
            "duration_ms": (time.time() - step_start) * 1000,
            "llm_call": False
        })
        
        # In test_real_autogen(), replace the config_list with:
        config_list = [
            {
                "api_type": "openai",
                "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
                "model": to_openai_format(MODEL_NAME_GEMINI),  # Use consistent naming
                "api_key": GOOGLE_API_KEY,
            }
        ]
        

        llm_config = {
            "config_list": config_list,
            "temperature": TEMPERATURE,
            "top_p": 1.0,
            "max_tokens": 1024,
            "timeout": 120,
        }

        
        print(f"   🔧 Config: Gemini 2.0 Flash, Temp: {TEMPERATURE}")
        print("MODEL_NAME_GEMINI:", MODEL_NAME_GEMINI)             # expect: models/gemini-2.0-flash (native) OR gemini-2.0-flash (compat)
        safe_cfg = {**llm_config, "config_list": [{**llm_config["config_list"][0], "api_key": "***"}]}
        print("AutoGen llm_config (masked):", safe_cfg)


        
        # REAL AUTOGEN AGENTS - Updated with hybrid approach
        data_analyst = AssistantAgent(
            name="Data_Analyst",
            system_message=ANALYST_CORE,   # <-- use shared
            llm_config=llm_config,
            max_consecutive_auto_reply=1
        )
        
        project_manager = AssistantAgent(
            name="Project_Manager",
            system_message=PM_REQUIREMENTS_CORE,  # <-- produces requirements (not the final email)
            llm_config=llm_config,
            max_consecutive_auto_reply=1
        )

        # NEW: Comms agent writes the final email
        comms_specialist = AssistantAgent(
            name="Comms_Specialist",
            system_message=EXEC_EMAIL_CORE,
            llm_config=llm_config,
            max_consecutive_auto_reply=1
        )
        
        user_proxy = UserProxyAgent(
            name="User_Proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=2,
            code_execution_config=False,
        )
        
        # REAL MULTI-AGENT CONVERSATION
        print("   🔄 Starting real agent conversation...")
        step_start = time.time()
        
    
        # 1) Analyst → insights
        analysis_result = timed_chat(
            user_proxy=user_proxy,
            recipient=data_analyst,
            message=ANALYST_CORE.format(
                BUSINESS_INTEL=pre_calculated_metrics,
                PM_REQUIREMENTS=""
            ),
            caller_label="AutoGen_Analyst",
            telemetry=telemetry
            
        )
        analysis_text = extract_last_message(analysis_result, "Data_Analyst") or ""
        print("\n=== ANALYST (3 bullets) ===\n", analysis_text)
        
        # 2) PM → requirements (uses shared template)
        pm_result = timed_chat(
            user_proxy=user_proxy,
            recipient=project_manager,
            message=PM_REQUIREMENTS_CORE.format(
                DATA_SUMMARY=data_summary,
                BUSINESS_INTEL=pre_calculated_metrics
            ),
            caller_label="AutoGen_PM",
            telemetry=telemetry
        )
        pm_requirements = extract_last_message(pm_result, "Project_Manager")
        print("\n=== PM REQUIREMENTS ===\n", pm_requirements)
        
        # 3) Comms → final email (uses shared template)
        email_result = timed_chat(
            user_proxy=user_proxy,
            recipient=comms_specialist,
            message=EXEC_EMAIL_CORE.format(
                ANALYSIS=analysis_text or "No analysis available",
                BUSINESS_INTEL=pre_calculated_metrics,
                PM_REQUIREMENTS=pm_requirements or ""
            ),
            caller_label="AutoGen_Comms",
            telemetry=telemetry
        )

        
        conversation_duration = (time.time() - step_start) * 1000

        # --- Extract Comms reply and slice only the final email block
        raw_reply = extract_last_message(email_result, "Comms_Specialist") or ""
        START, END = "---BEGIN EMAIL---", "---END EMAIL---"

        
        
        if START in raw_reply and END in raw_reply:
            email_body = raw_reply.split(START, 1)[1].split(END, 1)[0].strip()
        else:
            # Fallback if the model ignored markers: treat the whole reply as body
            email_body = raw_reply.strip()
            
        print("\n=== COMMS EMAIL (FINAL) ===\n", email_body)
        
        # ✅ Validate ONLY the email body (not the prompt you sent)
        ok, bad = validate_email_numbers(pre_calculated_metrics, email_body)
        if not ok:
            raise ValueError(f"AutoGen output used numbers not in dataset: {bad}")

        # Build final output using the body
        final_output = (
            "=== Analyst (3 bullets) ===\n" + analysis_text.strip() + "\n\n"
            "=== PM Requirements ===\n" + pm_requirements.strip() + "\n\n"
            "=== Final Email ===\n" + email_body + "\n"
        )

        # (Optional) include a tiny run stat for clarity
        if hasattr(email_result, 'chat_history') and email_result.chat_history:
            final_output += f"Total messages exchanged: {len(email_result.chat_history)}"
        else:
            final_output += "No chat history available."

        
        telemetry.steps.append({
            "step": "autogen_multi_agent_conversation",
            "duration_ms": conversation_duration,
            "llm_call": True,
            "agents": 3,  # Data_Analyst, Project_Manager, Comms_Specialist
            "conversation_turns": len(telemetry.llm_calls)  # 3, from timed_chat()
        })

        
        telemetry.total_duration_ms = (time.time() - start_time) * 1000
        
        return final_output, "gemini-2.0-flash", telemetry
        
    except ImportError as e:
        return f"❌ Missing dependency: {e}. Try: pip install pyautogen google-generativeai", "None", telemetry
    except Exception as e:
        print(f"DEBUG: AutoGen Error: {e}")
        telemetry.total_duration_ms = (time.time() - start_time) * 1000
        return f"❌ AutoGen runtime error: {e}", "None", telemetry

def test_real_langgraph():
    """REAL LangGraph with actual state machine and agent nodes"""
    print("🧪 REAL LANGGRAPH TEST")
    print("=" * 50)
    
    telemetry = FrameworkTelemetry(framework="LangGraph", total_duration_ms=0)
    start_time = time.time()
    
    if not GOOGLE_API_KEY:
        return "❌ API key missing", "None", telemetry
    
    try:
        # REAL LANGGRAPH IMPORTS
        from langgraph.graph import StateGraph
        import google.generativeai as genai
        
        print("   🤖 Using REAL LangGraph State Machine")
        print("   📥 Supervisor → 👨‍💼 PM → 👩‍🔬 DA → 🔍 Reviewer (Workflow)")
        
        # Setup Gemini for LangGraph nodes
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel(MODEL_NAME_GEMINI)
        
        # Step 1: Fetch data
        step_start = time.time()
        csv_text = fetch_data()
        
        # 🚀 HYBRID APPROACH: Pre-calculate business intelligence
        pre_calculated_metrics = calculate_business_intelligence(csv_text)
        data_summary = analyze_data(csv_text)
        
        telemetry.steps.append({
            "step": "fetch_analyze_data",
            "duration_ms": (time.time() - step_start) * 1000,
            "llm_call": False
        })
        
        # REAL LANGGRAPH STATE DEFINITION
        class AgentState(TypedDict, total=False):
            data_summary: str
            messages: list
            requirements: str
            analysis: str
            email_draft: str
            final_output: str
            next: str
            business_intelligence: str  # 🚀 ADD: Pre-calculated metrics
        
        # REAL LANGGRAPH NODES
        def supervisor_node(state: AgentState) -> AgentState:
            """LangGraph supervisor that routes workflow (no LLM calls here)"""
            # Routing logic only — deterministic, no model inference
            if "requirements" not in state or not state.get("requirements"):
                next_step = "project_manager"
                task = "Define email requirements based on data analysis"
            elif "analysis" not in state or not state.get("analysis"):
                next_step = "data_analyst"
                task = "Analyze data and create insights"
            elif "final_output" not in state or not state.get("final_output"):
                next_step = "reviewer"
                task = "Review and finalize email"
            else:
                next_step = "__end__"
                task = "Workflow complete"

            state["next"] = next_step
            state["messages"] = state.get("messages", []) + [
                {"role": "system", "content": f"Routing to {next_step}: {task}"}
            ]
            return state

        
        def project_manager_node(state: AgentState) -> AgentState:
            """LangGraph project manager node"""
            step_start = time.time()
            
            prompt = PM_REQUIREMENTS_CORE.format(
                DATA_SUMMARY=state['data_summary'],
                BUSINESS_INTEL=state['business_intelligence']
            )
            
            response = model.generate_content(prompt)
            llm_duration = (time.time() - step_start) * 1000
            
            telemetry.llm_calls.append(LLMCall(
                timestamp=datetime.now().isoformat(),
                model="gemini-2.0-flash",
                input_tokens=count_tokens(prompt),
                output_tokens=count_tokens(response.text),
                latency_ms=llm_duration,
                caller="ProjectManager_Node"
            ))
            
            state["requirements"] = response.text
            state["next"] = "supervisor"
            state["messages"] = state.get("messages", []) + [{"role": "assistant", "content": f"Requirements defined: {response.text[:100]}..."}]
            return state
        
        def data_analyst_node(state: AgentState) -> AgentState:
            """LangGraph data analyst node - USING PRE-CALCULATED METRICS"""
            step_start = time.time()
            
            prompt = ANALYST_CORE.format(
                BUSINESS_INTEL=state['business_intelligence'],
                PM_REQUIREMENTS=state.get('requirements', '')
            )
            
            response = model.generate_content(prompt)
            llm_duration = (time.time() - step_start) * 1000
            
            telemetry.llm_calls.append(LLMCall(
                timestamp=datetime.now().isoformat(),
                model="gemini-2.0-flash",
                input_tokens=count_tokens(prompt),
                output_tokens=count_tokens(response.text),
                latency_ms=llm_duration,
                caller="DataAnalyst_Node"
            ))
            
            state["analysis"] = response.text
            state["email_draft"] = response.text
            state["next"] = "supervisor"
            state["messages"] = state.get("messages", []) + [{"role": "assistant", "content": f"Analysis complete using real metrics"}]
            return state
        
        def reviewer_node(state: AgentState) -> AgentState:
            """LangGraph reviewer node"""
            step_start = time.time()
            
            prompt = EXEC_EMAIL_CORE.format(
                ANALYSIS=state['analysis'],
                BUSINESS_INTEL=state['business_intelligence'],
                PM_REQUIREMENTS=state.get('requirements', '')
            )
            response = model.generate_content(prompt)
            
            # ✅ Validate here using the node's state
            text = response.text
            ok, bad = validate_email_numbers(state['business_intelligence'], text)
            if not ok:
                raise ValueError(f"Reviewer node produced invalid numbers: {bad}")
       
            # telemetry logging
            llm_duration = (time.time() - step_start) * 1000
            telemetry.llm_calls.append(LLMCall(
                timestamp=datetime.now().isoformat(),
                model="gemini-2.0-flash",
                input_tokens=count_tokens(prompt),
                output_tokens=count_tokens(response.text),
                latency_ms=llm_duration,
                caller="Reviewer_Node"
            ))
            
            state["final_output"] = response.text
            state["next"] = "__end__"
            state["messages"] = state.get("messages", []) + [{"role": "assistant", "content": f"Final email: {response.text[:100]}..."}]
            return state
        
        # BUILD REAL LANGGRAPH WORKFLOW
        workflow = StateGraph(AgentState)
        
        # Add nodes to the graph
        workflow.add_node("supervisor", supervisor_node)
        workflow.add_node("project_manager", project_manager_node)
        workflow.add_node("data_analyst", data_analyst_node)
        workflow.add_node("reviewer", reviewer_node)
        
        # Define the graph edges
        workflow.set_entry_point("supervisor")
        
        # Routing function
        def route_next(state: AgentState) -> str:
            return state.get("next", "__end__")
        
        # Supervisor conditionally routes to next node
        workflow.add_conditional_edges(
            "supervisor",
            route_next,
            {
                "project_manager": "project_manager",
                "data_analyst": "data_analyst",
                "reviewer": "reviewer",
                "__end__": "__end__"
            }
        )
        
        # Other nodes return to supervisor for next routing decision
        workflow.add_edge("project_manager", "supervisor")
        workflow.add_edge("data_analyst", "supervisor")
        workflow.add_edge("reviewer", "supervisor")
        
        # COMPILE AND EXECUTE REAL LANGGRAPH
        step_start = time.time()
        app = workflow.compile()
        result = app.invoke({
            "data_summary": data_summary,
            "business_intelligence": pre_calculated_metrics,  # 🚀 ADD PRE-CALCULATED METRICS
            "messages": []
        })
        graph_duration = (time.time() - step_start) * 1000
        
        telemetry.steps.append({
            "step": "langgraph_workflow_execution",
            "duration_ms": graph_duration,
            "llm_call": True,
            "agents": 3,  # PM node, Analyst node, Reviewer node (Supervisor does no LLM call)
            "nodes_activated": 4,
            "state_transitions": len(result.get('messages', [])),
            "conversation_turns": len(telemetry.llm_calls)  # should be 3, matching the 3 model.generate_content calls
        })
        
        telemetry.total_duration_ms = (time.time() - start_time) * 1000
        
        final_output = f"LangGraph Workflow Complete\n\n"
        final_output += f"Final Email Output:\n{result.get('final_output', 'Workflow completed')}\n\n"
        final_output += f"Workflow Messages: {len(result.get('messages', []))}"
        
        

        
        return final_output, "gemini-2.0-flash", telemetry
        
    except ImportError:
        return "❌ LangGraph not installed. Run: pip install langgraph", "None", telemetry
    except Exception as e:
        telemetry.total_duration_ms = (time.time() - start_time) * 1000
        return f"LangGraph error: {e}", "None", telemetry

def test_real_crewai():
    """REAL CrewAI with actual role-based agents"""
    print("🧪 REAL CREWAI TEST")
    print("=" * 50)
    
    telemetry = FrameworkTelemetry(framework="CrewAI", total_duration_ms=0)
    start_time = time.time()
    
    if not GOOGLE_API_KEY:
        return "❌ API key missing", "None", telemetry
    
    try:
        # REAL CREWAI IMPORTS
        from crewai import Agent, Task, Crew, Process, LLM
        import os
        
        print("   👥 Using REAL CrewAI Multi-Agent System")
        print("   👔 PM → 👨‍💻 DA → 📧 Comms (Role-Based Tasks)")  # ← ADDED EMOJIS
        
        os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY
        
        # Step 1: Fetch data
        step_start = time.time()
        csv_text = fetch_data()
        pre_calculated_metrics = calculate_business_intelligence(csv_text)
        data_summary = analyze_data(csv_text)

        telemetry.steps.append({
            "step": "fetch_analyze_data",
            "duration_ms": (time.time() - step_start) * 1000,
            "llm_call": False
        })

        # LLM config - FIXED MODEL NAME
        try:
            llm = LLM(model=MODEL_NAME_CREWAI, api_key=GOOGLE_API_KEY, temperature=TEMPERATURE)
            used_model = "gemini-2.0-flash"
            print(f"   🤖 Model: {used_model}")
        except Exception as e:
            print(f"   ❌ Model setup failed: {e}")
            return f"CrewAI model error: {e}", "None", telemetry

        # --- Agents ---
        project_manager = Agent(
            role="Senior Project Manager",
            goal="Define clear business requirements and ensure stakeholder alignment for data analysis projects",
            backstory=(
                "Experienced PM in hospitality analytics. Translates metrics into executive-ready requirements "
                "and ensures deliverables meet standards."
            ),
            llm=llm,
            verbose=VERBOSE,
            allow_delegation=False
        )

        data_analyst = Agent(
            role="Senior Data Analyst",
            goal="Analyze pre-calculated metrics to extract 3 actionable insights using actual numbers",
            backstory=(
                "Senior analyst focused on turning KPIs into decisions. Disciplined about using only provided metrics."
            ),
            llm=llm,
            verbose=VERBOSE,
            allow_delegation=False
        )

        communications_specialist = Agent(
            role="Communications Specialist",
            goal="Craft a concise executive email (<200 words) with 3 bullets using actual metrics and a clear CTA",
            backstory=(
                "Exec comms expert who writes crisp, data-grounded updates for stakeholders."
            ),
            llm=llm,
            verbose=VERBOSE,
            allow_delegation=False
        )

        # --- Tasks ---
        define_requirements = Task(
            description=PM_REQUIREMENTS_CORE.format(
                DATA_SUMMARY=data_summary,
                BUSINESS_INTEL=pre_calculated_metrics
            ),
            agent=project_manager,
            expected_output="Numbered requirements list (5–7 items) for the exec email",
            async_execution=False
        )

        analyze_data_insights = Task(
            description=ANALYST_CORE.format(
                BUSINESS_INTEL=pre_calculated_metrics,
                PM_REQUIREMENTS="Use the requirements produced by the Project Manager task available in context."
            ),
            agent=data_analyst,
            expected_output="Exactly 3 one-line bullet insights using actual metrics",
            context=[define_requirements],
            async_execution=False
        )

        create_stakeholder_email = Task(
            description=EXEC_EMAIL_CORE.format(
                ANALYSIS="Use the 3 insights produced by the Data Analyst task available in context.",
                BUSINESS_INTEL=pre_calculated_metrics,
                PM_REQUIREMENTS="Use the requirements from the Project Manager task available in context."
            ),
            agent=communications_specialist,
            expected_output="Plain-text executive email with subject, 3 bullets, CTA (<200 words)",
            context=[define_requirements, analyze_data_insights],
            async_execution=False
        )

        # --- Crew and kickoff ---
        analytics_crew = Crew(
            agents=[project_manager, data_analyst, communications_specialist],
            tasks=[define_requirements, analyze_data_insights, create_stakeholder_email],
            process=Process.sequential,
            verbose=VERBOSE
        )

        crew_start = time.time()
        result = analytics_crew.kickoff()
        crew_duration = (time.time() - crew_start) * 1000

        # Optional: post-run validation to guarantee "actual data only"
        final_email_text = str(result)
        ok, bad = validate_email_numbers(pre_calculated_metrics, final_email_text)
        if not ok:
            raise ValueError(f"CrewAI email used numbers not in dataset: {bad}")

        # High-level step telemetry
        telemetry.steps.append({
            "step": "crewai_workflow_execution",
            "duration_ms": crew_duration,
            "llm_call": True,
            "agents": 3,
        })

        telemetry.total_duration_ms = (time.time() - start_time) * 1000
        return final_email_text, used_model, telemetry
        
    except ImportError:
        return "❌ CrewAI not installed. Run: pip install crewai", "None", telemetry
    except Exception as e:
        telemetry.total_duration_ms = (time.time() - start_time) * 1000
        return f"CrewAI error: {e}", "None", telemetry



def print_telemetry_comparison(telemetries: Dict[str, FrameworkTelemetry]):
    """Print telemetry comparison"""
    print("\n" + "=" * 80)
    print("📊 TELEMETRY COMPARISON - REAL FRAMEWORKS")
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
            step_name = step.get('step') or step.get('task', 'Unknown')
            duration = step.get('duration_ms', 0)
            is_llm = step.get('llm_call', False)
            marker = "🤖" if is_llm else "⚙️ "
            details = []
            
            if step.get('agents'): details.append(f"{step['agents']} agents")
            if step.get('estimated_conversation_turns'): details.append(f"{step['estimated_conversation_turns']} turns")
            if step.get('llm_calls_estimated'): details.append(f"~{step['llm_calls_estimated']} calls")
            
            detail_str = f" ({', '.join(details)})" if details else ""
            print(f"      {i}. {marker} {step_name}: {duration:.0f}ms{detail_str}")

def validate_environment():
    """Validate all required environment variables."""
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable is required")
    
    required_vars = {
        'GOOGLE_API_KEY': GOOGLE_API_KEY,
        'MODEL_NAME_GEMINI': MODEL_NAME_GEMINI,
        'MODEL_NAME_CREWAI': MODEL_NAME_CREWAI,
        'DEFAULT_CSV': DEFAULT_CSV,
    }
    
    missing = [name for name, value in required_vars.items() if not value]
    if missing:
        raise ValueError(f"Missing configuration: {', '.join(missing)}")
    
    print("✅ Environment validation passed")
    print(f"   - Model: {MODEL_NAME_GEMINI}")
    print(f"   - Data Source: {DEFAULT_CSV}")
    print(f"   - Temperature: {TEMPERATURE}")

def main():
    print("🔬 THREE-WAY FRAMEWORK COMPARISON - REAL FRAMEWORKS")
    print("=" * 60)
    print("AutoGen vs LangGraph vs CrewAI - ACTUAL FRAMEWORKS")
    print("=" * 60)
    
    try:
        validate_environment()  # Add this validation
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("Please check your .env file and ensure GOOGLE_API_KEY is set")
        return
    
    print(f"🔧 Configuration:")
    print(f"   - Data Source: {DEFAULT_CSV}")
    print(f"   - Target Model: {MODEL_NAME_GEMINI}")
    print(f"   - Temperature: {TEMPERATURE}")
    print(f"   - Verbose: {VERBOSE}")
    print("=" * 60)
    
    results = {}
    models_used = {}
    telemetries = {}
    
    # Test all three REAL frameworks
    results['autogen'], models_used['autogen'], telemetries['autogen'] = test_real_autogen()
    print(f"\n📧 AUTOGEN RESULT ({len(results['autogen'].split())} words):")
    print("-" * 40)
    print(results['autogen'])  # Show FULL output, no truncation
    
    print("\n" + "=" * 80)
    
    results['langgraph'], models_used['langgraph'], telemetries['langgraph'] = test_real_langgraph()
    print(f"\n📧 LANGGRAPH RESULT ({len(results['langgraph'].split())} words):")
    print("-" * 40)
    print(results['langgraph'])  # Show FULL output, no truncation
    
    print("\n" + "=" * 80)
    
    results['crewai'], models_used['crewai'], telemetries['crewai'] = test_real_crewai()
    print(f"\n📧 CREWAI RESULT ({len(results['crewai'].split())} words):")
    print("-" * 40)
    print(results['crewai'])  # Show FULL output, no truncation
    
    # Print telemetry comparison
    print_telemetry_comparison(telemetries)

    print("\n" + "=" * 80)
    print("🎯 REAL FRAMEWORK COMPARISON ANALYSIS")
    print("=" * 80)

    for framework, result in results.items():
        word_count = len(result.split())
        model = models_used[framework]
        email_body = extract_email_body(result)
        
        print(f"\n{framework.upper():<10}: {len(email_body.split()):>3} words | Model: {model}")
        print("-" * 60)
        print(email_body)
        print("-" * 60)

    print(f"\n💡 REAL FRAMEWORK CHARACTERISTICS:")
    print(f"• AUTOGEN:  Autonomous multi-agent conversations with group chat")
    print(f"• LANGGRAPH: State machine workflow with conditional routing")  
    print(f"• CREWAI:   Role-based task execution with sequential processes")


    
    
if __name__ == "__main__":
    main()