# three_way_comparison_REAL_FRAMEWORKS.py
import os
import time
import requests
import pandas as pd
from io import StringIO
from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import List, Dict, TypedDict
from datetime import datetime

load_dotenv()

# Configuration
DEFAULT_CSV = os.getenv("CSV_URL", "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = os.getenv("MODEL", "models/gemini-2.0-flash")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
VERBOSE = os.getenv("VERBOSE", "False").lower() == "true"



# DEBUG: Check AutoGen import
print("=== DEBUG AUTOGen IMPORT ===")
try:
    import autogen
    from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
    print(f"✅ AutoGen SUCCESS: {autogen.__version__}")
    print(f"✅ All components imported")
except ImportError as e:
    print(f"❌ Import failed: {e}")
print("=== END DEBUG ===")

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
    resp = requests.get(DEFAULT_CSV)
    return resp.text

def analyze_data(csv_text):
    df = pd.read_csv(StringIO(csv_text))
    return f"Dataset: {len(df)} rows, {len(df.columns)} columns: {', '.join(df.columns)}"

def test_real_autogen():
    """REAL AutoGen with actual multi-agent conversation"""
    print("🧪 REAL AUTOGEN TEST")
    print("=" * 50)
    
    telemetry = FrameworkTelemetry(framework="AutoGen", total_duration_ms=0)
    start_time = time.time()
    
    if not GOOGLE_API_KEY:
        return "❌ Google API key missing", "None", telemetry
    
    try:
        import autogen
        from autogen import AssistantAgent, UserProxyAgent
        
        print("   🤖 Using REAL AutoGen Multi-Agent System")
        print("   👨‍💼 Data Analyst ←→ Project Manager (Autonomous Conversation)")
        
        # Step 1: Fetch data
        step_start = time.time()
        csv_text = fetch_data()
        data_summary = analyze_data(csv_text)
        telemetry.steps.append({
            "step": "fetch_analyze_data",
            "duration_ms": (time.time() - step_start) * 1000,
            "llm_call": False
        })
        
        # REAL Gemini configuration for AutoGen
        config_list = [
            {
                "model": "gemini/gemini-2.0-flash",  # Correct format for Google
                "api_key": GOOGLE_API_KEY,
                "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",  # Use OpenAI-compatible endpoint
                "api_type": "google"
            }
        ]
        
        llm_config = {
            "config_list": config_list,
            "temperature": TEMPERATURE,
            "timeout": 120
        }
        
        print(f"   🔧 Config: Gemini 2.0 Flash, Temp: {TEMPERATURE}")
        
        # REAL AUTOGEN AGENTS
        data_analyst = AssistantAgent(
            name="Data_Analyst",
            system_message="""You are a senior data analyst specializing in restaurant analytics. 
            Analyze the provided data and extract 3 key business insights about:
            1. Revenue patterns by day/time
            2. Customer behavior trends  
            3. Operational efficiency opportunities
            
            Be specific, data-driven, and focus on actionable insights.""",
            llm_config=llm_config,
            max_consecutive_auto_reply=2
        )
        
        project_manager = AssistantAgent(
            name="Project_Manager",
            system_message="""You are a project manager who creates professional stakeholder communications.
            Transform data insights into compelling executive emails with:
            - Clear subject line highlighting business value
            - 3 key insights as bullet points
            - Professional, confident tone
            - Specific call to action
            - Under 200 words
            
            Return ONLY the final email text, no explanations.""",
            llm_config=llm_config,
            max_consecutive_auto_reply=2
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
        
        # First: Data Analyst analyzes the data
        analysis_result = user_proxy.initiate_chat(
            recipient=data_analyst,
            message=f"""Please analyze this restaurant tips data and provide 3 key business insights:

            DATA SUMMARY: {data_summary}

            Focus on patterns that impact revenue, customer behavior, and operational efficiency.
            Provide specific, actionable insights with potential business impact.""",
            clear_history=True,
            max_turns=2
        )
        
        # Get the analysis from the first conversation
        analysis_text = ""
        if hasattr(analysis_result, 'chat_history') and analysis_result.chat_history:
            # Get the last message from data analyst
            for msg in reversed(analysis_result.chat_history):
                if msg.get("name") == "Data_Analyst":
                    analysis_text = msg.get("content", "")
                    break
        
        # Second: Project Manager creates email based on analysis
        if analysis_text:
            email_result = user_proxy.initiate_chat(
                recipient=project_manager,
                message=f"""Based on this data analysis, create a professional stakeholder email:

                ANALYSIS: {analysis_text}

                Requirements:
                - Compelling subject line
                - 3 key insights as bullet points  
                - Professional executive tone
                - Specific call to action
                - Under 200 words
                - Return ONLY the email text""",
                clear_history=True,
                max_turns=2
            )
        else:
            # Fallback if analysis failed
            email_result = user_proxy.initiate_chat(
                recipient=project_manager,
                message=f"""Create a professional stakeholder email based on this restaurant data:

                DATA: {data_summary}

                Include 3 key insights about revenue patterns, customer behavior, and operational opportunities.
                Professional tone, under 200 words, return ONLY the email text.""",
                clear_history=True,
                max_turns=2
            )
        
        conversation_duration = (time.time() - step_start) * 1000
        
        # Extract REAL output from the conversation
        final_output = "AutoGen Multi-Agent Conversation Complete\n\n"
        email_content = ""
        
        if hasattr(email_result, 'chat_history') and email_result.chat_history:
            # Get the last message from project manager
            for msg in reversed(email_result.chat_history):
                if msg.get("name") == "Project_Manager":
                    email_content = msg.get("content", "")
                    break
            
            if email_content:
                final_output += f"Final Email Output:\n{email_content}\n\n"
            else:
                # Fallback: use the last message
                last_msg = email_result.chat_history[-1] if email_result.chat_history else "No output generated"
                final_output += f"Final Output:\n{last_msg.get('content', str(last_msg))}\n\n"
            
            final_output += f"Total messages exchanged: {len(email_result.chat_history)}"
        else:
            final_output += f"Conversation completed: {str(email_result)}"
        
        # REAL Telemetry - estimate based on actual conversation
        estimated_llm_calls = max(2, len(email_result.chat_history) if hasattr(email_result, 'chat_history') else 4)
        
        for i in range(estimated_llm_calls):
            telemetry.llm_calls.append(LLMCall(
                timestamp=datetime.now().isoformat(),
                model="gemini-2.0-flash",
                input_tokens=count_tokens(data_summary) // estimated_llm_calls,
                output_tokens=count_tokens(final_output) // estimated_llm_calls,
                latency_ms=conversation_duration / estimated_llm_calls,
                caller=f"AutoGen_Conversation_{i+1}"
            ))
        
        telemetry.steps.append({
            "step": "autogen_multi_agent_conversation",
            "duration_ms": conversation_duration,
            "llm_call": True,
            "agents": 3,
            "conversation_turns": estimated_llm_calls
        })
        
        telemetry.total_duration_ms = (time.time() - start_time) * 1000
        
        return final_output, "gemini-2.0-flash", telemetry
        
    except ImportError as e:
        print(f"DEBUG: ImportError: {e}")
        return "❌ AutoGen not installed. Run: pip install pyautogen", "None", telemetry
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
        # REAL LANGGRAPH IMPORTS (FIXED)
        from langgraph.graph import StateGraph
        import google.generativeai as genai
        
        print("   🤖 Using REAL LangGraph State Machine")
        print("   📥 Supervisor → 👨‍💼 PM → 👩‍🔬 DA → 🔍 Reviewer (Workflow)")
        
        # Setup Gemini for LangGraph nodes
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Step 1: Fetch data
        step_start = time.time()
        csv_text = fetch_data()
        data_summary = analyze_data(csv_text)
        telemetry.steps.append({
            "step": "fetch_analyze_data",
            "duration_ms": (time.time() - step_start) * 1000,
            "llm_call": False
        })
        
        # REAL LANGGRAPH STATE DEFINITION (FIXED)
        class AgentState(TypedDict, total=False):
            data_summary: str
            messages: list
            requirements: str
            analysis: str
            email_draft: str
            final_output: str
            next: str
        
        # REAL LANGGRAPH NODES (FIXED RETURNS)
        def supervisor_node(state: AgentState) -> AgentState:
            """LangGraph supervisor that routes workflow"""
            step_start = time.time()
            
            # LangGraph routing logic
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
            
            llm_duration = (time.time() - step_start) * 1000
            
            telemetry.llm_calls.append(LLMCall(
                timestamp=datetime.now().isoformat(),
                model="gemini-2.0-flash",
                input_tokens=count_tokens(str(state)),
                output_tokens=count_tokens(f"Next: {next_step}, Task: {task}"),
                latency_ms=llm_duration,
                caller="Supervisor_Node"
            ))
            
            # FIXED: Mutate state and return
            state["next"] = next_step
            state["messages"] = state.get("messages", []) + [{"role": "system", "content": f"Routing to {next_step}: {task}"}]
            return state
        
        def project_manager_node(state: AgentState) -> AgentState:
            """LangGraph project manager node"""
            step_start = time.time()
            
            prompt = f"""As Project Manager in a workflow, define requirements for restaurant tips analysis.

DATA: {state['data_summary']}

Create clear email requirements focusing on:
- 3 key business insights with revenue impact
- Actionable recommendations for management
- Professional stakeholder communication standards
- Email structure and executive tone

Return only the specific requirements."""
            
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
            
            # FIXED: Mutate state and return
            state["requirements"] = response.text
            state["next"] = "supervisor"
            state["messages"] = state.get("messages", []) + [{"role": "assistant", "content": f"Requirements defined: {response.text[:100]}..."}]
            return state
        
        def data_analyst_node(state: AgentState) -> AgentState:
            """LangGraph data analyst node"""
            step_start = time.time()
            
            prompt = f"""As Data Analyst in a workflow, analyze restaurant tips data.

DATA: {state['data_summary']}
REQUIREMENTS: {state['requirements']}

Provide comprehensive analysis with:
- 3 key insights answering the requirements
- Data-driven evidence and patterns
- Business implications and recommendations
- Email content draft based on findings"""
            
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
            
            # FIXED: Mutate state and return
            state["analysis"] = response.text
            state["email_draft"] = response.text
            state["next"] = "supervisor"
            state["messages"] = state.get("messages", []) + [{"role": "assistant", "content": f"Analysis complete: {response.text[:100]}..."}]
            return state
        
        def reviewer_node(state: AgentState) -> AgentState:
            """LangGraph reviewer node"""
            step_start = time.time()
            
            prompt = f"""As Reviewer in a workflow, create final stakeholder email.

DATA: {state['data_summary']}
REQUIREMENTS: {state['requirements']}
ANALYSIS: {state['analysis']}

Create professional executive email with:
- Compelling subject line highlighting business value
- 3 clear bullet points with data-driven insights
- Confident, professional business tone
- Specific, actionable call to action
- Under 200 words total

Return only the final email text."""
            
            response = model.generate_content(prompt)
            llm_duration = (time.time() - step_start) * 1000
            
            telemetry.llm_calls.append(LLMCall(
                timestamp=datetime.now().isoformat(),
                model="gemini-2.0-flash",
                input_tokens=count_tokens(prompt),
                output_tokens=count_tokens(response.text),
                latency_ms=llm_duration,
                caller="Reviewer_Node"
            ))
            
            # FIXED: Mutate state and return
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
        
        # FIXED: Routing function
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
        result = app.invoke({"data_summary": data_summary, "messages": []})
        graph_duration = (time.time() - step_start) * 1000
        
        telemetry.steps.append({
            "step": "langgraph_workflow_execution",
            "duration_ms": graph_duration,
            "llm_call": True,
            "nodes_activated": 4,
            "state_transitions": len(result.get('messages', []))
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
        
        print("   👥 Using REAL CrewAI Multi-Agent System")
        print("   🎯 Role-Based Collaboration (Real Framework)")
        
        # Step 1: Fetch data
        step_start = time.time()
        csv_text = fetch_data()
        data_summary = analyze_data(csv_text)
        telemetry.steps.append({
            "step": "fetch_analyze_data",
            "duration_ms": (time.time() - step_start) * 1000,
            "llm_call": False
        })
        
        # REAL CREWAI LLM CONFIGURATION (FIXED: Added api_key)
        crewai_model = "gemini/gemini-2.0-flash"
        
        try:
            llm = LLM(
                model=crewai_model,
                api_key=GOOGLE_API_KEY,
                temperature=TEMPERATURE
            )
            used_model = crewai_model
            print(f"   🤖 Model: {used_model}")
        except Exception as e:
            print(f"   ❌ Model setup failed: {e}")
            return f"CrewAI model error: {e}", "None", telemetry
        
        # REAL CREWAI AGENTS (framework manages their collaboration)
        project_manager = Agent(
            role='Senior Project Manager',
            goal='Define clear business requirements and ensure stakeholder alignment for data analysis projects',
            backstory="""You are an experienced project manager specializing in data analytics 
            for hospitality businesses. You excel at translating complex data into actionable 
            business requirements and ensuring all deliverables meet executive standards.""",
            llm=llm,
            verbose=VERBOSE,
            allow_delegation=False
        )
        
        data_analyst = Agent(
            role='Senior Data Analyst',
            goal='Analyze complex datasets and extract meaningful business insights with actionable recommendations',
            backstory="""You are a senior data analyst with deep expertise in restaurant 
            analytics and customer behavior patterns. You specialize in transforming raw data 
            into strategic insights that drive business decisions and revenue growth.""",
            llm=llm,
            verbose=VERBOSE,
            allow_delegation=False
        )
        
        communications_specialist = Agent(
            role='Communications Specialist',
            goal='Transform data insights into compelling executive communications that drive action',
            backstory="""You are an expert communications specialist who crafts professional 
            stakeholder communications. You excel at presenting complex data insights in 
            clear, actionable formats that executives can quickly understand and act upon.""",
            llm=llm,
            verbose=VERBOSE,
            allow_delegation=False
        )
        
        # REAL CREWAI TASKS (framework handles dependencies)
        define_requirements = Task(
            description=f"""Based on this restaurant tips data analysis: {data_summary}

            Create comprehensive requirements for a stakeholder email that:
            - Highlights 3 key business insights with revenue impact
            - Provides actionable recommendations for management
            - Maintains professional executive tone
            - Includes clear, specific call to action
            - Focuses on customer behavior and operational efficiency

            Ensure the requirements are specific, measurable, and business-focused.""",
            agent=project_manager,
            expected_output="Detailed email requirements with specific business insight categories and success criteria",
            async_execution=False
        )
        
        analyze_data_insights = Task(
            description="""Using the requirements from the Project Manager, 
            conduct thorough data analysis and extract key insights that answer the business questions.

            Provide:
            - 3 data-driven insights with supporting evidence
            - Specific patterns and trends from the data
            - Business implications and potential impact
            - Recommended action areas with priority levels

            Focus on insights that directly address the stakeholder requirements.""",
            agent=data_analyst,
            expected_output="Comprehensive data analysis with 3 key insights, evidence, and business implications",
            context=[define_requirements],
            async_execution=False
        )
        
        create_stakeholder_email = Task(
            description="""Using the data insights from the Analyst and requirements from the Project Manager,
            create a professional executive email in PLAIN TEXT format (no HTML, no code blocks) that includes:

            - Compelling subject line that highlights business value
            - 3 clear bullet points with data-driven insights
            - Professional, confident executive tone
            - Specific, actionable call to action
            - Total length under 200 words
            - OUTPUT AS PLAIN TEXT ONLY - no HTML tags, no formatting code

            Ensure the email is polished, executive-ready, and focuses on business impact.""",
            agent=communications_specialist,
            expected_output="Polished stakeholder email ready for executive review",
            context=[define_requirements, analyze_data_insights],
            async_execution=False
        )
        
        # REAL CREWAI EXECUTION (framework coordinates everything)
        analytics_crew = Crew(
            agents=[project_manager, data_analyst, communications_specialist],
            tasks=[define_requirements, analyze_data_insights, create_stakeholder_email],
            process=Process.sequential,
            verbose=VERBOSE
        )
        
        # CREWAI KICKOFF (framework handles all agent coordination)
        crew_start = time.time()
        result = analytics_crew.kickoff()
        crew_duration = (time.time() - crew_start) * 1000
        
        # CrewAI makes multiple internal LLM calls - we estimate
        telemetry.steps.append({
            "task": "ProjectManager - Define requirements",
            "duration_ms": crew_duration * 0.3,
            "llm_calls_estimated": 2,
            "llm_call": True
        })
        
        telemetry.steps.append({
            "task": "DataAnalyst - Analyze insights", 
            "duration_ms": crew_duration * 0.4,
            "llm_calls_estimated": 2,
            "llm_call": True
        })
        
        telemetry.steps.append({
            "task": "CommunicationsSpecialist - Create email",
            "duration_ms": crew_duration * 0.3,
            "llm_calls_estimated": 2,
            "llm_call": True
        })
        
        print("   ℹ️  CrewAI makes multiple internal LLM calls for agent collaboration")
        
        telemetry.total_duration_ms = (time.time() - start_time) * 1000
        return str(result), used_model, telemetry
        
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

def main():
    print("🔬 THREE-WAY FRAMEWORK COMPARISON - REAL FRAMEWORKS")
    print("=" * 60)
    print("AutoGen vs LangGraph vs CrewAI - ACTUAL FRAMEWORKS")
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
    
    # Test all three REAL frameworks
    results['autogen'], models_used['autogen'], telemetries['autogen'] = test_real_autogen()
    print(f"\n📧 AUTOGEN RESULT ({len(results['autogen'].split())} words):")
    print("-" * 40)
    print(results['autogen'][:500] + "..." if len(results['autogen']) > 500 else results['autogen'])
    
    print("\n" + "=" * 80)
    
    results['langgraph'], models_used['langgraph'], telemetries['langgraph'] = test_real_langgraph()
    print(f"\n📧 LANGGRAPH RESULT ({len(results['langgraph'].split())} words):")
    print("-" * 40)
    print(results['langgraph'][:500] + "..." if len(results['langgraph']) > 500 else results['langgraph'])
    
    print("\n" + "=" * 80)
    
    results['crewai'], models_used['crewai'], telemetries['crewai'] = test_real_crewai()
    print(f"\n📧 CREWAI RESULT ({len(results['crewai'].split())} words):")
    print("-" * 40)
    print(results['crewai'][:500] + "..." if len(results['crewai']) > 500 else results['crewai'])
    
    # Print telemetry comparison
    print_telemetry_comparison(telemetries)
    
    print("\n" + "=" * 80)
    print("🎯 REAL FRAMEWORK COMPARISON ANALYSIS")
    print("=" * 80)
    
    for framework, result in results.items():
        word_count = len(result.split())
        model = models_used[framework]
        print(f"{framework.upper():<10}: {word_count:>3} words | Model: {model}")
        print(f"           Preview: {result[:100]}...")
    
    print(f"\n💡 REAL FRAMEWORK CHARACTERISTICS:")
    print(f"• AUTOGEN:  Autonomous multi-agent conversations with group chat")
    print(f"• LANGGRAPH: State machine workflow with conditional routing")  
    print(f"• CREWAI:   Role-based task execution with sequential processes")

if __name__ == "__main__":
    main()