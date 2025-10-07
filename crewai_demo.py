# crewai_demo.py - CrewAI Implementation (No Hardcoded Values)
import os
import requests
import pandas as pd
from io import StringIO
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process

load_dotenv()

print("🚀 CREWAI: Role-Based Multi-Agent Crew")
print("=" * 60)

# Configuration from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CSV_URL = os.getenv("CSV_URL", "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv")
MODEL_NAME = os.getenv("MODEL", "models/gemini-2.0-flash")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
VERBOSE = os.getenv("VERBOSE", "True").lower() == "true"

if not GOOGLE_API_KEY:
    print("❌ Please set GOOGLE_API_KEY in your .env file")
    exit()

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    print("❌ Please install: pip install langchain-google-genai")
    exit()

def fetch_csv_data(url=CSV_URL):
    """Fetch and return CSV data as text"""
    print("📥 Fetching data...")
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.text

def analyze_data_structure(csv_text):
    """Analyze CSV structure for the agents"""
    df = pd.read_csv(StringIO(csv_text))
    summary = {
        "rows": len(df),
        "columns": list(df.columns),
        "sample_data": df.head(3).to_dict('records')
    }
    return summary

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
            "models/gemini-pro-latest"
        ]

def get_working_llm():
    """Get a working LLM with fallback models"""
    available_models = get_available_models()
    
    for model_name in available_models:
        try:
            print(f"🔄 Testing model: {model_name}")
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=GOOGLE_API_KEY,
                temperature=TEMPERATURE
            )
            # Test the model
            llm.invoke("Test")
            print(f"✅ Using model: {model_name}")
            return llm, model_name
        except Exception as e:
            print(f"❌ {model_name} failed: {str(e)[:80]}...")
            continue
    
    raise Exception("No working models found. Check your .env configuration.")

def main():
    print("🔧 Configuration:")
    print(f"   - Data Source: {CSV_URL}")
    print(f"   - Target Model: {MODEL_NAME}")
    print(f"   - Temperature: {TEMPERATURE}")
    print(f"   - Verbose: {VERBOSE}")
    print("=" * 60)

    # Get working LLM
    try:
        llm, working_model = get_working_llm()
        print(f"🤖 AI Model: {working_model}")
    except Exception as e:
        print(f"❌ Failed to initialize LLM: {e}")
        return

    print("\n🔧 Initializing Specialized Agent Crew...")
    print("Agents: 👨‍💼 Project Manager + 👩‍🔬 Data Analyst + 📊 Business Strategist")
    print("=" * 60)

    # =====================
    # CREWAI AGENT DEFINITIONS
    # =====================

    # Agent 1: Project Manager - Defines requirements and scope
    project_manager = Agent(
        role='Senior Project Manager',
        goal='Define clear business requirements and ensure the analysis meets stakeholder needs',
        backstory="""You are an experienced project manager specializing in data analytics projects. 
        You have a keen eye for business value and ensure that every analysis delivers actionable 
        insights for decision-makers. You're excellent at translating business problems into 
        clear analytical requirements.""",
        llm=llm,
        verbose=VERBOSE
    )

    # Agent 2: Data Analyst - Performs the technical analysis
    data_analyst = Agent(
        role='Senior Data Analyst',
        goal='Extract meaningful insights from data and identify key patterns',
        backstory="""You are a skilled data analyst with expertise in statistical analysis and 
        data visualization. You excel at finding hidden patterns in data and translating complex 
        findings into understandable insights. You're meticulous about data quality and 
        analytical rigor.""",
        llm=llm,
        verbose=VERBOSE
    )

    # Agent 3: Business Strategist - Provides strategic recommendations
    business_strategist = Agent(
        role='Business Strategy Consultant',
        goal='Translate data insights into actionable business recommendations',
        backstory="""You are a strategic consultant who helps businesses make data-driven decisions. 
        You excel at connecting analytical findings to real-world business outcomes and 
        providing clear, actionable recommendations. You understand restaurant operations, 
        customer behavior, and revenue optimization strategies.""",
        llm=llm,
        verbose=VERBOSE
    )

    # =====================
    # DATA PREPARATION
    # =====================
    print("\n📊 Preparing data for analysis...")
    csv_text = fetch_csv_data()
    data_summary = analyze_data_structure(csv_text)
    
    print(f"📈 Dataset Overview:")
    print(f"   - Rows: {data_summary['rows']}")
    print(f"   - Columns: {len(data_summary['columns'])}")
    print(f"   - Features: {', '.join(data_summary['columns'])}")

    # =====================
    # CREWAI TASK DEFINITIONS
    # =====================

    # Task 1: Project Manager defines requirements
    define_requirements_task = Task(
        description=f"""Analyze this restaurant tips dataset and define the key business questions:

        DATASET OVERVIEW:
        - Size: {data_summary['rows']} transactions
        - Features: {', '.join(data_summary['columns'])}
        
        Create a comprehensive set of requirements that will guide the data analysis. Focus on:
        1. Key business metrics to track
        2. Customer behavior patterns to investigate
        3. Operational insights that would be valuable
        4. Revenue optimization opportunities
        
        Provide clear, specific questions for the data analyst to answer.""",
        agent=project_manager,
        expected_output="A detailed set of analytical requirements and key business questions"
    )

    # Task 2: Data Analyst performs analysis
    analyze_data_task = Task(
        description="""Using the requirements from the Project Manager, perform a comprehensive 
        analysis of the restaurant tips data. Your analysis should:

        1. Identify key patterns and correlations in the data
        2. Calculate important metrics and statistics
        3. Highlight surprising or counter-intuitive findings
        4. Provide data-driven insights about customer behavior
        5. Note any data quality issues or limitations

        Focus on providing clear, evidence-based insights that the business strategist can use.""",
        agent=data_analyst,
        expected_output="Comprehensive data analysis with key patterns, metrics, and insights",
        context=[define_requirements_task]
    )

    # Task 3: Business Strategist creates stakeholder communication
    create_stakeholder_email_task = Task(
        description="""Based on the data analysis findings, create a professional stakeholder email that:

        MUST INCLUDE:
        - Clear, compelling subject line
        - Executive summary of key findings
        - 3-5 data-driven insights with business implications
        - Specific, actionable recommendations
        - Professional tone suitable for executives
        - Clear call to action for next steps
        - Total length under 250 words

        Focus on business impact and actionable insights rather than technical details.
        Make it compelling and easy to understand for non-technical stakeholders.""",
        agent=business_strategist,
        expected_output="Professional stakeholder email with insights and recommendations",
        context=[analyze_data_task]
    )

    # =====================
    # CREW EXECUTION
    # =====================
    print("\n👥 Assembling and launching agent crew...")
    print("🔄 Process: Sequential task execution with dependencies")
    print("-" * 50)

    # Create the crew
    analytics_crew = Crew(
        agents=[project_manager, data_analyst, business_strategist],
        tasks=[define_requirements_task, analyze_data_task, create_stakeholder_email_task],
        process=Process.sequential,  # Tasks execute in order with dependencies
        verbose=VERBOSE
    )

    print("\n🎯 Starting crew execution...")
    print("1. 👨‍💼 Project Manager defining requirements...")
    print("2. 👩‍🔬 Data Analyst performing analysis...") 
    print("3. 📊 Business Strategist creating stakeholder email...")
    print("-" * 50)

    # Execute the crew
    try:
        result = analytics_crew.kickoff()
        
        print("\n" + "=" * 80)
        print("📧 FINAL STAKEHOLDER EMAIL")
        print("=" * 80)
        print(result)
        
        print("\n" + "=" * 80)
        print("🔄 CREWAI WORKFLOW COMPLETED")
        print("=" * 80)
        print("1. 👨‍💼 Project Manager: Defined analytical requirements")
        print("2. 👩‍🔬 Data Analyst: Performed comprehensive data analysis")
        print("3. 📊 Business Strategist: Created stakeholder communication")
        print("4. 🤝 Crew Coordination: Sequential task execution")
        print(f"5. 🤖 AI Model: {working_model}")
        print("6. 📧 Result: Professional business email delivered")
        
        print("\n🎉 CREWAI DEMO SUCCESSFUL!")
        print("Three specialized agents collaborated to transform raw data into business insights!")
        
    except Exception as e:
        print(f"❌ Crew execution failed: {e}")
        print("\n💡 Troubleshooting tips:")
        print("• Check your GOOGLE_API_KEY in .env")
        print("• Ensure crewai and langchain-google-genai are installed")
        print("• Verify internet connectivity for API calls")
        print("• Try different models via AVAILABLE_MODELS in .env")

if __name__ == "__main__":
    main()
