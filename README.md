# 🤖 AI Agent Face-Off: AutoGen vs LangGraph vs CrewAI

A practical, **real-world comparison** of three leading AI agent frameworks — **AutoGen**, **LangGraph**, and **CrewAI** — solving the same business intelligence problem using *authentic computed data* (not simulations).

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Frameworks](https://img.shields.io/badge/Frameworks-AutoGen%20|%20LangGraph%20|%20CrewAI-orange)
![LLM](https://img.shields.io/badge/LLM-Gemini%202.0%20Flash%20%7C%20OpenAI-purple)

---

## 🎯 What We’re Building

An **automated data-reporting system** that:

- 📊 Fetches and analyzes CSV data automatically  
- ✍️ Generates professional stakeholder emails  
- 🤖 Runs completely autonomously across three frameworks  
- 🔄 Benchmarks agent reasoning, collaboration, and orchestration  

---

## 📁 Project Structure
ai-agent-faceoff/
├── .gitignore # Protects API keys
├── README.md # This file
├── requirements.txt # Dependencies
├── .env # Your environment vars (create from template)
├── env.template # Sample .env file
├── common_utils.py # Shared data utilities
├── autogen_demo.py # Conversational agents (AutoGen)
├── langgraph_demo.py # Workflow-based agents (LangGraph)
├── crewai_demo.py # Role-based collaboration (CrewAI)
└── three_way_comparison_pre_cal_real_frameworks.py # Unified benchmark script

---

## 🚀 Quick Start

### 1️⃣ Prerequisites
- Python 3.10 +
- API Key (OpenAI or Gemini)
- Internet access for CSV data

### 2️⃣ Setup & Installation
```bash
# Clone and enter
git clone https://github.com/YOUR_USERNAME/ai-agent-faceoff.git
cd ai-agent-faceoff

# Create and activate virtual env
python -m venv .venv
source .venv/bin/activate  # or .\venv\Scripts\activate on Windows

# Install deps
pip install -r requirements.txt

3️⃣ Configuration
cp env.template .env

Edit .env:

# Required: Your Google API Key
GOOGLE_API_KEY=my google gemini key
MODEL=models/gemini-2.0-flash

OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini
CSV_URL=https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv

# Optional: Comma-separated list of models to try (override defaults)
AVAILABLE_MODELS=models/gemini-2.0-flash,models/gemini-2.5-flash,models/gemini-pro-latest
CREWAI_FALLBACK_MODELS=gemini/gemini-1.5-flash,gemini/gemini-1.5-pro

# Optional: CrewAI tracing (for debugging)
CREWAI_TRACING_ENABLED=true


▶️ Run the Tests
# AutoGen demo (conversational)
python autogen_demo.py

# LangGraph demo (workflow)
python langgraph_demo.py

# CrewAI demo (collaborative)
python crewai_demo.py

# Unified comparison with telemetry
python three_way_comparison_pre_cal_real_frameworks.py

📊 Framework Comparison
Framework	Paradigm	Key Strength	Ideal Use
AutoGen	Conversational agents	Emergent reasoning & autonomy	Exploratory analytics, research
LangGraph	State-driven workflow	Deterministic orchestration & traceability	Data pipelines, ETL processes
CrewAI	Role-based collaboration	Human-like teamwork & business fidelity	Business reporting, multi-role automation

🧠 How It Works
🔹 AutoGen Approach
Creates two AI agents (ProjectManager ↔ DataAnalyst)
Agents collaborate through multi-turn conversation
Emergent workflow – agents decide how to approach the task

🔸 LangGraph Approach
Builds structured graph nodes: fetch → analyze → report
Deterministic execution with state persistence
Predictable and production-ready

🟢 CrewAI Approach
Defines role-based agents (ProjectManager, Analyst, Comms)
Executes sequential collaboration with shared context
Produces high-fidelity, stakeholder-ready reports

📈 Real Telemetry (Authentic Run)
From three_way_comparison_pre_cal_real_frameworks.py using real metrics from tips.csv:
Framework	LLM Calls	Duration	Behavior
AutoGen	2	~5.6 s	Conversational exchange (PM ↔ Analyst)
LangGraph	7	~4.3 s	Structured node execution (Supervisor → Analyst → Reviewer)
CrewAI	(Cloud trace)	~21 s	Multi-role collaboration via CrewAI Plus trace URL
Dataset Used:
Authentic Seaborn tips.csv → pre-calculated metrics:
Total Revenue: $4,827.77
Total Tips: $731.58
Average Tip Rate: 16.1 %

💡 Key Features
Data Agnostic – works with any CSV URL
Professional Output – executive-ready emails
Error Resilient – handles network and data issues
Security First – .gitignore protects API keys
Framework Agnostic Core – shared data logic in common_utils.py

🧾 Example Output
Subject: Restaurant Tip Analysis – Key Insights for Business Optimization

• Dataset contains 244 transactions with 7 variables.  
• Average tip rate ≈ 16.1 %.  
• Friday and Saturday dinners show higher average spend.  
• Recommendations: Target weekend evenings for promotions and staff optimization.

🧰 Extend It
Add multi-source data (REST, SQL, S3)
Add human approval steps (“human-in-the-loop”)
Export to Slack, PDF, or QuickSight
Integrate CrewAI memory for cross-report context

⚠️ Security Notice
.gitignore prevents .env from being committed.
Never share API keys in code or logs.
Always check git status before committing.
