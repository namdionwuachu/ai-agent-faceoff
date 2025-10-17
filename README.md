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
└── three_way_framework_comparison_draft_v0.2.py # Unified benchmark script

---

## 🚀 Quick Start

### 1️⃣ Prerequisites
- Python 3.10+ (tested on 3.11)
- Google Gemini API key (required for the unified benchmark)
- Internet access for CSV data
- (Optional) OpenAI API key if you want to run OpenAI variants in the individual demos

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

# Sanity check: make sure installs are visible to THIS interpreter
python - <<'PY'
import sys
print("Python:", sys.executable)
import autogen, google.generativeai, pandas, langgraph
print("autogen:", getattr(autogen, "__version__", "unknown"))
import crewai
print("crewai:", getattr(crewai, "__version__", "unknown"))
print("google-generativeai OK")
PY



### 🔒 Optional: Lock your working environment

Once everything runs successfully, you can capture the exact versions that worked:

```bash
# Freeze your current environment
python -m pip freeze > constraints.txt

# Reproduce the same setup later (or on another machine)
python -m pip install -r requirements.txt -c constraints.txt

# 🧠 Why it’s useful
- Keeps your results reproducible even if PyPI updates break backward compatibility.
- Great for multi-framework projects like yours (AutoGen, LangGraph, CrewAI evolve fast).
- A lightweight alternative to full dependency managers (like Poetry or Pipenv).

3️⃣ Configuration
cp env.template .env

Edit .env:

# --- Required for unified benchmark (Gemini) ---
GOOGLE_API_KEY=your-google-api-key

# Model names:
# - Native Gemini client uses "models/gemini-2.0-flash"
# - OpenAI-compatible path uses "gemini-2.0-flash" (no "models/" prefix)

# Used by native google-generativeai calls (LangGraph, sanity checks)
MODEL=models/gemini-2.0-flash

# Used by CrewAI (gemini provider naming)
CREWAI_MODEL=gemini/gemini-2.0-flash

# Optional CSV override
CSV_URL=https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv

# --- Optional: OpenAI (for separate demos) ---
OPENAI_API_KEY=sk-your-openai-key
OPENAI_MODEL=gpt-4o-mini

# --- Optional: multi-model lists for experiments ---
AVAILABLE_MODELS=models/gemini-2.0-flash,models/gemini-2.5-flash
CREWAI_FALLBACK_MODELS=gemini/gemini-1.5-flash,gemini/gemini-1.5-pro

# --- Optional: debug/noise controls ---
VERBOSE=False
CREWAI_TRACING_ENABLED=true
GRPC_VERBOSITY=ERROR
GLOG_minloglevel=2


▶️ Run the Tests
# Activate your venv first
source .venv/bin/activate

# Demos
python autogen_demo.py
python langgraph_demo.py
python crewai_demo.py

# Unified benchmark
python three_way_framework_comparison_draft_v0.2.py


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

## 🛠️ Troubleshooting

❌ ImportError: No module named 'autogen'
- You’re likely in a different Python env. Confirm with:
  `which python` and `python -m pip -V`
- Install into that interpreter:
  `python -m pip install pyautogen==0.2.27`

❌ AutoGen runtime error: Completions.create() got an unexpected keyword argument 'price'
- Cause: passing unsupported keys in AutoGen `config_list`.
- Fix: Do NOT include custom fields like `price` in the LLM config.
  The script already sanitizes and uses only supported keys.

ℹ️ ALTS creds ignored. Not running on GCP...
- Benign gRPC message; can be silenced with:
  `export GRPC_VERBOSITY=ERROR; export GLOG_minloglevel=2`

⚠️ flaml.automl is not available...
- Harmless. Install optional extra to silence:
  `python -m pip install "flaml[automl]"`


🧰 Extend It
Add multi-source data (REST, SQL, S3)
Add human approval steps (“human-in-the-loop”)
Export to Slack, PDF, or QuickSight
Integrate CrewAI memory for cross-report context

⚠️ Security Notice
.gitignore prevents .env from being committed.
Never share API keys in code or logs.
Always check git status before committing.


# 🤖 AI Agent Faceoff: AutoGen vs LangGraph vs CrewAI

## Overview

This experiment benchmarks **three multi-agent orchestration frameworks** — **AutoGen**, **LangGraph**, and **CrewAI** — under identical conditions.

Each framework was tasked with analyzing the same dataset (`tips.csv`) using **Gemini 2.0 Flash**, producing an **executive email** from pre-calculated business intelligence (BI) metrics.  
The prompts, roles, temperature, and BI block were identical across all frameworks to ensure fairness.

---

## 🎯 Objectives

1. Validate **numerical fidelity** — no fabricated or derived figures.
2. Compare **autonomy and structure** of agent communications.
3. Measure **runtime, token use, and workflow latency**.
4. Observe **tone and executive suitability** of final outputs.

---

## ⚙️ Configuration

| Parameter | Value |
|------------|--------|
| **Model** | `gemini-2.0-flash` |
| **Temperature** | 0.2 |
| **Data Source** | [tips.csv (Seaborn dataset)](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv) |
| **BI Metrics** | Pre-calculated (no in-model computation) |
| **Output Format** | Plain text email between markers `---BEGIN EMAIL---` and `---END EMAIL---` |
| **Validation** | Strict numeric match (verbatim from BI block) |

---

## 📊 Framework Comparison Summary

| Framework | Subject | Final Email Summary | Numerical Accuracy | Protocol Compliance | Tone & Structure | Runtime | Observations |
|------------|----------|---------------------|--------------------|---------------------|------------------|----------|---------------|
| **AutoGen** | *Revenue Growth Opportunities* | Dinner $3660.30 vs Lunch $1167.47; Saturday 87 tx; Weekend $3405.56 vs Weekday $1422.21; CTA to discuss strategies. | ✅ Perfect (verbatim BI values) | ✅ BEGIN/END respected | Concise, plain-text, 3 bullets, <200 words | ⚡ **Fastest (~0.4s)** | Autonomous multi-agent chat; minimal drift; strong baseline output. |

| **LangGraph** | *Weekly Performance Update / Key Business Insights* | Total revenue $4827.77; Dinner $3660.30 ($20.80 avg); Weekend $3405.56 vs Weekday $1422.21; Sat 87 tx; clear exec tone. | ✅ Perfect (verbatim BI values) | ✅ BEGIN/END respected | Structured executive summary + recommendations; <200 words | ⏱️ **Moderate (~5.1s)** | Most complete and professional output; ideal for executive summaries. |

| **CrewAI** | *Restaurant Performance Update - Key Insights and Recommendations* | Total $4827.77 revenue, $731.58 tips; Sat 87 tx ($1778.40); Dinner $3660.30 vs Lunch $1167.47; Weekend $3405.56 vs Weekday $1422.21; CTA. | ✅ Perfect (verbatim BI values) | ✅ BEGIN/END respected | Comprehensive, slightly verbose, 3 bullets + summary | 🕒 **Slowest (~11s)** | Sequential role-based orchestration; robust but higher latency. |

---

## 🧩 Detailed Observations

### 🧠 AutoGen
- **Type:** Autonomous multi-agent group chat  
- **Behavior:** Rapid response chain; each agent independently interprets instructions.  
- **Strength:** Fastest runtime; consistent numeric validation pass.  
- **Weakness:** Minimal context framing (no high-level executive intro).  

### ⚙️ LangGraph
- **Type:** Deterministic state-machine workflow  
- **Behavior:** Clean agent transitions (PM → Analyst → Reviewer).  
- **Strength:** Balanced structure; best executive readability.  
- **Weakness:** Slightly longer latency due to strict state validation.  

### 👥 CrewAI
- **Type:** Sequential, role-based collaboration  
- **Behavior:** Explicit task chaining with full context pass.  
- **Strength:** Most human-like narrative; rich context retention.  
- **Weakness:** Heaviest runtime; limited telemetry (tokens hidden by default).  

---

## 🧪 Technical Takeaways

- All three frameworks now **pass numeric validation** thanks to:
  - “Verbatim numbers only” rule  
  - BEGIN/END email extraction  
  - Validation of **Comms output**, not prompt text  

- **Speed vs Structure tradeoff:**
  - AutoGen → ⚡ Speed  
  - LangGraph → 🎯 Clarity  
  - CrewAI → 🧩 Context depth  

- **Best overall balance:** *LangGraph* (exec polish)  
- **Best for rapid experimentation:** *AutoGen*  
- **Best for storytelling and context-rich output:** *CrewAI*

---

## 📁 Files Included

| File | Description |
|------|--------------|
| `Framework_Comparison_Summary.csv` | CSV version of comparison table |
| `granualar detail of comms between agents.docx` | Original step-by-step message log |
| `same_data_same_prompt_same_model_draft_v0.2.py` | Test harness for identical multi-framework runs |
| `README.md` | This documentation |

---

## 🚀 Next Steps

1. Integrate token and latency telemetry for CrewAI via custom wrapper.  
2. Add LLM variants (e.g., Claude 3.5, GPT-4o) under same test harness.  
3. Visualize framework runtimes and token efficiency over multiple datasets.  

---

**Author:** Namdi Onwuachu  
**Project:** *AI Agent Faceoff — Evaluating Multi-Agent Framework Behavior Under Identical Conditions*  
**Date:** October 2025
