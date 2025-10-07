🤖 AI Agent Face-Off: AutoGen vs LangGraph

A practical comparison of two leading AI agent frameworks solving the same real-world business problem. This demo showcases different approaches to building autonomous AI systems for data analysis and reporting.

https://img.shields.io/badge/Python-3.10%252B-blue
https://img.shields.io/badge/License-MIT-green
https://img.shields.io/badge/LLM-OpenAI-purple

🎯 What We're Building

Automated Data Reporting System that:

📊 Fetches and analyzes CSV data automatically
✍️ Generates professional stakeholder emails
🤖 Runs completely autonomously
🔄 Compares two different AI agent architectures
📁 Project Structure

text
ai-agent-faceoff/
├── .gitignore                 # Git ignore rules (security critical!)
├── README.md                  # This file
├── requirements.txt           # Pinned dependencies
├── .env                       # Environment variables (create this)
├── env.template              # Template for environment variables
├── common_utils.py           # Shared data processing utilities
├── autogen_demo.py           # Chat-based agents (AutoGen)
└── langgraph_demo.py         # Workflow-based agents (LangGraph)
🚀 Quick Start

1. Prerequisites

Python 3.10+
OpenAI API key (Get one here)
2. Setup & Installation

bash
# Clone or create project directory
mkdir ai-agent-faceoff && cd ai-agent-faceoff

# Create virtual environment
python -m venv .venv

# Activate environment
# macOS/Linux:
source .venv/bin/activate
# Windows (PowerShell):
# .\.venv\Scripts\Activate.ps1
# Windows (CMD):
# .\venv\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt
3. Configuration

Create .env file by copying the template:

bash
# Copy the template
cp env.template .env

# Then edit .env with your actual API key
Edit .env file:

env
# REQUIRED: Your OpenAI API key
OPENAI_API_KEY=sk-your-actual-key-here

# OPTIONAL: Customize these
OPENAI_MODEL=gpt-4o-mini
CSV_URL=https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv
4. Run the Demos

bash
# Run AutoGen demo (chat-based agents)
python autogen_demo.py

# Run LangGraph demo (workflow-based agents)  
python langgraph_demo.py
📊 Framework Comparison

Aspect	AutoGen	LangGraph
Approach	Conversational AI team	Deterministic workflow
Metaphor	Team meeting	Assembly line
Strengths	Dynamic interactions, brainstorming	Predictable, debuggable, production-ready
Best For	Creative tasks, research	Data pipelines, ETL processes
Complexity	Higher (emergent behavior)	Lower (explicit flow)
Control	Agent-driven workflow	Developer-defined workflow
🧠 How It Works

AutoGen Approach (autogen_demo.py)

Creates two AI agents that converse like a human team
ProjectManager orchestrates the task
DataAnalyst analyzes data and writes the email
Agents collaborate through multi-turn conversation
Emergent workflow - agents decide how to approach the task
LangGraph Approach (langgraph_demo.py)

Builds a structured workflow with clear steps
Node 1: Fetch CSV data
Node 2: Analyze data
Node 3: Draft email
Deterministic execution from start to finish
Explicit workflow - developer defines every step
💡 Key Features

Data Agnostic: Works with any public CSV URL
Professional Output: Generates executive-ready emails
Error Resilient: Handles network issues and data problems
Production Ready: Includes proper error handling and logging
Framework Agnostic Core: Shared data utilities work with both approaches
Security First: Proper .gitignore to protect API keys
🛠️ Customization

Change Data Source

Edit .env:

env
CSV_URL=https://raw.githubusercontent.com/your-username/your-repo/data.csv
Modify Email Template

Edit the prompt in both demo files to change:

Email tone and style
Required sections
Insight count and format
Add New Analysis

Extend common_utils.py:

python
def custom_analysis(df):
    # Add your custom analysis logic
    return insights
🐛 Troubleshooting

Common Issues

API Key Errors:

bash
# Ensure your .env file is properly formatted
OPENAI_API_KEY=sk-your-key-here
# No quotes, no spaces around =
Module Not Found:

bash
# Reinstall dependencies
pip install -r requirements.txt
Network Issues:

Check if CSV URL is accessible
Corporate firewalls may block GitHub raw content
Try alternative CSV URLs
Version Conflicts:

bash
# Use exact versions from requirements.txt
pip freeze | grep -E "(autogen|langgraph)"
Debug Mode

Add debug prints to see the workflow:

python
# In common_utils.py
print(f"📥 Fetched {len(csv_text)} characters")
print(f"📊 Analyzed {summary['rows']} rows, {len(summary['columns'])} columns")
📈 Example Output

Both scripts will generate professional emails like:

text
Subject: Key Insights from Restaurant Tips Analysis

• Dataset contains 244 transactions with 7 variables including total bill, tip amount, and party size
• Analysis reveals patterns in customer spending behavior across different days and times
• Data supports segmentation by customer demographics and dining preferences

Recommended next step: Schedule a deep-dive session to explore specific customer segments and optimization opportunities.
🎯 Use Cases

Ideal For:

Business Intelligence: Automated weekly reporting
Data Monitoring: Real-time metric alerts
Stakeholder Updates: Executive communication
Proof of Concepts: Agent framework evaluation
Education: Learning different AI architectures
Extend For:

Multi-source data (APIs, databases)
Custom analysis (industry-specific metrics)
Multi-format output (Slack messages, PDF reports)
Approval workflows (human-in-the-loop)
🔧 Technical Details

Dependencies

Data Processing: pandas, requests
AI Frameworks: pyautogen, langgraph, langchain
LLM: OpenAI GPT models
Environment: python-dotenv
Architecture

Separation of Concerns: Data processing vs. AI reasoning
Error Handling: Graceful failure at each step
Configurable: Environment-based configuration
Extensible: Modular design for easy modifications
🚀 GitHub Deployment

First-Time Setup (After Cloning)

bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/ai-agent-faceoff.git
cd ai-agent-faceoff

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .\venv\Scripts\activate on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy environment template
cp env.template .env

# 5. Edit .env with your API key
# Open .env and replace with your actual OpenAI API key
For Project Maintainers

bash
# Initialize git (if starting fresh)
git init
git add .
git commit -m "feat: Initial AI agent comparison demo"

# Connect to GitHub and push
git remote add origin https://github.com/YOUR_USERNAME/ai-agent-faceoff.git
git branch -M main
git push -u origin main
⚠️ Security Notice

The .gitignore file ensures your .env with API keys is never committed to version control
Never remove .gitignore from the project
New users must create their own .env file after cloning
Always verify git status doesn't show .env before committing
🤝 Contributing

Feel free to:

Add new agent frameworks (CrewAI, etc.)
Implement additional data sources
Create custom analysis modules
Improve error handling and logging
Add more examples and use cases
📄 License

MIT License - feel free to use in personal and commercial projects.

🆕 What's Next?

Add CrewAI for a third framework comparison
Implement database sources (SQL, Snowflake)
Add human approval steps
Create web interface for easy configuration
Add monitoring and performance tracking
Happy agent building! 🚀

For questions or issues, please open a GitHub Issue in the repository.
