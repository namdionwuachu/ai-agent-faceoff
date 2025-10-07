# common_utils.py
import os
import pandas as pd
import requests
from io import StringIO
from dotenv import load_dotenv

load_dotenv()

# Default to a sample dataset (restaurant tips)
DEFAULT_CSV = os.getenv(
    "CSV_URL",
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
)

def fetch_csv_text(url: str = DEFAULT_CSV, timeout: int = 30) -> str:
    """Fetch CSV data from URL and return as text"""
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.text

def analyse_csv_text(csv_text: str) -> dict:
    """Analyze CSV and return structured summary"""
    df = pd.read_csv(StringIO(csv_text))
    
    summary = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "preview_head": df.head(5).to_dict(orient="records"),  # First 5 rows
    }
    
    # Add basic statistics
    try:
        desc = df.describe(include="all").to_dict()
    except Exception:
        desc = {}
    summary["describe"] = desc
    
    return summary

def format_summary_for_prompt(summary: dict, max_cols: int = 10) -> str:
    """Format the analysis for LLM consumption"""
    cols = summary.get("columns", [])[:max_cols]
    lines = [
        f"Rows: {summary.get('rows', 0)}",
        f"Columns ({len(summary.get('columns', []))}): {', '.join(cols)}",
    ]
    if "describe" in summary and summary["describe"]:
        lines.append("Basic stats available.")
    if "preview_head" in summary:
        lines.append(f"Preview rows: {len(summary['preview_head'])}")
    return "\n".join(lines)
