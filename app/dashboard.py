"""
app/dashboard.py
Entry-point for the Streamlit dashboard.

Run with:
    streamlit run app/dashboard.py
"""
import sys
from pathlib import Path

# Make the project root importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Re-export the Streamlit app defined in dashboard/app.py
from dashboard.app import *  # noqa: F401, F403, E402
