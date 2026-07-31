"""Streamlit entrypoint for Streamlit Cloud deployments.

This file simply executes the existing frontend Streamlit app at
`frontend/app.py` so Streamlit Cloud can discover and run the app
from the repository root as `streamlit_app.py`.
"""
import runpy
from pathlib import Path


ROOT = Path(__file__).parent
FRONTEND_APP = ROOT / "frontend" / "app.py"

if not FRONTEND_APP.exists():
    raise FileNotFoundError(f"Could not find frontend app at {FRONTEND_APP}")

# Execute the frontend app script in a fresh module namespace.
runpy.run_path(str(FRONTEND_APP), run_name="__main__")
