"""Punto de entrada por defecto para Streamlit Community Cloud: ejecuta el dashboard."""
import os
import runpy

runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard", "app.py"), run_name="__main__")
