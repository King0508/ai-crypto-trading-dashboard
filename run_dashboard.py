"""
Start the Streamlit dashboard.

Usage:
    python run_dashboard.py
"""
import subprocess
import sys

if __name__ == "__main__":
    print("\n" + "="*60)
    print("STARTING DASHBOARD")
    print("="*60)
    print("Opening in browser...")
    print("Dashboard will be available at: http://localhost:8501")
    print("Press Ctrl+C to stop")
    print("="*60 + "\n")
    
    subprocess.run([
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "dashboard/app_live.py",
        "--server.headless=true"
    ])

