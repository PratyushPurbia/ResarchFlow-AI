"""
ResearchFlow AI - Entry point
Run with: streamlit run app.py
"""
import subprocess
import sys

if __name__ == "__main__":
    print("Starting ResearchFlow AI...")
    print("Opening Streamlit app at http://localhost:8501")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
