import os
import sys
import subprocess

def main():
    """
    Runner script for the Streamlit application.
    Resolves the path to the app inside src/ui_app and runs it.
    """
    # Get the absolute path of the current file's directory (root of the project)
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the streamlit app file
    app_path = os.path.join(project_root, "src", "ui_app", "app.py")
    
    if not os.path.exists(app_path):
        print(f"Error: Could not find Streamlit app at {app_path}")
        sys.exit(1)
    
    print(f"Starting Streamlit app from: {app_path}")
    
    # Construct the command
    # We use sys.executable to ensure we use the same python interpreter (and its environment)
    cmd = [sys.executable, "-m", "streamlit", "run", app_path]
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nStopping Streamlit app...")

if __name__ == "__main__":
    main()
