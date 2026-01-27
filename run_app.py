import subprocess
import time
import os
import sys

def run_servers():
    # Start FastAPI
    print("Starting FastAPI on port 8000...")
    api_process = subprocess.Popen([sys.executable, "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"], cwd=os.path.dirname(os.path.abspath(__file__)))
    
    # Start Flask
    print("Starting Flask on port 5000...")
    web_process = subprocess.Popen([sys.executable, "web/app.py"], cwd=os.path.dirname(os.path.abspath(__file__)))
    
    print("Both servers are running!")
    print("Go to http://localhost:5000 to use the app.")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping servers...")
        api_process.terminate()
        web_process.terminate()
        print("Servers stopped.")

if __name__ == "__main__":
    run_servers()
