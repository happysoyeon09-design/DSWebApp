import subprocess
import sys
import os
import time
import requests
import signal

def run_tests():
    print("Starting integration tests...")
    
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    api_main = os.path.join(base_dir, "api", "main.py")
    web_app = os.path.join(base_dir, "web", "app.py")
    
    # Start FastAPI
    print("Launching FastAPI...")
    api_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=base_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Start Flask
    print("Launching Flask...")
    web_process = subprocess.Popen(
        [sys.executable, "web/app.py"],
        cwd=base_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for servers to start
    time.sleep(5)
    
    try:
        # Test 1: Flask Index
        print("Testing Flask Homepage...")
        r_web = requests.get("http://127.0.0.1:5000/")
        assert r_web.status_code == 200, f"Flask failed: {r_web.status_code}"
        print("PASS: Flask Homepage served.")
        
        # Test 2: FastAPI Upload
        print("Testing CSV Upload...")
        with open("test_data.csv", "rb") as f:
            files = {"file": ("test_data.csv", f, "text/csv")}
            r_upload = requests.post("http://127.0.0.1:8000/upload", files=files)
        
        assert r_upload.status_code == 200, f"Upload failed: {r_upload.status_code}"
        json_upload = r_upload.json()
        assert "columns" in json_upload, "No columns in upload response"
        print("PASS: File upload successful.")
        
        # Test 3: Analyze
        print("Testing Data Analysis...")
        r_analyze = requests.get("http://127.0.0.1:8000/analyze")
        assert r_analyze.status_code == 200, f"Analyze failed: {r_analyze.status_code}"
        print("PASS: Analysis successful.")
        
        # Test 4: Model Code Gen
        print("Testing Code Generation...")
        r_code = requests.post("http://127.0.0.1:8000/model_code", data={"target": "target"})
        assert r_code.status_code == 200, f"CodeGen failed: {r_code.status_code}"
        json_code = r_code.json()
        assert "code" in json_code, "No code generated"
        print("PASS: Model code generation successful.")
        
        print("\nALL TESTS PASSED!")
        
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        # Print logs for debugging
        print("\n--- FastAPI Logs ---")
        print(api_process.stdout.read().decode())
        print(api_process.stderr.read().decode())
        print("\n--- Flask Logs ---")
        print(web_process.stdout.read().decode())
        print(web_process.stderr.read().decode())

    finally:
        print("Terminating servers...")
        api_process.terminate()
        web_process.terminate()

if __name__ == "__main__":
    run_tests()
