"""
Start API + Streamlit + MLflow locally.
Stop with Ctrl+C.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    api_port = os.getenv("API_PORT", "5000")
    streamlit_port = os.getenv("STREAMLIT_PORT", "8501")
    mlflow_port = os.getenv("MLFLOW_PORT", "5005")

    processes = []
    try:
        # MLflow server (optional)
        mlflow_cmd = f"mlflow server --host 0.0.0.0 --port {mlflow_port}"
        processes.append(subprocess.Popen(mlflow_cmd, cwd=ROOT, shell=True))

        api_cmd = f"{sys.executable} api/app.py --port {api_port}"
        processes.append(subprocess.Popen(api_cmd, cwd=ROOT, shell=True))

        streamlit_cmd = f"streamlit run web/dashboard.py --server.port {streamlit_port} --server.address 0.0.0.0"
        processes.append(subprocess.Popen(streamlit_cmd, cwd=ROOT, shell=True))

        print(f"API: http://localhost:{api_port}")
        print(f"Dashboard: http://localhost:{streamlit_port}")
        print(f"MLflow: http://localhost:{mlflow_port}")
        print("Press Ctrl+C to stop.")
        for p in processes:
            p.wait()
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        for p in processes:
            if p and p.poll() is None:
                p.terminate()
        for p in processes:
            if p and p.poll() is None:
                p.kill()


if __name__ == "__main__":
    main()
