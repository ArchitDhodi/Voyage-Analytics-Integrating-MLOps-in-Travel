import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def run_cmd(cmd, cwd=None):
    try:
        subprocess.run(cmd, cwd=cwd or ROOT, check=True, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {cmd}")
        sys.exit(1)


def start_dev():
    """
    Start API + Streamlit (and MLflow server) without helper scripts.
    Runs three subprocesses. Stop with Ctrl+C.
    """
    processes = []
    try:
        mlflow_port = os.getenv("MLFLOW_PORT", "5005")
        mlflow_backend = os.getenv(
            "MLFLOW_TRACKING_URI",
            f"file://{(ROOT / 'mlruns_local').resolve()}",
        )
        api_port = os.getenv("API_PORT", "5000")
        streamlit_port = os.getenv("STREAMLIT_PORT", "8501")

        # Start MLflow server (optional but matches previous behavior)
        mlflow_cmd = f'mlflow server --host 0.0.0.0 --port {mlflow_port} --backend-store-uri "{mlflow_backend}"'
        processes.append(subprocess.Popen(mlflow_cmd, cwd=ROOT, shell=True))

        # Start Flask API
        api_cmd = f"{sys.executable} api/app.py --port {api_port}"
        processes.append(subprocess.Popen(api_cmd, cwd=ROOT, shell=True))

        # Start Streamlit dashboard
        streamlit_cmd = f"streamlit run web/dashboard.py --server.port {streamlit_port} --server.address 0.0.0.0"
        processes.append(subprocess.Popen(streamlit_cmd, cwd=ROOT, shell=True))

        print("\nServices starting...")
        print(f"- API: http://localhost:{api_port}")
        print(f"- Dashboard: http://localhost:{streamlit_port}")
        print(f"- MLflow: http://localhost:{mlflow_port}")
        print("Press Ctrl+C to stop.")
        for p in processes:
            p.wait()
    except KeyboardInterrupt:
        print("\nStopping services...")
        for p in processes:
            try:
                p.terminate()
            except Exception:
                pass
        for p in processes:
            try:
                p.wait(timeout=5)
            except Exception:
                p.kill()
    except Exception as e:
        print(f"Error starting services: {e}")
        for p in processes:
            try:
                p.terminate()
            except Exception:
                pass


def start_full():
    print("\nBuilding and starting full stack via docker-compose (app + Airflow)...")
    run_cmd("docker-compose up -d", cwd=ROOT / "deployment")
    run_cmd("docker-compose up -d", cwd=ROOT / "pipelines" / "airflow")
    api_port = os.getenv("API_PORT", "5000")
    streamlit_port = os.getenv("STREAMLIT_PORT", "8501")
    mlflow_port = os.getenv("MLFLOW_PORT", "5005")
    airflow_port = os.getenv("AIRFLOW_PORT", "8081")
    print("Services:")
    print(f"- API health: http://localhost:{api_port}/health")
    print(f"- Dashboard: http://localhost:{streamlit_port}")
    print(f"- MLflow: http://localhost:{mlflow_port}")
    print(f"- Airflow: http://localhost:{airflow_port}")


def main():
    print("\n=== ABMSM Beginner Setup ===")
    print("1) Start (local) dev stack (API + Streamlit + MLflow)")
    print("2) Start full stack (Docker Compose)")
    print("3) Exit")
    choice = input("Choose an option: ").strip()
    if choice == "1":
        start_dev()
    elif choice == "2":
        start_full()
    else:
        print("Bye!")


if __name__ == "__main__":
    main()
