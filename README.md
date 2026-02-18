# ABMSM Beginner Flight/MLOps Stack

Beginner-friendly end-to-end MLOps project for travel analytics: flight price regression, gender classification, and hotel recommendations, served via a Flask API and Streamlit dashboard, tracked with MLflow, deployed with Docker/K8s, and orchestrated with Airflow.

## Prerequisites
- Python 3.9+ (tested with 3.9)
- Docker Desktop (for Docker/K8s/Airflow). Ensure it is running before `docker-compose` commands.

## Quickstart (local)
1) Create venv and install:
```
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```
2) Put real CSVs in `data/`:
- `data/flights.csv`
- `data/users.csv`
- `data/hotels.csv`
3) Start dev stack:
- Preferred (reads `.env`):
```
python setup.py
```
Choose option 1.
- Or:
```
python scripts/run_dev.py
```
If using `scripts/run_dev.py`, set `MLFLOW_TRACKING_URI` in your shell to point at `mlruns_local`.

Services:
- API: http://localhost:5000
- Dashboard: http://localhost:8501
- MLflow: http://localhost:5005

## Objectives checklist
- Regression model (flight price) in `notebooks/abmsm_all_in_one.ipynb`, artifact `models/trained/price_model.pkl`
- REST API in `api/app.py`
- Docker in `deployment/docker-compose.yml` and `deployment/Dockerfile`
- Kubernetes manifests in `deployment/kubernetes/`
- Airflow DAG in `pipelines/airflow/dags/flight_price_pipeline.py`
- MLflow tracking in `mlruns_local/`
- Gender classification in `notebooks/abmsm_all_in_one.ipynb`, artifact `models/trained/gender_model.pkl`
- Hotel recommendation in `notebooks/abmsm_all_in_one.ipynb`, artifact `models/trained/recommender.pkl`, UI in `web/dashboard.py`

## Project structure
- `api/`:
  - `api/app.py` (Flask API: `/health`, `/predict`, `/predict_batch`, `/gender`, `/recommend`)
- `data/`:
  - `data/flights.csv`, `data/users.csv`, `data/hotels.csv`
- `deployment/`:
  - `deployment/Dockerfile`
  - `deployment/docker-compose.yml`
  - `deployment/kubernetes/` (K8s manifests)
- `mlflow/`: reserved placeholder (can remain empty)
- `mlruns_local/`: MLflow file store (runs, params, metrics, artifacts)
- `models/`:
  - `models/trained/price_model.pkl`
  - `models/trained/gender_model.pkl`
  - `models/trained/recommender.pkl`
- `notebooks/`:
  - `notebooks/abmsm_all_in_one.ipynb`
- `pipelines/`:
  - `pipelines/airflow/docker-compose.yml`
  - `pipelines/airflow/dags/flight_price_pipeline.py`
- `scripts/`:
  - `scripts/run_dev.py`
  - `scripts/train_price.py`
  - `scripts/train_price_mlflow.py`
  - `scripts/train_gender.py`
  - `scripts/train_recommender.py`
  - `scripts/train_all.py`
- `web/`:
  - `web/dashboard.py` (Streamlit UI)
- `.env` (local settings)
- `requirements.txt`
- `setup.py`

## Environment variables (.env)
```
API_PORT=5000
STREAMLIT_PORT=8501
MLFLOW_PORT=5005
API_URL=http://localhost:5000
MODEL_PATH=models/trained
MLFLOW_TRACKING_URI=file:///C:/Users/archi/New folder/abmsm_beginner/mlruns_local
ENABLE_MLFLOW=true
ENABLE_AIRFLOW=false
ENABLE_K8S=false
```
Update `MLFLOW_TRACKING_URI` if your path differs.

## Training
- Notebook: run `notebooks/abmsm_all_in_one.ipynb` top-to-bottom.
- Scripts:
```
python scripts/train_price.py
python scripts/train_gender.py
python scripts/train_recommender.py
```
Artifacts are written to `models/trained/`.

### API payloads
- `/predict` expects `source,destination,flight_type,time,distance,agency,date`.
- `/gender` accepts `name`, `company`, and `age` (preferred), or legacy `features` list.
- `/recommend` expects query params `index` and `n`.

## Docker
```
cd deployment
docker-compose up -d
```
Runs API, Streamlit, and MLflow. The compose file mounts local `data/`, `models/trained/`, `api/`, `web/`, and `mlruns_local/`.
Stop services:
```
docker-compose down
```

## Airflow
```
cd pipelines/airflow
docker-compose up -d
```
Airflow UI: http://localhost:8081 (default `admin/admin`). DAG: `flight_price_pipeline`.
Stop Airflow:
```
docker-compose down
```
If login is inconsistent, reset once:
```
docker-compose down -v
docker-compose up -d
```

## Kubernetes (optional)
```
kubectl apply -f deployment/kubernetes/
kubectl port-forward svc/abmsm-api 5000:5000
kubectl port-forward svc/abmsm-dashboard 8501:80
```
Update image name in the Deployment before applying.

## MLflow
- File store: `mlruns_local/`
- UI (local):
```
mlflow server --host 0.0.0.0 --port 5005 --backend-store-uri "file:///C:/Users/archi/New folder/abmsm_beginner/mlruns_local"
```
Open http://localhost:5005 and select the experiment to view runs.

## How to Use (end-to-end)
### 1) Generate model artifacts (one-time or whenever you retrain)
1) Open `notebooks/abmsm_all_in_one.ipynb`.
2) Run the notebook top-to-bottom.
3) This creates/overwrites:
   - `models/trained/price_model.pkl`
   - `models/trained/gender_model.pkl`
   - `models/trained/recommender.pkl`
4) MLflow runs are logged to `mlruns_local/` (make sure the path in `.env` is correct).

### 2) Run the app with `setup.py`
From the project root:
```
python setup.py
```
You will see:
```
1) Start (local) dev stack (API + Streamlit + MLflow)
2) Start full stack (Docker Compose + Airflow)
3) Exit
```

#### Option 1: Local dev stack (API + Streamlit + MLflow)
Steps:
1) Ensure your virtualenv is active and `.env` exists.
2) Run `python setup.py` and choose `1`.
3) Services will start:
   - API health: http://localhost:5000/health
   - Dashboard: http://localhost:8501
   - MLflow UI: http://localhost:5005

Notes:
- If you run `scripts/run_dev.py` instead, set `MLFLOW_TRACKING_URI` in the shell (or rely on `.env`).

#### Option 2: Full stack (Docker Compose + Airflow)
Before running:
1) Start Docker Desktop (Linux containers).
2) Ensure ports 5000, 8501, 5005, and 8081 are free.
3) If a local MLflow server is already running on 5005, stop it first.

Run:
```
python setup.py
```
Choose `2`. This will run:
- `deployment/docker-compose.yml` (API, Streamlit, MLflow)
- `pipelines/airflow/docker-compose.yml` (Airflow)

Services:
- API health: http://localhost:5000/health
- Dashboard: http://localhost:8501
- MLflow UI: http://localhost:5005
- Airflow UI: http://localhost:8081 (default admin/admin)

Stop services:
```
cd deployment
docker-compose down
cd ..\pipelines\airflow
docker-compose down
```

Stop services (from anywhere, with full paths):
```
cd "C:\Users\archi\New folder\ABSM Project 1\deployment"
docker-compose down
cd "C:\Users\archi\New folder\ABSM Project 1\pipelines\airflow"
docker-compose down
```

### 3) Kubernetes (optional)
Before running:
1) Enable Kubernetes in Docker Desktop.
2) Install `kubectl` and verify: `kubectl get nodes`.
3) Stop local Docker Compose stack to avoid port conflicts.

Run:
```
kubectl apply -f deployment/kubernetes/
kubectl port-forward svc/abmsm-api 5000:5000
kubectl port-forward svc/abmsm-dashboard 8501:80
```
Then open:
- API health: http://localhost:5000/health
- Dashboard: http://localhost:8501
