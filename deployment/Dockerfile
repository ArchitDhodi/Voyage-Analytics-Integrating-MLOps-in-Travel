FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV API_PORT=5000 STREAMLIT_PORT=8501 MLFLOW_PORT=5005 MODEL_PATH=/app/models/trained

CMD ["bash", "-c", "python api/app.py --port ${API_PORT}"]