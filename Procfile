worker: mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow-artifact-root --host 0.0.0.0 --port=${PORT:-5000}
web: sh setup.sh && streamlit run src/stream_serve.py