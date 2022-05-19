# price-prediction
# Preconditions
- docker pull timescale/timescaledb-ha:pg14-latest
- docker run -d --name timescaledb -p 5432:5432 -e POSTGRES_PASSWORD=password timescale/timescaledb:latest-pg14
- docker exec -it timescaledb psql -U postgres
- run sql commands that inside the data-related/create_tables.sql
## Second Step
- - - - - 
- cd src
- python3 -m venv venv
- pip3 install -r requirments.txt
- mlflow server \
 --backend-store-uri sqlite:///mlflow.db \
 --default-artifact-root ./mlflow-artifact-root \
 --host 0.0.0.0

 - open mlflow and create two experiment: random_f, logistic_f

- streamlit run stream_serve.py

# GOAL
- making a complete solution for price prediction problem.