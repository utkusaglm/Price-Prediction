import os
from dotenv import load_dotenv
load_dotenv()
DB_HOST = os.getenv('DB_HOST')
DB_USER = os.getenv('DB_USER')
DB_PASS = os.getenv('DB_PASS')
DB_NAME = os.getenv('DB_NAME')
DB_PORT = os.getenv('DB_PORT')
TRACKING_URL = f"http://localhost:{os.getenv('MLFLOW_PORT')}"
SYMBOLS=['MSFT','AAPL','NVDA','UBER']
INTERVAL = '5m'
