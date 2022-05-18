from statistics import mode
import warnings
warnings.filterwarnings('ignore')
from sqlalchemy import create_engine
import psycopg2
import psycopg2.extras
import config
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
from mlflow import pyfunc
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from data_related import populate_data
from save_and_train_model import train_model
import mlflow.sklearn

TRACKING_URL = "http://localhost:5000"

# query = """select * from assets ORDER BY date ASC;"""
# connection = psycopg2.connect(host=config.DB_HOST, database=config.DB_NAME, user= config.DB_USER, password=config.DB_PASS)
# cursor = connection.cursor(cursor_factory=psycopg2.extras.DictCursor)
# engine = create_engine(f'postgresql://{config.DB_USER}:{config.DB_PASS}@{config.DB_HOST}:{config.DB_PORT}/{config.DB_NAME}')


# df = pd.read_sql_query(query,con=engine)
# SYMBOLS = ['MSFT','AAPL','NVDA','UBER']
# IS_INVESTED = False

def insert_new_data():
    populate_data.insert_data()

def train_model_and_save():
    return train_model()

def serve_model(model_type, symbol):
    client = MlflowClient(registry_uri=TRACKING_URL)
    mlflow.set_tracking_uri(TRACKING_URL)
    if model_type == 'r_f':
        symbol += 'r_f'
        timestamp,source = str(client.get_latest_versions(symbol)[0]).split(',')[3], str(client.get_latest_versions(symbol)[0]).split(',')[7]
        timestamp = timestamp.split('=')[1]
        source = source.split('=')[1].replace("'","")
    elif model_type == 'l_r':
        symbol += 'l_r'
        timestamp,source = str(client.get_latest_versions(symbol)[0]).split(',')[3], str(client.get_latest_versions(symbol)[0]).split(',')[7]
        timestamp = timestamp.split('=')[1]
        source = source.split('=')[1].replace("'","")
    model = mlflow.sklearn.load_model(source)
    
    return model
    


# if __name__ == '__main__':