import mlflow
import mlflow.sklearn
import config
from data_related import populate_data
from train_and_save_model import train_model
from model import Model

TRACKING_URL = config.TRACKING_URL
model_c = Model()

def insert_new_data():
    """if one week passed from last download run"""
    populate_data.populate_data()

def train_model_and_save():
    """train model"""
    return train_model()

def serve_model(model_type, symbol):
    """serve_model"""
    client = model_c.client
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
