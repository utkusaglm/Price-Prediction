import time
import numpy as np
from requests.sessions import InvalidSchema
import yfinance as yf
import time
import datetime

import psycopg2
import psycopg2.extras

from sqlalchemy import create_engine

STOCKS = ["MSFT","AAPL","GOOG"]
CRYPTO = ["BTC-USD","ETH-USD"]
ASSETS = STOCKS + CRYPTO

#TODO:CONFIG DISARI ALINICAK
#TODO: ASYNCPG İLE YAPILICAK
class Config:
    DB_HOST ='localhost'
    DB_USER ='postgres'
    DB_PASS ='password'
    DB_NAME ='prices'
    API_URL = "https://paper-api.alpaca.markets"
    API_KEY  = "PKTE1915P0HLD2R6B8E6"
    API_SECRET ="OtIa186qJPsGCPzYTryYJI74dkMq5xEEhs0bJNZP"
    DB_PORT ="5432"

config=Config()
connection=psycopg2.connect(host=config.DB_HOST, database=config.DB_NAME, user= config.DB_USER, password=config.DB_PASS)
cursor = connection.cursor(cursor_factory=psycopg2.extras.DictCursor)
engine = create_engine(f'postgresql://{config.DB_USER}:{config.DB_PASS}@{config.DB_HOST}:{config.DB_PORT}/{config.DB_NAME}')

TODAY = time.time() 
INTERVAL = '1d'


def get_data(name,st,et,interval):
    return yf.download(name, start= st, end= et , interval=interval)

def insert_data():
    global TODAY,INTERVAL,engine,cursor
    cursor.execute("""
    select exists(select 1 from assets);             
    """   
    )
    fetched =cursor.fetchone()
    if not fetched[0]:
        for asset in ASSETS:
            start = TODAY - 31104000 #360d
            end = start+ 604800 #7d
            print(asset)
            while end<TODAY:
                start_datetime=datetime.datetime.fromtimestamp(start).strftime('%Y-%m-%d')
                end_datetime=datetime.datetime.fromtimestamp(end).strftime('%Y-%m-%d')
                data= get_data(asset,start_datetime,end_datetime,INTERVAL)
                data["name"] = asset
                data =data.rename(columns={'Open': 'open','High': 'high', 'Low': 'low','Close': 'close', 'Adj Close': 'adjclose','Volume': 'volume'})
                print(data)
                data["date"]=data.index
                data.to_sql('assets',engine,if_exists='append',index=False)
                start = end + 86400
                end += 604800
    cursor.execute("""
        SELECT extract(epoch from date) as last_date
        FROM assets
        ORDER BY date DESC
        LIMIT 1;       
                   """)
    fetched =cursor.fetchone()
    fetched_control = fetched[0]+604800
    if fetched_control < TODAY:
        for asset in ASSETS:
            start = TODAY - 31104000 #360d
            end = start+ 604800 #7d
            print(asset)
            while end <TODAY:
                start_datetime=datetime.datetime.fromtimestamp(start).strftime('%Y-%m-%d')
                end_datetime=datetime.datetime.fromtimestamp(end).strftime('%Y-%m-%d')
                data= get_data(asset,start_datetime,end_datetime,INTERVAL)
                data["name"] = asset
                data =data.rename(columns={'Open': 'open','High': 'high', 'Low': 'low','Close': 'close', 'Adj Close': 'adjclose','Volume': 'volume'})
                print(data)
                data["date"]=data.index
                data.to_sql('assets',engine,if_exists='append',index=False)
                start += 604800
                end += 604800

if __name__ == '__main__':
    a =time.time()
    insert_data()
    b =time.time()
    print(b-a)