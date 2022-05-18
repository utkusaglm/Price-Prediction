import time
import yfinance as yf
import datetime
import psycopg2
import psycopg2.extras
from sqlalchemy import create_engine
import config

STOCKS = ["MSFT","AAPL","GOOG",'NVDA','UBER','COIN']
# CRYPTO = ["BTC-USD","ETH-USD"]
ASSETS = STOCKS #+ CRYPTO
connection = psycopg2.connect(host=config.DB_HOST, database=config.DB_NAME, user= config.DB_USER, password=config.DB_PASS)
cursor = connection.cursor(cursor_factory=psycopg2.extras.DictCursor)
engine = create_engine(f'postgresql://{config.DB_USER}:{config.DB_PASS}@{config.DB_HOST}:{config.DB_PORT}/{config.DB_NAME}')

TODAY = time.time()
INTERVAL = '5m'

def get_data(name, st, et, interval):
    """
    Get the data
    """
    return yf.download(name, start= st, end= et , interval=interval)

def insert_data():
    """insert the data."""
    global TODAY, INTERVAL, engine,cursor
    #Insert data in every week
    cursor.execute("""
    select exists(select 1 from assets);             
    """
    )
    fetched = cursor.fetchone()
    try:
        if not fetched[0]:
            for asset in ASSETS:
                start = TODAY - 4665600 #360d
                end = start+ 86400 #1d
                print(asset)           
                while end<TODAY:
                    start_datetime = datetime.datetime.fromtimestamp(start).strftime('%Y-%m-%d')
                    end_datetime = datetime.datetime.fromtimestamp(end).strftime('%Y-%m-%d')
                    data = get_data(asset,start_datetime,end_datetime,INTERVAL).iloc[:-1]
                    data["name"] = asset
                    data = data.rename(columns={'Open': 'open','High': 'high', 'Low': 'low','Close': 'close', 'Adj Close': 'adjclose','Volume': 'volume'})
                    print(data)
                    data["date"] = data.index
                    data.to_sql('assets',engine,if_exists='append',index=False)
                    start = end 
                    end += 86400
        cursor.execute("""
            SELECT extract(epoch from date) as last_date
            FROM assets
            ORDER BY date DESC
            LIMIT 1;       
            """)
    except Exception as e:
        # print(e)
        pass
    try:
        fetched =cursor.fetchone()
        fetched_control = fetched[0]+86400
        if fetched_control < TODAY:
            for asset in ASSETS:
                start = TODAY - 604800 #720d
                end = start+ 86400 #1d
                print(asset)
                while end <TODAY:
                    start_datetime=datetime.datetime.fromtimestamp(start).strftime('%Y-%m-%d')
                    end_datetime=datetime.datetime.fromtimestamp(end).strftime('%Y-%m-%d')
                    data= get_data(asset,start_datetime,end_datetime,INTERVAL).iloc[:-1]
                    data["name"] = asset
                    data =data.rename(columns={'Open': 'open','High': 'high', 'Low': 'low','Close': 'close', 'Adj Close': 'adjclose','Volume': 'volume'})
                    print(data)
                    data["date"]=data.index
                    data.to_sql('assets',engine,if_exists='append',index=False)
                    start 
                    end += 86400
    except Exception as e:
        pass
        # print(e)
if __name__ == '__main__':
    a =time.time()
    insert_data()
    b =time.time()
    print(b-a)
