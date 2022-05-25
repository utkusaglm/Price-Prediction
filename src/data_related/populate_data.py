import time
import datetime
import yfinance as yf
import psycopg2
import psycopg2.extras
import config
from query import get_data

ASSETS = config.SYMBOLS
conn = get_data.connection
cur= conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
eng = get_data.engine
INTERVAL = config.INTERVAL

class Yfinance:
    """yfinance library"""
    def __init__(self,connection,cursor,engine,today_unix,interval,assets):
        self.connection = connection
        self.cursor = cursor
        self.engine = engine
        self.today_unix = today_unix
        self.interval = interval
        self.assets = assets

    @staticmethod
    def download_data(name, st, et, interval):
        """ download data from yfinance """
        return yf.download(name, start= st, end= et , interval=interval)

    def insert_data(self,start_day,daily,start_day_for_control):
        """add data to database"""
        try:
            #Insert data in every week
            self.cursor.execute("""
            select exists(select 1 from assets);             
            """
            )
            fetched = self.cursor.fetchone()
            if not fetched[0]:
                for asset in self.assets:
                    __import__('ipdb').set_trace()
                    start = self.today_unix - start_day #60d
                    end = start + daily  #1d
                    print(asset)          
                    while end<self.today_unix:
                        start_datetime = datetime.datetime.fromtimestamp(start).strftime('%Y-%m-%d')
                        end_datetime = datetime.datetime.fromtimestamp(end).strftime('%Y-%m-%d')
                        data = self.download_data(asset,start_datetime,end_datetime,INTERVAL).iloc[:-1]
                        if(len(data)!=0):
                            data["name"] = asset
                            data = data.rename(columns={'Open': 'open','High': 'high', 'Low': 'low','Close': 'close', 'Adj Close': 'adjclose','Volume': 'volume'})
                            print(data)
                            data["date"] = data.index
                            data.to_sql('assets',self.engine,if_exists='append',index=False)
                        start += daily
                        end += daily
            self.cursor.execute("""
                SELECT extract(epoch from date) as last_date
                FROM assets
                ORDER BY date DESC
                LIMIT 1;       
                """)
            fetched =self.cursor.fetchone()
            fetched_control = fetched[0]+86400
            if fetched_control < self.today_unix:
                __import__('ipdb').set_trace()
                for asset in self.assets:
                    start = self.today_unix - start_day_for_control #7d
                    end = start+ daily #1d
                    print(asset)
                    while end <self.today_unix:
                        start_datetime=datetime.datetime.fromtimestamp(start).strftime('%Y-%m-%d')
                        end_datetime=datetime.datetime.fromtimestamp(end).strftime('%Y-%m-%d')
                        data= get_data(asset,start_datetime,end_datetime,INTERVAL).iloc[:-1]
                        if len(data) != 0:
                            data["name"] = asset
                            data =data.rename(columns={'Open': 'open','High': 'high', 'Low': 'low','Close': 'close', 'Adj Close': 'adjclose','Volume': 'volume'})
                            print(data)
                            data["date"]=data.index
                            data.to_sql('assets',self.engine,if_exists='append',index=False)
                        start += daily
                        end += daily
        except Exception as e:
            pass

def populate_data(connection,cursor,engine,today_unix,interval,assets,start_day,daily,start_day_for_control):
    """ populate data"""
    y_finance = Yfinance(connection,cursor,engine,today_unix,interval,assets)
    #start_day = 5097600     TODAY- start_day
    #daily = 86400           download day by day
    #start_day_for_control = 604800 download last week
    y_finance.insert_data(start_day,daily,start_day_for_control)

if __name__ == '__main__':
    #start_day = 5097600     TODAY- start_day
    #daily = 86400           download day by day
    #start_day_for_control = 604800 start download from last week
    a =time.time()
    populate_data(conn,cur,eng,time.time(),INTERVAL,ASSETS,5097600,86400,604800)
    b =time.time()
    print(b-a)
