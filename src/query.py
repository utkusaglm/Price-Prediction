import psycopg2
from sqlalchemy import create_engine
import pandas as pd
import config

HOST = config.DB_HOST
DATABASE = config.DB_NAME
USER = config.DB_USER
PASSWORD = config.DB_PASS
PORT = config.DB_PORT
QUERY = """select * from assets ORDER BY date ASC;"""

class GetDataFromDatabase:
    """query functions"""
    
    def __call__(self):
        return self
    
    def __init__(self,query) -> None:
        self.query = query
        self.engine = create_engine(f'postgresql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}')
        self.connection = psycopg2.connect(host=HOST, database=DATABASE, user= USER, password=PASSWORD)

get_data = GetDataFromDatabase(query=QUERY)

df = pd.read_sql_query(get_data.query, con=get_data.engine)