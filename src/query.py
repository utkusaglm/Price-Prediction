import psycopg2
from sqlalchemy import create_engine,text
import pandas as pd
import config

HOST = config.DB_HOST
DATABASE = config.DB_NAME
USER = config.DB_USER
PASSWORD = config.DB_PASS
PORT = config.DB_PORT

QUERY = """select * from assets ORDER BY date ASC;"""
CREATE_TABLE = """
    CREATE TABLE assets(
  	name TEXT NOT NULL,
     date TIMESTAMP WITHOUT TIME ZONE NOT NULL,
     Open NUMERIC NOT NULL, 
     High NUMERIC NOT NULL,
     Low NUMERIC NOT NULL,
     Close NUMERIC NOT NULL, 
 	AdjClose NUMERIC NOT NULL, 
 	Volume NUMERIC NOT NULL,
 	PRIMARY KEY (name, date))
 	;
    """


class GetDataFromDatabase:
    """query functions"""
    def __call__(self):
        return self

    def __init__(self,query) -> None:
        self.query = query
        self.engine = create_engine(f'postgresql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}')
        self.connection = psycopg2.connect(f"postgres://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}")

get_data = GetDataFromDatabase(query=QUERY)
try:
    get_data.engine.execute(text(CREATE_TABLE))
except Exception as e:
    print(e)
df = pd.read_sql_query(get_data.query, con=get_data.engine)