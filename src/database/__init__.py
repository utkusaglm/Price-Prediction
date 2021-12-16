
#import configs
from sqlalchemy.sql.expression import insert, table
from sqlalchemy import TIMESTAMP,ARRAY,Integer,Boolean,DateTime,Enum
from sqlalchemy.sql import func
from sqlalchemy import create_engine,Column
from sqlalchemy.orm import session, sessionmaker
from sqlalchemy.sql.sqltypes import INTEGER
from sqlalchemy_utils import database_exists,create_database
from sqlalchemy.ext.declarative import declarative_base
import traceback
import datetime
import logging
from .config import DATABASE_USER,DATABASE_PASSWORD,DATABASE_NAME,DATABASE_HOST,DATABASE_DEFAULT_NAME,DATABASE_PORT

USER = DATABASE_USER
HOST = DATABASE_HOST
NAME = DATABASE_NAME
PASSWD = DATABASE_PASSWORD
PORT = DATABASE_PORT

BASE = declarative_base()

import enum




#TODO:AYRI DOSYA
class LOG(Enum):
    def info(msg, content=None):
        logging.basicConfig(filename='app.log', filemode='a',format='%(asctime)s - %(message)s', level=logging.INFO)
        if content is not None:
            logging.info("Info: {info}, Content: {content} ".format(info=msg, content=content))
        else:
            logging.info("Info: {info} ".format(info=msg))

    
    def error(msg, content=None):
        logging.basicConfig(filename='app.log', filemode='a',format='%(asctime)s - %(message)s', level=logging.ERROR)
        if content is not None:
            logging.error("Error: {error}, Content: {content} ".format(error=msg, content=content))
        else:    
            logging.error("Error: {error} ".format(error=msg))



def get_engine():
      global USER,HOST,NAME,PASSWD,PORT
      url =f"postgresql://{USER}:{PASSWD}@{HOST}:{PORT}/{NAME}"
      if not database_exists(url):
            #TODO: TableCreator
            create_database(url)
      engine = create_engine(url,pool_size=20,echo=False)
      return engine

def get_session():
      engine = get_engine()
      session = sessionmaker(bind=engine)()
      return session
engine = get_engine()
            
class Interval(enum.Enum):
    c10 = "c10";
    c150 = "c150";
    m10 = "m10";
    h2 = "3";

class PhaseNames(enum.Enum):
    NE = "NE";
    L1N = "L1N";
    L2N = "L2N";
    L3N = "L3N";

class EventTypes(enum.Enum):
      SNAPSHOT = "SNAPSHOT"
      VOLTAGE_DIP = "VOLTAGE_DIP"
      VOLTAGE_SWELL = "VOLTAGE_SWELL"
      INTERRUPTION = "INTERRUPTION"
      
class BaseTable(BASE):
      __tablename__ ="base_table"
      insert_time = Column('insert_time',DateTime(timezone=False), server_default=func.now(),primary_key=True)
      device_id = Column('device_id',Integer)
      calc_time = Column('calc_time',DateTime)
      interval = Column("interval",Enum(Interval))
      frequency = Column("frequency",ARRAY(Integer))
      u_zero = Column("u_zero",Integer)
      u_negative = Column("u_negative",Integer)
      msv = Column("msv",Integer)
      flag = Column("flag",Boolean)
      frequency_avg = Column("frequency_avg",Integer)
      def __init__(self,kwargs):
            try:
                  self.insert_time = func.now()
                  self.device_id = kwargs["device_id"]
                  self.calc_time = datetime.datetime.utcfromtimestamp(kwargs["calc_time"])
                  self.interval = kwargs["interval"]
                  self.frequency = kwargs["frequency"]
                  self.u_zero = kwargs["u_zero"]
                  self.u_negative = kwargs["u_negative"]
                  self.msv = kwargs["msv"]
                  self.flag = kwargs["flag"]
                  self.frequency_avg = kwargs["frequency_avg"]
            except Exception:
                  LOG.error(msg=traceback.format_exc())

class PhaseTable(BASE):
      __tablename__ ="phase_table"
      device_id = Column('device_id',Integer)
      calc_time = Column('calc_time',DateTime,primary_key=True)
      phase_name = Column('phase_name',Enum(PhaseNames))
      angle = Column('angle',Integer)
      v_rms = Column('v_rms',Integer)
      v_max = Column('v_max',Integer)
      v_min = Column('v_min',Integer)
      thd = Column('thd',Integer)
      pst = Column('pst',Integer)
      plt = Column('plt',Integer)
      def __init__(self,kwargs):
            try:
                  self.device_id = kwargs["device_id"]
                  self.calc_time = datetime.datetime.utcfromtimestamp(kwargs["calc_time"])
                  self.phase_name = kwargs["phase_name"]
                  self.angle = kwargs["angle"]
                  self.v_rms = kwargs["v_rms"]
                  self.v_max = kwargs["v_rms"]
                  self.v_min = kwargs["v_min"]
                  self.thd = kwargs["thd"]
                  self.pst = kwargs["pst"]
                  self.plt = kwargs["plt"]
            except Exception:
                  LOG.error(msg=traceback.format_exc())

class EventTable(BASE):
      __tablename__ ="event_table"
      insert_time = Column('insert_time',DateTime(timezone=False), server_default=func.now(),primary_key=True)
      device_id = Column('device_id',Integer)
      detected_phase = Column('detected_phase',Enum(PhaseNames))
      event_type = Column('event_type',Enum(EventTypes))
      raise_time = Column('raise_time',DateTime)
      clear_time = Column('clear_time',DateTime)
      observed_value_l1n = Column('observed_value_l1n',Integer)
      observed_value_l2n = Column('observed_value_l2n',Integer)
      observed_value_l3n = Column('observed_value_l3n',Integer)

      def __init__(self,kwargs):
            try:
                  self.insert_time = func.now()
                  self.device_id = kwargs["device_id"]
                  self.detected_phase = kwargs["detected_phase"]
                  self.event_type = kwargs["event_type"]
                  self.raise_time =  datetime.datetime.utcfromtimestamp(kwargs["raise_time"])
                  self.clear_time =  datetime.datetime.utcfromtimestamp(kwargs["clear_time"])
                  self.observed_value_l1n = kwargs["observed_value_l1n"]
                  self.observed_value_l2n = kwargs["observed_value_l2n"]
                  self.observed_value_l3n = kwargs["observed_value_l3n"]
            except:
                  LOG.error(msg=traceback.format_exc())
#TODO:column sayısı
class HarmonicTable(BASE):
      __tablename__ ="harmonic_table"
      device_id = Column('device_id',Integer)
      calc_time = Column('calc_time',DateTime,primary_key=True)
      h_type = Column('h_type',Enum(Interval))
      h0 = Column('h0',Integer)
      h1 = Column('h1',Integer)
      h2 = Column('h2',Integer)
      h3 = Column('h3',Integer)
      h4 = Column('h4',Integer)
      h5 = Column('h5',Integer)
      h6 = Column('h6',Integer)
      h7 = Column('h7',Integer)
      h8 = Column('h8',Integer)
      h9 = Column('h9',Integer)
      h10 = Column('h10',Integer)
      h11 = Column('h11',Integer)
      h12 = Column('h12',Integer)
      h13 = Column('h13',Integer)
      h14 = Column('h14',Integer)
      h15 = Column('h15',Integer)
      h16 = Column('h16',Integer)
      h17 = Column('h17',Integer)
      h18 = Column('h18',Integer)
      h19 = Column('h19',Integer)
      h20 = Column('h20',Integer)
      h21 = Column('h21',Integer)
      h22 = Column('h22',Integer)
      h23 = Column('h23',Integer)
      h24 = Column('h24',Integer)
      h25 = Column('h25',Integer)
      h26 = Column('h26',Integer)
      h27 = Column('h27',Integer)
      h28 = Column('h28',Integer)
      h29 = Column('h29',Integer)
      h30 = Column('h30',Integer)
      h31 = Column('h31',Integer)
      h32 = Column('h32',Integer)
      h33 = Column('h33',Integer)
      h34 = Column('h34',Integer)
      h35 = Column('h35',Integer)
      h36 = Column('h36',Integer)
      h37 = Column('h37',Integer)
      h38 = Column('h38',Integer)
      h39 = Column('h39',Integer)
      h40 = Column('h40',Integer)
      h41 = Column('h41',Integer)
      h42 = Column('h42',Integer)
      h43 = Column('h43',Integer)
      h44 = Column('h44',Integer)
      h45 = Column('h45',Integer)
      h46 = Column('h46',Integer)
      h47 = Column('h47',Integer)
      h48 = Column('h48',Integer)
      h49 = Column('h49',Integer)

      ##TODO:TURN THIS TO *args
      #self,device_id,calc_time,h_type,h0=None,h1=None,h2=None,h3=None,h4=None,h5=None,h6=None,h7=None,h8=None,h9=None,h10=None,h11=None,h12=None,h13=None,h14=None,h15=None,h16=None,h17=None,h18=None,h19=None,h20=None,h21=None,h22=None,h23=None,h24=None,h25=None,h26=None,h27=None,h28=None,h29=None,h30=None,h31=None,h32=None,h33=None,h34=None,h35=None,h36=None,h37=None,h38=None,h39=None,h40=None,h41=None,h42=None,h43=None,h44=None,h45=None,h46=None,h47=None,h48=None,h49=None
      def __init__(self,*args):
            try:      
                  self.device_id = args[0][0]
                  self.calc_time = datetime.datetime.utcfromtimestamp(args[0][1])
                  self.h_type = args[0][2]
                  self.h0 = args[0][3]
                  self.h1 = args[0][4]
                  self.h2 = args[0][5]
                  self.h3 = args[0][6]
                  self.h4 = args[0][7]
                  self.h5 = args[0][8]
                  self.h6 = args[0][9]
                  self.h7 = args[0][10]
                  self.h8 = args[0][11]
                  self.h9 = args[0][12]
                  self.h10 =args[0][12]
                  self.h11 =args[0][13]
                  self.h12 =args[0][14]
                  self.h13 =args[0][15]
                  self.h14 =args[0][16]
                  self.h15 =args[0][17]
                  self.h16 =args[0][18]
                  self.h17 =args[0][19]
                  self.h18 =args[0][20]
                  self.h19 =args[0][21]
                  self.h20 =args[0][22]
                  self.h21 =args[0][23]
                  self.h22 =args[0][24]
                  self.h23 =args[0][25]
                  self.h24 =args[0][26]
                  self.h25 =args[0][27]
                  self.h26 =args[0][28]
                  self.h27 =args[0][29]
                  self.h28 =args[0][30]
                  self.h29 =args[0][31]
                  self.h30 =args[0][32]
                  self.h31 =args[0][33]
                  self.h32 =args[0][34]
                  self.h33 =args[0][35]
                  self.h34 =args[0][36]
                  self.h35 =args[0][37]
                  self.h36 =args[0][38]
                  self.h37 =args[0][39]
                  self.h38 =args[0][40]
                  self.h39 =args[0][41]
                  self.h40 =args[0][42]
                  self.h41 =args[0][43]
                  self.h42 =args[0][44]
                  self.h43 =args[0][45]
                  self.h44 =args[0][46]
                  self.h45 =args[0][47]
                  self.h46 =args[0][48]
                  self.h47 =args[0][49]
                  self.h48 =args[0][50]
                  self.h49 =args[0][51]
            except:
                  pass

def insert_data(table_name,values):
      if 'event' in table_name:
            object_ = EventTable(values) 
      elif 'base' in table_name:
            object_ = BaseTable(values)
      elif 'phase' in table_name:
            object_ = PhaseTable(values)
      elif 'harmonic' in table_name:
            object_ = HarmonicTable(values)
      session =get_session()
      session.add(object_)
      try:
            session.commit()
      except Exception:
            LOG.error(msg=traceback.format_exc())
            session.close()
      finally:
            session.close()