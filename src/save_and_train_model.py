import warnings
warnings.filterwarnings('ignore')
from sqlalchemy import create_engine
import psycopg2
import psycopg2.extras
import config
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
import random

import pickle
import time
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus

TRACKING_URL = "http://localhost:5000"

query = """select * from assets ORDER BY date ASC;"""
connection = psycopg2.connect(host=config.DB_HOST, database=config.DB_NAME, user= config.DB_USER, password=config.DB_PASS)
cursor = connection.cursor(cursor_factory=psycopg2.extras.DictCursor)
engine = create_engine(f'postgresql://{config.DB_USER}:{config.DB_PASS}@{config.DB_HOST}:{config.DB_PORT}/{config.DB_NAME}')



df = pd.read_sql_query(query,con=engine)
SYMBOLS = ['MSFT','AAPL','NVDA','UBER']
IS_INVESTED = False

def wait_model_transition(model_name, model_version, stage):
    """wait_model_transition"""
    client = MlflowClient(registry_uri=TRACKING_URL)
    for _ in range(10):
        model_version_details = client.get_model_version(name=model_name,
                                                         version=model_version,
                                                         )
        status = ModelVersionStatus.from_string(model_version_details.status)
        print("Model status: %s" % ModelVersionStatus.to_string(status))
        if status == ModelVersionStatus.READY:
            client.transition_model_version_stage(
              name=model_name,
              version=model_version,
              stage=stage,
            )
            break
        time.sleep(1)


def save_model(artifact_path, model, experiment_name, accuracy_train,accuracy_test,roc_auc):
    """save_model"""
    client = MlflowClient(registry_uri= TRACKING_URL)
    mlflow.set_tracking_uri(TRACKING_URL)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run() as run:
        run_num = run.info.run_id
        model_uri = "runs:/{run_id}/{artifact_path}".format(run_id=run_num, artifact_path=artifact_path)
        mlflow.log_metric('accuracy_train', accuracy_train)
        mlflow.log_metric('accuracy_test', accuracy_test)
        # mlflow.log_metrics('roc_auc_score',roc_auc)
        mlflow.sklearn.log_model(model, artifact_path)

        mlflow.register_model(model_uri=model_uri,
                                  name=artifact_path)
    model_version_infos = client.search_model_versions("name = '%s'" % artifact_path)
    new_model_version = max([model_version_info.version for model_version_info in model_version_infos])
    # Add a description
    client.update_model_version(
    name=artifact_path,
    version=new_model_version,
    description=experiment_name
    )
    # Necessary to wait to version models
    try:
        # Move the previous model to None version
        wait_model_transition(artifact_path, int(new_model_version)-1, "None")
    except:
        # Move the latest model to Staging (could also be Production)
        wait_model_transition(artifact_path, new_model_version, "Staging")
        
        
def get_rsi(df, rsi_period):
    """ get_rsi """
    chg = df['close'].diff(1)
    gain = chg.mask(chg<0,0)
    loss = chg.mask(chg>0,0)
    avg_gain = gain.ewm(com=rsi_period-1, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(com=rsi_period-1, min_periods=rsi_period).mean()
    rs = abs(avg_gain/avg_loss)
    rsi = 100 - (100/(1+rs))
    return rsi


def add_vwap(df):
    '''Returns dataframe with additional columns:
        vwap (float): Volume Weighted Average Price
        vwap_var (float): % variance of close from vwap
    Args:
        df (pandas.DataFrame): Dataframe with at least columns:
            datetime
            open
            high
            low
            adj_close
            volume
    Returns:
        df (pandas.DataFrame)
    '''
    df['vwap'] = (df['volume']*(df['high']+df['low']+df['adjclose'])/3).cumsum()/df['volume'].cumsum()
    df['vwap'] = df['vwap'].fillna(df['adjclose'])
    df['vwap_var'] = (df['adjclose']/df['vwap'])-1
    return df

def add_indicators(df):
    """ add_indicators """
    # relative strength index
    df['rsi14'] = get_rsi(df, 14)
    # moving averages
    df['sma9'] = df['close'].rolling(9).mean()
    df['sma180'] = df['close'].rolling(180).mean()
    df['sma9_var'] = (df['close']/df['sma9'])-1
    df['sma180_var'] = (df['close']/df['sma180'])-1
    # spreads
    df['spread']=((df['close']/df['open'])-1).abs()
    df['spread14_e']=df['spread'].ewm(span=14).mean()
    # volume-based indicator
    df['volume14'] = df['volume'].rolling(14).mean()
    df['volume34'] = df['volume'].rolling(34).mean()
    df['volume14_34_var'] = (df['volume14']/df['volume34'])-1
    df = add_vwap(df)
    return df


def drop_na_bf_fill(df):
    """drop_na_bf_fill"""
    df.dropna(axis=0,how='all',inplace=True)
    df.dropna(axis=1,how='any',inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    return df

def upside_down(df):
    """upside_down"""
    symbols = df['name'].unique()
    df_msft = df[df['name'] == 'MSFT']
    symbol = 'MSFT'
    final_df = pd.DataFrame(data=df_msft[['close','rsi14','sma9','sma180','sma9_var','sma180_var','spread','spread14_e','volume14','volume34','volume14_34_var','vwap','vwap_var']].to_numpy(), index = df_msft['date'],columns=[[f'{symbol}_Close',f'{symbol}_Rsi14',f'{symbol}_Sma9',f'{symbol}_Sma180',f'{symbol}_Sma9_var',f'{symbol}_Sma180_var',f'{symbol}_Spread',f'{symbol}_Spread14_e',f'{symbol}_Volume14',f'{symbol}_Volume34',f'{symbol}_Volume14_34_var',f'{symbol}_Vwap',f'{symbol}_vwap_var']])
    for symbol in symbols:
        if symbol != 'MSFT':
            df_sym = df[df['name'] == symbol]
            df_tmp = pd.DataFrame(data=df_sym[['close','rsi14','sma9','sma180','sma9_var','sma180_var','spread','spread14_e','volume14','volume34','volume14_34_var','vwap','vwap_var']].to_numpy(), index = df_sym['date'],columns=[[f'{symbol}_Close',f'{symbol}_Rsi14',f'{symbol}_Sma9',f'{symbol}_Sma180',f'{symbol}_Sma9_var',f'{symbol}_Sma180_var',f'{symbol}_Spread',f'{symbol}_Spread14_e',f'{symbol}_Volume14',f'{symbol}_Volume34',f'{symbol}_Volume14_34_var',f'{symbol}_Vwap',f'{symbol}_vwap_var']])
            final_df =final_df.join(df_tmp)
    return drop_na_bf_fill(final_df)

def return_prev(df,sym=SYMBOLS):
    """return_prev"""
    for symbol in sym:
        df[f'{symbol}_Prev_close']=df[f'{symbol}_Close'].shift(1)
        df[f'{symbol}_Return'] = np.array(df[f'{symbol}_Close']) / np.array(df[f'{symbol}_Prev_close']) - 1
        df[f'{symbol}_Log_Return'] = np.log(df[f'{symbol}_Return'] + 1)
    return df

def shifted_log_return(df,sym=SYMBOLS):
    """shifted_log_return"""
    for symbol in sym:
        df[f'{symbol}_Shifted_Log_Return'] = df[f'{symbol}_Log_Return'].shift(-1)
    return df

def train_columns(df,sym=SYMBOLS):
    """ train_columns """
    train_ready = {}
    for symbol in sym: 
        train_columns = [list(c)[0] for c in df.columns.values if symbol in str(c)]
        train_columns.extend([f'{c_name}_Shifted_Log_Return' for c_name in sym if c_name != symbol] )
        train_columns.remove(f'{symbol}_Shifted_Log_Return')
        train_ready[symbol] = train_columns
    return train_ready


def random_forest(df, t_columns, symbol):
    """ random_forest """
    df_v = df[t_columns]
    df_l =df[f'{symbol}_Shifted_Log_Return']
    Ntest = int(len(df_l)*70/100)
    train_msft = df_v.iloc[1:Ntest]
    test_msft = df_v.iloc[Ntest:-1]
    train_msft_l = df_l.iloc[1:Ntest]
    test_msft_l = df_l.iloc[Ntest:-1]
    Ctrain = (train_msft_l>0)
    Ctest = (test_msft_l>0)
    
    # load_model= pickle.load(open('./mlflow-artifact-root/2/00a7a15ca663466daaa2834abcf5f141/artifacts/random_forest/model.pkl','rb'))
    model = RandomForestClassifier(random_state=10)
    model.fit(train_msft, Ctrain)
    # print(f'Train_Score = {model.score(train_msft, Ctrain)}, Test_Score = {model.score(test_msft,Ctest)}')
    y_test_pp = model.predict_proba(test_msft)
    accuracy_train = model.score(train_msft, Ctrain)
    accuracy_test = model.score(test_msft, Ctest)
    roc_auc = roc_auc_score(Ctest, y_test_pp[:,1])
    # fpr, tpr, thres = roc_curve(Ctest, y_test_pp[:,1]
    symbol += 'r_f'
    fpr, tpr, thres = roc_curve(Ctest, y_test_pp[:,1])
    save_model(symbol,model,'random_f',accuracy_train,accuracy_test,roc_auc)
    return [accuracy_train,accuracy_test,roc_auc,fpr,tpr,thres]
    # ax = plt.plot(fpr, tpr)
    # plt.show()
    # train_algo = random_forest_df.iloc[1:Ntest][f'{random_c_c}_L_O_Algo_Return'].sum()
    # test_algo = random_forest_df.iloc[Ntest:-1][f'{random_c_c}_L_O_Algo_Return'].sum()
    # print(f' Train_Return:{train_algo}, Test_Return:{test_algo} ||  Buy_And_Holn_Train:{Ytrain.sum()}, Buy_And_Holn_Test:{Ytest.sum()}')

def logistic_reg(df, t_columns, symbol):
    """ random_forest """
    df_v = df[t_columns]
    df_l =df[f'{symbol}_Shifted_Log_Return']
    Ntest = int(len(df_l)*70/100)
    train_msft = df_v.iloc[1:Ntest]
    test_msft = df_v.iloc[Ntest:-1]
    train_msft_l = df_l.iloc[1:Ntest]
    test_msft_l = df_l.iloc[Ntest:-1]
    Ctrain = (train_msft_l>0)
    Ctest = (test_msft_l>0)
    model = LogisticRegression(C=10)
    model.fit(train_msft, Ctrain)
    # print(f'Train_Score = {model.score(train_msft, Ctrain)}, Test_Score = {model.score(test_msft,Ctest)}')
    y_test_pp = model.predict_proba(test_msft)
    accuracy_train = model.score(train_msft, Ctrain)
    accuracy_test = model.score(test_msft, Ctest)
    roc_auc = roc_auc_score(Ctest, y_test_pp[:,1])
    symbol += 'l_r'
    fpr, tpr, thres = roc_curve(Ctest, y_test_pp[:,1])
    save_model(symbol,model,'logistic_f',accuracy_train,accuracy_test,roc_auc)
    
    return [accuracy_train,accuracy_test,roc_auc,fpr,tpr,thres]
    
    # print('roc_auc_score:', roc_auc_score(Ctest, y_test_pp[:,1]))
    # print(f'----------{symbol}----------')
    # fpr, tpr, thres = roc_curve(Ctest, y_test_pp[:,1])
    # print(fpr)
    # print(tpr)
    # print(thres)
    # ax = plt.plot(fpr, tpr)
    # plt.show()
    # train_algo = random_forest_df.iloc[1:Ntest][f'{random_c_c}_L_O_Algo_Return'].sum()
    # test_algo = random_forest_df.iloc[Ntest:-1][f'{random_c_c}_L_O_Algo_Return'].sum()
    # print(f' Train_Return:{train_algo}, Test_Return:{test_algo} ||  Buy_And_Holn_Train:{Ytrain.sum()}, Buy_And_Holn_Test:{Ytest.sum()}')

def make_data_ready(df, model_name):
    df.columns = map(str.lower, df.columns)
    df=df.rename(columns = {'adj close':'adjclose'})
    m = model_name
    m_n = np.array([m for i in range(0,len(df))])
    df['name'] = m_n
    df = add_indicators(df)
    df = upside_down(df)
    df = return_prev(df,[m])
    df = shifted_log_return(df,[m])
    t_columns = train_columns(df,[m])
    for s in SYMBOLS:
        if s != model_name:
            t_columns[m].append(f'{s}_Shifted_Log_Return')
            df[f'{s}_Shifted_Log_Return'] = np.array([0 for i in range(0,len(df))])
    return df,t_columns


def train_model(df=df):
    """train_model"""
    df = add_indicators(df)
    df = upside_down(df)
    df = return_prev(df)
    df = shifted_log_return(df)
    t_columns = train_columns(df)
    r_f ={}
    l_r ={}    
    for symbol in SYMBOLS:
        print('Random Forest')
        r_f[symbol]=random_forest(df,t_columns[symbol],symbol)
        print('---------------------------------')
        print('Logistic Reg')
        l_r[symbol]=logistic_reg(df,t_columns[symbol],symbol)
        print('---------------------------------')
    return df,r_f,l_r
    
if __name__ == '__main__':
    train_model(df)
    
