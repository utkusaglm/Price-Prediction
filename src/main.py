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
import random
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

query = """select * from assets ORDER BY date ASC;"""
connection = psycopg2.connect(host=config.DB_HOST, database=config.DB_NAME, user= config.DB_USER, password=config.DB_PASS)
cursor = connection.cursor(cursor_factory=psycopg2.extras.DictCursor)
engine = create_engine(f'postgresql://{config.DB_USER}:{config.DB_PASS}@{config.DB_HOST}:{config.DB_PORT}/{config.DB_NAME}')

df = pd.read_sql_query(query,con=engine)
SYMBOLS = ['MSFT','AAPL','NVDA','UBER']
IS_INVESTED = False

def drop_na_bf_fill(df):
    """drop_na_bf_fill"""
    df.dropna(axis=0,how='all',inplace=True)
    df.dropna(axis=1,how='any',inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    return df

def get_close_values_of_df(df):
    """get_close_values_of_df"""
    symbols = df['name'].unique()
    df_msft = df[df['name'] == 'MSFT']
    final_df = pd.DataFrame(data=df_msft['close'].to_numpy(), index = df_msft['date'],columns=['MSFT'])
    for symbol in symbols:
        if symbol != 'MSFT':
            df_sym = df[df['name'] == symbol]
            df_tmp = pd.DataFrame(data=df_sym['close'].to_numpy(), index = df_sym['date'],columns=[symbol])
            final_df =final_df.join(df_tmp)
    return drop_na_bf_fill(final_df)

def get_returns(df):
    """get_returns"""
    df = get_close_values_of_df(df)
    for symbol in df.columns.values:
        df[f'{symbol}_Prev_close']=df[symbol].shift(1)
        df[f'{symbol}_Return'] = df[f'{symbol}'] / df[f'{symbol}_Prev_close'] - 1
        df[f'{symbol}_Log_Return'] = np.log(df[f'{symbol}_Return'] + 1)
    return df

def assign_is_invested(buy,sell):
    """assign_is_invested"""
    global IS_INVESTED
    if IS_INVESTED and sell:
        IS_INVESTED = False
    if not IS_INVESTED and buy:
        IS_INVESTED = True
    return IS_INVESTED

def sma_close(df,slow,fast):
    """ get_close """
    global SYMBOLS
    for symbol in SYMBOLS:
        df[f'{symbol}_Slow_SMA'] = df[symbol].rolling(slow).mean()
        df[f'{symbol}_Fast_SMA'] = df[symbol].rolling(fast).mean()
        df[f'{symbol}_Signal'] = np.where(df[f'{symbol}_Fast_SMA'] >= df[f'{symbol}_Fast_SMA'],1,0)
        df[f'{symbol}_Prev_Signal'] = df[f'{symbol}_Signal'].shift(1)
        df[f'{symbol}_Buy']= (df[f'{symbol}_Prev_Signal'] == 0) & (df[f'{symbol}_Signal'] == 1) # Fast< Slow --> Fast > Slow
        df[f'{symbol}_Sell']= (df[f'{symbol}_Prev_Signal'] == 1) & (df[f'{symbol}_Signal'] == 0) # Fast> Slow --> Fast > 
        df[f'{symbol}_Is_Invested'] = df.apply(lambda row: assign_is_invested(row[f'{symbol}_Buy'], [f'{symbol}_Sell']),axis=1)
    return df
def algo_log_return(df):
    """algo_log_return"""
    global SYMBOLS
    for symbol in SYMBOLS:
        df[f'{symbol}_Algo_Log_Return'] = df[f'{symbol}_Is_Invested'] * df[f'{symbol}_Log_Return']
    return df

def shifted_log_return(df):
    """shifted_log_return"""
    global SYMBOLS
    for symbol in SYMBOLS:
        df[f'{symbol}_Shifted_Log_Return'] = df[f'{symbol}_Log_Return'].shift(-1)
    return df

def linear_regression_f(df):
    """ linear_regression_f """
    linear_df= df[[c for c in df.columns.values if 'Shifted_Log_Return' in c ]]
    
    Ntest = 400
    train = linear_df.iloc[1:Ntest]
    test = linear_df.iloc[Ntest:-1]
    column_val = linear_df.columns.values
    random_c_c = random.choice(linear_df.columns.values)
    train_c = list(column_val)
    train_c.remove(random_c_c)

    Xtrain =train[train_c]
    Ytrain =train[random_c_c]
    Xtest = test[train_c]
    Ytest = test[random_c_c]
    
    model =LinearRegression()
    model.fit(Xtrain, Ytrain)
    print(f'Train_Score = {model.score(Xtrain, Ytrain)}, Test_Score = {model.score(Xtest,Ytest)}')
    #Direction
    Ptrain =model.predict(Xtrain)
    Ptest =model.predict(Xtest)
    # np.mean( np.sign(Ptrain) == np.sign(Ytrain)), np.mean(np.sign(Ptest) == np.sign(Ytest))
    
    linear_df[f'{random_c_c}_L_Position'] = 0
    linear_df.loc[1:Ntest,f'{random_c_c}_L_Position'] = (Ptrain > 0)
    linear_df.loc[Ntest:-1,f'{random_c_c}_L_Position'] = (Ptest > 0)
    linear_df[f'{random_c_c}_L_Algo_Return'] = linear_df[f'{random_c_c}_L_Position'] *  linear_df[random_c_c]
    
    #total algo log return train
    train_algo = linear_df.iloc[1:Ntest][f'{random_c_c}_L_Algo_Return'].sum()
    test_algo = linear_df.iloc[Ntest:-1][f'{random_c_c}_L_Algo_Return'].sum()
    print(f' Train_Return:{train_algo}, Test_Return:{test_algo} ||  Buy_And_Holn_Train:{Ytrain.sum()}, Buy_And_Holn_Test:{Ytest.sum()}')   

def logistic_regression_f(df):
    """ logistic_regression_f """
    logistic_df= df[[c for c in df.columns.values if 'Shifted_Log_Return' in c ]]
    
    Ntest = 400
    train = logistic_df.iloc[1:Ntest]
    test = logistic_df.iloc[Ntest:-1]
    column_val = logistic_df.columns.values
    random_c_c = random.choice(logistic_df.columns.values)
    train_c = list(column_val)
    train_c.remove(random_c_c)

    Xtrain =train[train_c]
    Ytrain =train[random_c_c]
    Xtest = test[train_c]
    Ytest = test[random_c_c]
    
    #reg penality, prevent the weight fro becoming too large
    model =LogisticRegression(C=10)
    Ctrain = (Ytrain>0)
    Ctest = (Ytest>0)
    model.fit(Xtrain, Ctrain)
    print(f'Train_Score = {model.score(Xtrain, Ctrain)}, Test_Score = {model.score(Xtest,Ctest)}')
    #Direction
    Ptrain =model.predict(Xtrain)
    Ptest =model.predict(Xtest)
    # np.mean( np.sign(Ptrain) == np.sign(Ytrain)), np.mean(np.sign(Ptest) == np.sign(Ytest))
    logistic_df[f'{random_c_c}_L_O_Position'] = 0
    logistic_df.loc[1:Ntest,f'{random_c_c}_L_O_Position'] = (Ptrain > 0)
    logistic_df.loc[Ntest:-1,f'{random_c_c}_L_O_Position'] = (Ptest > 0)
    logistic_df[f'{random_c_c}_L_O_Algo_Return'] = logistic_df[f'{random_c_c}_L_O_Position'] *  logistic_df[random_c_c]
    
    #total algo log return train
    train_algo = logistic_df.iloc[1:Ntest][f'{random_c_c}_L_O_Algo_Return'].sum()
    test_algo = logistic_df.iloc[Ntest:-1][f'{random_c_c}_L_O_Algo_Return'].sum()
    print(f' Train_Return:{train_algo}, Test_Return:{test_algo} ||  Buy_And_Holn_Train:{Ytrain.sum()}, Buy_And_Holn_Test:{Ytest.sum()}')

def random_forest_f(df):
    """random_forests"""
    random_forest_df= df[[c for c in df.columns.values if 'Shifted_Log_Return' in c ]]
    
    Ntest = 1600
    train = random_forest_df.iloc[1:Ntest]
    test = random_forest_df.iloc[Ntest:-1]
    column_val = random_forest_df.columns.values
    random_c_c = random.choice(random_forest_df.columns.values)
    train_c = list(column_val)
    train_c.remove(random_c_c)

    Xtrain =train[train_c]
    Ytrain =train[random_c_c]
    Xtest = test[train_c]
    Ytest = test[random_c_c]
    
    #reg penality, prevent the weight fro becoming too large
    model = RandomForestClassifier(random_state=0)
    Ctrain = (Ytrain>0)
    Ctest = (Ytest>0)
    model.fit(Xtrain, Ctrain)
    print(f'Train_Score = {model.score(Xtrain, Ctrain)}, Test_Score = {model.score(Xtest,Ctest)}')
    #Direction
    Ptrain =model.predict(Xtrain)
    Ptest =model.predict(Xtest)
    # np.mean( np.sign(Ptrain) == np.sign(Ytrain)), np.mean(np.sign(Ptest) == np.sign(Ytest))
    
    random_forest_df[f'{random_c_c}_L_O_Position'] = 0
    random_forest_df.loc[1:Ntest,f'{random_c_c}_L_O_Position'] = (Ptrain > 0)
    random_forest_df.loc[Ntest:-1,f'{random_c_c}_L_O_Position'] = (Ptest > 0)
    random_forest_df[f'{random_c_c}_L_O_Algo_Return'] = random_forest_df[f'{random_c_c}_L_O_Position'] *  random_forest_df[random_c_c]
    #total algo log return train
    train_algo = random_forest_df.iloc[1:Ntest][f'{random_c_c}_L_O_Algo_Return'].sum()
    test_algo = random_forest_df.iloc[Ntest:-1][f'{random_c_c}_L_O_Algo_Return'].sum()
    print(f' Train_Return:{train_algo}, Test_Return:{test_algo} ||  Buy_And_Holn_Train:{Ytrain.sum()}, Buy_And_Holn_Test:{Ytest.sum()}')


def add_rsi(df, rsi_period):
    '''Returns dataframe with additional columns:
        rsi (float)
    Args:
        df (pandas.DataFrame): Must be index sorted by datetime:
            adj_close (float)
        rsi_period (int): Number of rsi periods
    Returns:
        df (pandas.DataFrame)
    '''
    chg = df['adj_close'].diff(1)
    gain = chg.mask(chg<0,0)
    loss = chg.mask(chg>0,0)
    avg_gain = gain.ewm(com=rsi_period-1, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(com=rsi_period-1, min_periods=rsi_period).mean()
    rs = abs(avg_gain/avg_loss)
    rsi = 100 - (100/(1+rs))
    df['rsi14'] = rsi
    return df


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
    df['vwap'] = (df['volume']*(df['high']+df['low']+df['adj_close'])/3).cumsum()/df['volume'].cumsum()
    df['vwap'] = df['vwap'].fillna(df['adj_close'])
    df['vwap_var'] = (df['adj_close']/df['vwap'])-1
    return df

if __name__ == '__main__':
    df = get_returns(df)
    df = sma_close(df,30,10)
    df = algo_log_return(df)
    # ---CLASSIC_SMA-----------
    print("----------------------------")
    print("""
          Traditional SMA algorithm
          WE CAN SELECT DIFFERENT MOVING AVERAGE METRICS.
          """)
    print("Slow and Fast SMA")
    for symbol in SYMBOLS:
        a_l_r = df[f'{symbol}_Algo_Log_Return'].sum()
        l_r = df[f'{symbol}_Log_Return'].sum()
        print(f'{symbol}_Algo_Log_Return = {a_l_r} , {symbol}_Log_Return = {l_r}, is_algo_win = {a_l_r>l_r} ')   
    print("""
          TRY TO DETECT,ONE DAY LATER RESULT
          Data is matter
          mutliple lags can be used
          CNN or RNN are not better than linear !
          TWITTER NEWS or EUROPAN ASIAN MARKET can be used.
          """)
    df = shifted_log_return(df)
    print("----------------------------")
    print('LINEAR REGRESSION')
    linear_regression_f(df)
    print("----------------------------")
    print('LOGISTIC REGRESSION')
    logistic_regression_f(df)
    print("----------------------------")
    print('RANDOM FOREST')
    random_forest_f(df)
    print("----------------------------")

    