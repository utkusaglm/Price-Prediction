from os import name
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
import psycopg2
import psycopg2.extras

from sqlalchemy import create_engine
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

LagNames = []

#TODO: outside
STOCKS = ["MSFT","AAPL","GOOG"]
CRYPTO = ["BTC-USD","ETH-USD"]
ASSETS = CRYPTO + STOCKS

query ="""select * from assets where name = '{name_of_asset}'
ORDER BY date ASC;"""

def rolling_mean(asset,columns,window):
    rolling_windows = asset[columns].rolling(window,center=False)
    rolling_mean=rolling_windows.mean()
    return rolling_mean
    
def bollinger_bands(asset,columns,window):
    rm= rolling_mean(asset,columns,window)
    std_of_rolling = asset[columns].rolling(window=window,center=False).std()
    upper_band= rm+ std_of_rolling
    lower_band= rm- std_of_rolling
    return (upper_band,lower_band)

def daily_return(asset,columns):
    return asset[columns]/asset[columns].shift(1) -1

def cumulative_sum(asset,columns):
    return daily_return(asset,columns).sum()

def lagit(asset,lags):
    names = []
    for i in range(1,lags+1):
        asset['Lag_'+str(i)]=asset['daily_return'].shift(i)
        names.append('Lag_'+str(i))
    return names

def modify_assets(asset,columns):
    global LagNames
    asset["daily_return"]=daily_return(asset,columns)
    LagNames=lagit(asset,5)
    asset.dropna(inplace=True)

def linear_regression(asset):
    global LagNames
    train,test=train_test_split(asset,shuffle=False,
                        test_size=0.3,random_state=0)
    train=train.copy()
    test=test.copy()
    model = LinearRegression()
    model.fit(train[LagNames],train['daily_return'])
    test['prediction_LR'] = model.predict(test[LagNames])
    test['direction_LR'] = [1 if i> 0 else -1 for i in test['prediction_LR']]
    test['strat_LR'] = test['direction_LR'] * test['daily_return']
    return test
    # print(np.exp(test[['daily_return','strat_LR']].sum()))
    # print((test['direction_LR'].diff() !=0).value_counts())
    # np.exp(test[['daily_return','strat_LR']].cumsum()).plot()
    # plt.show()
    # return (test['strat_LR'])

def decision_tree_regressor(asset):
    global LagNames
    train,test=train_test_split(asset,shuffle=False,
                        test_size=0.3,random_state=0)
    train=train.copy()
    test=test.copy()
    regr = DecisionTreeRegressor(max_depth=2)
    regr.fit(train[LagNames],train['daily_return'])
    test['prediction_DT'] = regr.predict(test[LagNames])
    test['direction_DT'] = [1 if i> 0 else -1 for i in test['prediction_DT']]
    test['strat_DT'] = test['direction_DT'] * test['daily_return']
    return test
    # print(np.exp(test[['daily_return','strat_DT']].sum()))
    # print((test['direction_DT'].diff() !=0).value_counts())
    # np.exp(test[['daily_return','strat_DT']].cumsum()).plot()
    # plt.show()
    # return()

def svm_regressor(asset):
    global LagNames
    train,test=train_test_split(asset,shuffle=False,
                        test_size=0.3,random_state=0)
    train=train.copy()
    test=test.copy()
    regr=svm.SVR()
    regr.fit(train[LagNames],train['daily_return'])
    test['prediction_SVM'] = regr.predict(test[LagNames])
    test['direction_SVM'] = [1 if i> 0 else -1 for i in test['prediction_SVM']]
    test['strat_SVM'] = test['direction_SVM'] * test['daily_return']
    return test
    # print(np.exp(test[['daily_return','strat_SVM']].sum()))
    # print((test['direction_SVM'].diff() !=0).value_counts())
    # np.exp(test[['daily_return','strat_SVM']].cumsum()).plot()
    # plt.show()
    # return()

def knn_regressor(asset):
    global LagNames
    train,test=train_test_split(asset,shuffle=False,
                        test_size=0.3,random_state=0)
    train=train.copy()
    test=test.copy()
    regr=KNeighborsRegressor(n_neighbors=2)
    regr.fit(train[LagNames],train['daily_return'])
    test['prediction_KNN'] = regr.predict(test[LagNames])
    test['direction_KNN'] = [1 if i> 0 else -1 for i in test['prediction_KNN']]
    test['strat_KNN'] = test['direction_KNN'] * test['daily_return']
    return test
    # print(np.exp(test[['daily_return','strat_KNN']].sum()))
    # print((test['direction_KNN'].diff() !=0).value_counts())
    # np.exp(test[['daily_return','strat_KNN']].cumsum()).plot()
    # plt.show()
    # return()

def plot(assets_dfs):
    # ["BTC-USD","ETH-USD"] ["MSFT","AAPL","GOOG"]
    global ASSETS
    fig, axs = plt.subplots(2, 5)
    # for df_name in ASSETS:
    models = ["LR","DT","SVM","KNN"]
    assets_name =["BTC-USD","ETH-USD"]
    for i in range(0,len(assets_name)):
        df_list= assets_dfs[assets_name[i]]
        df_list[0]['summed'] = df_list[0]['prediction_LR']
        for j in range(0,len(df_list)):
            axs[i,j].set_title(models[j])
            
            sum_of_daily_return = str(np.exp(df_list[j][['daily_return',f'strat_{models[j]}']].sum())).replace('\n',' ')
            count =  str((df_list[j][f'direction_{models[j]}'].diff() !=0).value_counts()).replace('\n',' ')
            print("---------------------------------------------------------------")
            print(f"model: {models[j]}, sum of daily return {sum_of_daily_return}")
            print(f"model: {models[j]}, count of how many actions we need {count}")
            print("---------------------------------------------------------------")
            axs[i,j].plot(np.exp(df_list[j][['daily_return',f'strat_{models[j]}']].cumsum()),label=['daily_return',f'strat_{models[j]}'])
            axs[i,j].legend()
            if j != 0:
                df_list[0]['summed'] = df_list[0]['summed']+ df_list[j][f'prediction_{models[j]}']
        df_list[0]['summed'] /= (j+1)
        df_list[0]['direction_summed'] = [1 if i> 0 else -1 for i in df_list[0]['summed']]
        df_list[0]['strat_summed'] = df_list[0]['direction_summed'] * df_list[0]['daily_return']
        axs[i,(j+1)].set_title("ensemble")
        axs[i,(j+1)].plot(np.exp(df_list[0][['daily_return','strat_summed']].cumsum()),label=['daily_return','strat_summed'])
        axs[i,(j+1)].legend()
        sum_of_daily_return = str(np.exp(df_list[0][['daily_return','strat_summed']].sum())).replace('\n',' ')
        count =  str((df_list[0]['direction_summed'].diff() !=0).value_counts()).replace('\n',' ')
        print("***********************************************************************")
        print(f"model: ensemble, sum of daily return{sum_of_daily_return}")
        print(f"model:ensemble, count of how many actions we need {count}")
        print("***********************************************************************")
    plt.show()
        
           
            
    # axs[0,0].plot(df_list[0]['daily_return'],df_list[0]['strat_LR'], 'tab:orange')
    # np.exp(df_list[0][['daily_return','strat_LR']]).plot()
    # df_list[0][['daily_return','strat_LR']].plot()
    # np.exp(df_list[0][['daily_return','strat_LR']].cumsum()).plot()
    # __import__('ipdb').set_trace()



def run():
    all_dfs = {}
    for asset in CRYPTO:
        models_test_data = []
        df = pd.read_sql_query(query.format(name_of_asset=asset),con=engine)
        modify_assets(df,"close")
        bands= bollinger_bands(df,"close",10)
        df["upper_band"] =bands[0]
        df["lower_band"] =bands[1]
        lr= linear_regression(df)
        dtr= decision_tree_regressor(df)
        svm= svm_regressor(df)
        knn= knn_regressor(df)
        models_test_data.append(lr)
        models_test_data.append(dtr)
        models_test_data.append(svm)
        models_test_data.append(knn)
        all_dfs[asset]=models_test_data
    
    for asset in STOCKS:
        models_test_data = []
        df = pd.read_sql_query(query.format(name_of_asset=asset),con=engine)
        df.fillna(method='bfill')
        modify_assets(df,"close")
        bands= bollinger_bands(df,"close",10)
        df["upper_band"] =bands[0]
        df["lower_band"] =bands[1]
        lr= linear_regression(df)
        dtr= decision_tree_regressor(df)
        svm= svm_regressor(df)
        knn= knn_regressor(df)
        models_test_data.append(lr)
        models_test_data.append(dtr)
        models_test_data.append(svm)
        models_test_data.append(knn)
        all_dfs[asset]=models_test_data
    plot(all_dfs)

    
    
    
# modify_assets(df,"Close")
# decision_tree_regressor(df)
# svm_regressor(df)
# linear_regression(df)
# knn_regressor(df)

if __name__ == '__main__':
    run()