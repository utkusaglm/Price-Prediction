import numpy as np
from feature_engineering import get_features_df_and_train_columns, FeatureEngineering
from model import Model
import config

SYMBOLS = config.SYMBOLS
df,t_columns = get_features_df_and_train_columns()
model = Model(df)
feature_class = FeatureEngineering(symbols=SYMBOLS)

def train_model():
    """train_model and return every model
    [accuracy_train,accuracy_test,roc_auc,fpr,tpr,thres]
    """
    r_f = {}
    l_r = {}
    for symbol in SYMBOLS:
        print('Random Forest')
        r_f[symbol] = model.random_forest(t_columns[symbol],symbol)
        l_r[symbol] = model.logistic_reg(t_columns[symbol],symbol)
    return model.df, r_f, l_r

def make_data_ready(df_loaded, model_name):
    """
    make the given data ready for analyzing
    """
    df_loaded.columns = map(str.lower, df_loaded.columns)
    df_loaded=df_loaded.rename(columns = {'adj close':'adjclose'})
    m = model_name
    m_n = np.array([m for i in range(0,len(df_loaded))])
    df_loaded['name'] = m_n
    df_loaded = feature_class.add_indicators(df_loaded)
    df_loaded = feature_class.upside_down(df_loaded)
    df_loaded = feature_class.return_prev(df_loaded,[m])
    df_loaded = feature_class.shifted_log_return(df_loaded,[m])
    train_columns = feature_class.train_columns(df_loaded,[m])

    for s in SYMBOLS:
        if s != model_name:
            train_columns[m].append(f'{s}_Shifted_Log_Return')
            df_loaded[f'{s}_Shifted_Log_Return'] = np.array([0 for i in range(0,len(df_loaded))])
    train_columns[m] = train_columns[m][:-2]
    return df_loaded,train_columns
