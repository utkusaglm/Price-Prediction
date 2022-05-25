import pandas as pd
import numpy as np
import config
import query

SYMBOLS = config.SYMBOLS
DF = query.df

class FeatureEngineering:
    """raw data to features """
    def __init__(self,symbols):
        self.symbols = symbols

    def get_rsi(self, df, rsi_period):
        """get_rsi"""
        chg = df['close'].diff(1)
        gain = chg.mask(chg<0,0)
        loss = chg.mask(chg>0,0)
        avg_gain = gain.ewm(com=rsi_period-1, min_periods=rsi_period).mean()
        avg_loss = loss.ewm(com=rsi_period-1, min_periods=rsi_period).mean()
        rs = abs(avg_gain/avg_loss)
        rsi = 100 - (100/(1+rs))
        return rsi

    def add_vwap(self, df):
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
   
    def add_indicators(self,df):
        """ add_indicators """
        # relative strength index
        df['rsi14'] = self.get_rsi(df, 14)
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
        df = self.add_vwap(df)
        return df

    def drop_na_bf_fill(self,df):
        """drop_na_bf_fill"""
        df.dropna(axis=0,how='all',inplace=True)
        df.dropna(axis=1,how='any',inplace=True)
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        return df
    
    def upside_down(self,df):
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
        return self.drop_na_bf_fill(final_df)
    
    def return_prev(self,df):
        """return_prev"""
        for symbol in self.symbols:
            df[f'{symbol}_Prev_close']=df[f'{symbol}_Close'].shift(1)
            df[f'{symbol}_Return'] = np.array(df[f'{symbol}_Close']) / np.array(df[f'{symbol}_Prev_close']) - 1
            df[f'{symbol}_Log_Return'] = np.log(df[f'{symbol}_Return'] + 1)
        return df
    
    def shifted_log_return(self,df):
        """shifted_log_return"""
        for symbol in self.symbols:
            df[f'{symbol}_Shifted_Log_Return'] = df[f'{symbol}_Log_Return'].shift(-1)
        return df
    
    
    def train_columns(self,df):
        """ eliminate correlated columns. """
        train_ready = {}
        for symbol in self.symbols: 
            train_columns = [list(c)[0] for c in df.columns.values if symbol in str(c)]
            train_columns.extend([f'{c_name}_Shifted_Log_Return' for c_name in self.symbols if c_name != symbol] )
            train_columns.remove(f'{symbol}_Shifted_Log_Return')
            train_columns.remove(f'{symbol}_Log_Return')
            train_columns.remove(f'{symbol}_Return')
            train_columns.remove(f'{symbol}_Close')
            train_columns.remove(f'{symbol}_Prev_close')
            train_ready[symbol] = train_columns
        return train_ready

def get_features_df_and_train_columns(df=DF):
    """features train columns"""
    feature_class = FeatureEngineering(symbols=SYMBOLS)
    feature_df = feature_class.add_indicators(df)
    feature_df =  feature_class.upside_down(feature_df)
    feature_df =  feature_class.return_prev(feature_df)
    feature_df =  feature_class.shifted_log_return(feature_df)
    t_columns = feature_class.train_columns(feature_df)
    return feature_df, t_columns
    