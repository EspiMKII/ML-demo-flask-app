import pandas as pd
from joblib import load
import os

def ensemble_data():
    '''stuff i guess buh'''
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "data/tree-based-ensemble/ensemble-data.csv")
    df = pd.read_csv(csv_path)
    
    df.drop(['OpenInt', 'Volume'], axis=1, inplace=True)
    df = df.tail(7500)

    '''feature engineering'''

    # avg
    df['Average_price'] = (df['Close'] + df['Open']) / 2

    # Moving averages
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    # EMAs
    df['EMA_5'] = df['Close'].rolling(window=5).mean()
    df['EMA_10'] = df['Close'].rolling(window=10).mean()
    df['EMA_20'] = df['Close'].rolling(window=20).mean()

    df['STD_5'] = df['Close'].rolling(window=5).std()
    df['STD_10'] = df['Close'].rolling(window=10).std()
    df['STD_20'] = df['Close'].rolling(window=20).std()
    # Lags
    df['Close_t-1'] = df['Close'].shift(1)
    df['Close_t-2'] = df['Close'].shift(2)
    df['Close_t-3'] = df['Close'].shift(3)

    return df

if __name__ == "__main__":
    ensemble_data()