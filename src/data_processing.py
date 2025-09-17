import pandas as pd

def load_raw(path='data/raw/telecom_churn_sample.csv'):
    return pd.read_csv(path)

def basic_clean(df):
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(subset=['TotalCharges'], inplace=True)
    return df