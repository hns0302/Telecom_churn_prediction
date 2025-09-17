import pandas as pd

def create_features(df):
    # Example: tenure groups
    bins = [0,12,24,48,600]
    labels = ['0-12','12-24','24-48','48+']
    df['tenure_group'] = pd.cut(df['tenure'].astype(int), bins=bins, labels=labels, include_lowest=True)
    return df