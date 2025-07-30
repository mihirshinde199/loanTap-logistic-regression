"""
Preprocessing module for LoanTap Credit Risk Classification
"""

import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    return df

def basic_cleaning(df):
    df = df.drop(['address', 'title'], axis=1, errors='ignore')
    df.drop_duplicates(inplace=True)
    return df
