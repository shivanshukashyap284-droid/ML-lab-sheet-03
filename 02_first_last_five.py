import pandas as pd
import numpy as np

DATA_FILE = "regression_dataset.csv"

def load_data():
    return pd.read_csv(DATA_FILE)
df = load_data()
print("First five records:")
print(df.head())
print("\nLast five records:")
print(df.tail())
