import pandas as pd
import numpy as np

DATA_FILE = "regression_dataset.csv"

def load_data():
    return pd.read_csv(DATA_FILE)
df = load_data()
print("Dataset information:")
df.info()
print("\nDescriptive statistics:")
print(df.describe())
