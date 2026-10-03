import pandas as pd
import numpy as np

DATA_FILE = "regression_dataset.csv"

def load_data():
    return pd.read_csv(DATA_FILE)
df = load_data()
X = df[["Advertising"]]
y = df["Sales"]
print("Independent variable(s):", X.columns.tolist())
print("Dependent variable:", y.name)
