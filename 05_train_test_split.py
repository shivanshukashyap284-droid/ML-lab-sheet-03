import pandas as pd
import numpy as np

DATA_FILE = "regression_dataset.csv"

def load_data():
    return pd.read_csv(DATA_FILE)
from sklearn.model_selection import train_test_split
df = load_data()
X = df[["Advertising"]]
y = df["Sales"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
