import pandas as pd
import numpy as np

DATA_FILE = "regression_dataset.csv"

def load_data():
    return pd.read_csv(DATA_FILE)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
print("Simple Linear Regression model created successfully.")
print(model)
