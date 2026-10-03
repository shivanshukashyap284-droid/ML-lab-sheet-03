import pandas as pd
import numpy as np

DATA_FILE = "regression_dataset.csv"

def load_data():
    return pd.read_csv(DATA_FILE)
import joblib
model=joblib.load("linear_regression_model.joblib")
new_value=float(input("Enter Advertising value: "))
print("Predicted Sales:", model.predict([[new_value]])[0])
