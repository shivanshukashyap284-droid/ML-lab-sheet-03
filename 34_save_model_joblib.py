import pandas as pd
import numpy as np

DATA_FILE = "regression_dataset.csv"

def load_data():
    return pd.read_csv(DATA_FILE)
import joblib
from sklearn.linear_model import LinearRegression
df=load_data()
model=LinearRegression().fit(df[["Advertising"]],df["Sales"])
joblib.dump(model,"linear_regression_model.joblib")
print("Model saved as linear_regression_model.joblib")
