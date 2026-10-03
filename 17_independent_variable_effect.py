import pandas as pd
import numpy as np

DATA_FILE = "regression_dataset.csv"

def load_data():
    return pd.read_csv(DATA_FILE)
from sklearn.linear_model import LinearRegression
df = load_data()
features = ["TV","Radio","Newspaper"]
model = LinearRegression().fit(df[features], df["Sales"])
for feature, coefficient in zip(features, model.coef_):
    print(f"{feature}: coefficient = {coefficient:.4f}")
print("Positive coefficient indicates an increase in the feature tends to increase prediction, holding other features constant.")
