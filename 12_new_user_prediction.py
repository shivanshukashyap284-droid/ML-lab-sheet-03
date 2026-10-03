import pandas as pd
import numpy as np

DATA_FILE = "regression_dataset.csv"

def load_data():
    return pd.read_csv(DATA_FILE)
from sklearn.linear_model import LinearRegression
df = load_data()
model = LinearRegression().fit(df[["Advertising"]], df["Sales"])
new_value = float(input("Enter Advertising value: "))
prediction = model.predict([[new_value]])[0]
print("Predicted Sales:", prediction)
