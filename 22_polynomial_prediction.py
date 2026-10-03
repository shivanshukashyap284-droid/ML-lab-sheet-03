import pandas as pd
import numpy as np

DATA_FILE = "regression_dataset.csv"

def load_data():
    return pd.read_csv(DATA_FILE)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
df = load_data()
X, y = df[["Advertising"]], df["Sales"]
poly = PolynomialFeatures(degree=2)
model = LinearRegression().fit(poly.fit_transform(X), y)
new_value = float(input("Enter Advertising value: "))
print("Predicted Sales:", model.predict(poly.transform([[new_value]]))[0])
