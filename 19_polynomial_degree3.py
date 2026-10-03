import pandas as pd
import numpy as np

DATA_FILE = "regression_dataset.csv"

def load_data():
    return pd.read_csv(DATA_FILE)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
df = load_data()
X, y = df[["Advertising"]], df["Sales"]
poly = PolynomialFeatures(degree=3)
model = LinearRegression().fit(poly.fit_transform(X), y)
print("Degree-3 Polynomial Regression coefficients:", model.coef_)
print("Intercept:", model.intercept_)
