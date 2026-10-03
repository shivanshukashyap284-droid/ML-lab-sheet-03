import pandas as pd
import numpy as np

DATA_FILE = "regression_dataset.csv"

def load_data():
    return pd.read_csv(DATA_FILE)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
df = load_data()
X, y = df[["Advertising"]], df["Sales"]
linear = LinearRegression().fit(X, y)
print("Linear R²:", linear.score(X, y))
for degree in [2, 3]:
    poly = PolynomialFeatures(degree=degree)
    model = LinearRegression().fit(poly.fit_transform(X), y)
    print(f"Polynomial degree {degree} R²:", model.score(poly.transform(X), y))
