import pandas as pd
import numpy as np

DATA_FILE = "regression_dataset.csv"

def load_data():
    return pd.read_csv(DATA_FILE)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
df = load_data()
X, y = df[["Advertising"]], df["Sales"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model = LinearRegression().fit(X_train, y_train)
print("Model trained successfully.")
print("Coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)
