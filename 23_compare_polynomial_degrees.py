import pandas as pd
import numpy as np

DATA_FILE = "regression_dataset.csv"

def load_data():
    return pd.read_csv(DATA_FILE)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
df = load_data()
X, y = df[["Advertising"]], df["Sales"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=42)
for degree in [1,2,3]:
    poly = PolynomialFeatures(degree=degree)
    model = LinearRegression().fit(poly.fit_transform(X_train), y_train)
    pred = model.predict(poly.transform(X_test))
    print(f"Degree {degree} R²: {r2_score(y_test, pred):.4f}")
