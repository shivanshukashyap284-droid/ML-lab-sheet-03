import pandas as pd
import numpy as np

DATA_FILE = "regression_dataset.csv"

def load_data():
    return pd.read_csv(DATA_FILE)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
df = load_data()
X, y = df[["Advertising"]], df["Sales"]
x_line = np.linspace(X.min().iloc[0], X.max().iloc[0], 100).reshape(-1,1)
plt.scatter(X, y, label="Actual")
for degree in [2,3]:
    poly = PolynomialFeatures(degree=degree)
    model = LinearRegression().fit(poly.fit_transform(X), y)
    plt.plot(x_line, model.predict(poly.transform(x_line)), label=f"Degree {degree}")
plt.xlabel("Advertising")
plt.ylabel("Sales")
plt.title("Polynomial Regression Curves")
plt.legend()
plt.show()
