import pandas as pd
import numpy as np

DATA_FILE = "regression_dataset.csv"

def load_data():
    return pd.read_csv(DATA_FILE)
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
df = load_data()
X, y = df[["Advertising"]], df["Sales"]
model = LinearRegression().fit(X, y)
x_line = np.linspace(X.min().iloc[0], X.max().iloc[0], 100).reshape(-1, 1)
plt.scatter(X, y, label="Actual")
plt.plot(x_line, model.predict(x_line), label="Regression line")
plt.xlabel("Advertising")
plt.ylabel("Sales")
plt.title("Simple Linear Regression")
plt.legend()
plt.show()
