import pandas as pd
import numpy as np

DATA_FILE = "regression_dataset.csv"

def load_data():
    return pd.read_csv(DATA_FILE)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
df=load_data(); X,y=df[["Advertising"]],df["Sales"]
a,b,c,d=train_test_split(X,y,test_size=.25,random_state=42)
pred=LinearRegression().fit(a,c).predict(b)
errors=d.values-pred
plt.scatter(pred,errors)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted Sales")
plt.ylabel("Prediction Error")
plt.title("Prediction Errors")
plt.show()
