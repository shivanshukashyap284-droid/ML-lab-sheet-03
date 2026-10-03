import pandas as pd
import numpy as np

DATA_FILE = "regression_dataset.csv"

def load_data():
    return pd.read_csv(DATA_FILE)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
df=load_data(); X,y=df[["Advertising"]],df["Sales"]
a,b,c,d=train_test_split(X,y,test_size=.25,random_state=42)
p=LinearRegression().fit(a,c).predict(b)
print("MAE:", mean_absolute_error(d,p))
