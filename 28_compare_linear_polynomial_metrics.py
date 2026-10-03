import pandas as pd
import numpy as np

DATA_FILE = "regression_dataset.csv"

def load_data():
    return pd.read_csv(DATA_FILE)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
df=load_data(); X,y=df[["Advertising"]],df["Sales"]
a,b,c,d=train_test_split(X,y,test_size=.25,random_state=42)
models=[("Linear",LinearRegression(),None)]
poly=PolynomialFeatures(degree=2)
models.append(("Polynomial-2",LinearRegression(),poly))
for name,model,transformer in models:
    aa = transformer.fit_transform(a) if transformer else a
    bb = transformer.transform(b) if transformer else b
    pred=model.fit(aa,c).predict(bb)
    print(name, "MAE=",mean_absolute_error(d,pred), "MSE=",mean_squared_error(d,pred), "RMSE=",np.sqrt(mean_squared_error(d,pred)), "R²=",r2_score(d,pred))
