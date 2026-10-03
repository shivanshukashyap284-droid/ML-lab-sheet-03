import pandas as pd
import numpy as np

DATA_FILE = "regression_dataset.csv"

def load_data():
    return pd.read_csv(DATA_FILE)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
df=load_data(); X=df[["TV","Radio","Newspaper"]]; y=df["Sales"]
a,b,c,d=train_test_split(X,y,test_size=.25,random_state=42)
plain=LinearRegression().fit(a,c)
scaled=StandardScaler()
scaled_model=LinearRegression().fit(scaled.fit_transform(a),c)
print("Before scaling R²:",r2_score(d,plain.predict(b)))
print("After scaling R²:",r2_score(d,scaled_model.predict(scaled.transform(b))))
