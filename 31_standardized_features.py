import pandas as pd
import numpy as np

DATA_FILE = "regression_dataset.csv"

def load_data():
    return pd.read_csv(DATA_FILE)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
df=load_data(); X=df[["TV","Radio","Newspaper"]]; y=df["Sales"]
a,b,c,d=train_test_split(X,y,test_size=.25,random_state=42)
scaler=StandardScaler()
a_scaled=scaler.fit_transform(a); b_scaled=scaler.transform(b)
model=LinearRegression().fit(a_scaled,c)
print("Standardized-feature model R²:", model.score(b_scaled,d))
