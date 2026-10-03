import pandas as pd
import numpy as np

DATA_FILE = "regression_dataset.csv"

def load_data():
    return pd.read_csv(DATA_FILE)
from sklearn.metrics import mean_squared_error, r2_score
print("MSE measures the average squared prediction error; lower MSE is generally better.")
print("R² measures the proportion of target variance explained by the model; values closer to 1 indicate better fit.")
df=load_data()
print("Example R² of Sales against Advertising:", r2_score(df["Sales"], np.polyval(np.polyfit(df["Advertising"],df["Sales"],1),df["Advertising"])))
