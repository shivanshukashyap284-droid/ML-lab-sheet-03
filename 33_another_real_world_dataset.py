from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
data=load_diabetes(as_frame=True)
X=data.data
y=data.target
a,b,c,d=train_test_split(X,y,test_size=.2,random_state=42)
model=LinearRegression().fit(a,c)
print("Dataset: Scikit-learn Diabetes")
print("Test R²:", r2_score(d,model.predict(b)))
