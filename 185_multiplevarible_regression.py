from statistics import LinearRegression
import pandas as pd
from sklearn import linear_model
a = pd.read_csv(r"traning.csv.csv")
#print(a)
b = a.height.mean()
#print(b)
c = a.fillna(b)
reg = linear_model.LinearRegression()
reg.fit(c[["age ","height","weight"]],c["premim"])
#LinearRegression()
print(reg.coef_)
print(reg.intercept_)
print(reg.predict([[27,167.56,60]]))