import pandas as pd
import numpy as np
import seaborn as se
import matplotlib.pyplot as mp

dataset = pd.read_csv(r"position.csv")

x = dataset.iloc[:,1:2].values

y = dataset.iloc[:,2].values
#mp.title("POSITION",fontsize=15)
#mp.xlabel("LEVEL",fontsize=15)
#mp.ylabel("SALARY",fontsize=15)
#mp.scatter(x,y)
#mp.show()


#se.lmplot(x='level',y='salary',data=dataset)

#mp.show()


from sklearn import linear_model
reg=linear_model.LinearRegression()
reg.fit(x,y)
print(reg.predict([[6.5]]))

from sklearn.preprocessing import PolynomialFeatures
poly= PolynomialFeatures(degree=2)
x_poly=poly.fit_transform(x)
reg2=linear_model.LinearRegression()
reg2.fit(x_poly,y)
print(reg2.predict(poly.fit_transform([[6.5]])))