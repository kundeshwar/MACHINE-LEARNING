import pandas as pd
a = pd.read_csv(r"traningdata.csv")
#print(a)
x = a["age "]
#print(x)
y = a["premim"]
#print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
#print(x_test)
#print(x_train)
#print(y_test)
#print(y_train)
import matplotlib.pyplot as mp
from sklearn import linear_model

mp.xlabel("AGE",fontsize=15)
mp.ylabel("PRAMIUM",fontsize=15)
mp.title("INSURANCE",fontsize=15)

mp.scatter(x,y)

mp.show()
reg = linear_model.LinearRegression()
reg.fit(a[["age "]],a["premim"])
print(reg.predict([[21]]))
print(reg.predict([[50]]))
print(reg.coef_)
print(reg.intercept_)