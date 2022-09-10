import pandas as pd
import numpy as np
import seaborn as se
import matplotlib.pyplot as mp
# this type of regression work on true or fales , 0or1, probability,win or loss
dataset = pd.read_csv(r"insurance.csv")
dataset["insurance"].replace({"no":"0","yes":"1"},inplace=True)
#print(dataset)

#mp.scatter(x="age",y="insurance",data=dataset)
#mp.show()



from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(dataset[['age']],dataset["insurance"],test_size=0.2)
#print(x_test)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)
lr.predict(x_test)
print(lr.predict([[25]]))
print(lr.predict([[60]]))
