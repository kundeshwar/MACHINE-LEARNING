import pandas as pd
from pandas.core.algorithms import unique
dataset = pd.read_csv(r"flower.csv")
#dataset["species"].unique()
dataset["species"].replace({'i-v':0, 'i-s':1, 'i-vr':2, 's-vr':3},inplace=True)
#print(dataset)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(dataset[["sepal_l",'sepal_w','petal_l','petal_w']],dataset["species"],test_size=0.2)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)
#print(x_test)
#print(lr.predict(x_test))
print(lr.predict([[5.2,3.5,4.4,0.3]]))
#lr.score(x_test,y_test)it used to check our accurancy of model
import seaborn as sns
import matplotlib.pyplot as mp
a = sns.pairplot(dataset[["sepal_l",'sepal_w','petal_l','petal_w','species']],hue='species')
print(a)
mp.show()
