#multiple decision tree make forest is called random forest 
import pandas as pd
from pandas.core.algorithms import unique
datasets = pd.read_csv(r"flower.csv").values
from sklearn import datasets
iris = datasets.load_iris
#print(datasets)
print(iris.target_names)
print(iris.feature_names)
data=pd.DataFrame({"sepal length":iris.data[:,0],"sepal width":iris.data[:,1],"pedal length":iris.data[:,2],"pedal width":iris.data[:,3],"species":iris.target})
data.head()
x = data[["sepal length","sepal width","pedal length","pedal width"]]#features
y = data["species"]#label
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7)

from sklearn.ensemble import RandomForestClassifier
clf= RandomForestClassifier(n_estimators=10,criterion="gini")
clf.fit(x_train,y_train)
clf.predict(x_test,y_test)

feature_imp=pd.Series(clf.feature_importances_,index=iris.features_name).sort_values(ascending=False)
