#it is used for classification problem
import pandas as pd
dataset=pd.read_csv(r"play.csv") 
#print(dataset)

from sklearn.preprocessing import LabelEncoder# convert to integer from 
outlook = LabelEncoder()
humidity = LabelEncoder()
windy = LabelEncoder()
play = LabelEncoder()

dataset['outlook'] = outlook.fit_transform(dataset["outlook"])
dataset['humidity'] = outlook.fit_transform(dataset["humidity"])
dataset['windy'] = outlook.fit_transform(dataset["windy"])
dataset['play'] = outlook.fit_transform(dataset["play"])

#print(dataset)
feature_column = ['outlook',"humidity","windy"]#convert to depended and independed varible
x = dataset[feature_column]
y = dataset.play
#print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8)


from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='gini')
classifier.fit(x_train,y_train)
#print(classifier.predict(x_test))
#print(classifier.predict([[0,0,0]]))
classifier.score(x_test,y_test)

from sklearn import tree
print(tree.plot_tree(classifier))
