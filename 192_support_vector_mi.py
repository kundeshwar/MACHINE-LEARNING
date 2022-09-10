# it is work on both regression and classification problem
import pandas as pd
dataset = pd.read_csv(r"flower.csv")
#print(dataset)
x = dataset[['sepal_l', 'sepal_w', 'petal_l', 'petal_w']].values
#print(x)
y = dataset['species'].values
#print(y)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8)

from sklearn.svm import SVC
model=SVC(kernel='linear')
model.fit(x_train,y_train)
#print(model.predict(x_test))
#print(model.score(x_test,y_test))
