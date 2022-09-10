#it is also based on classified problem , it is basecly work on the probability problem
# it is work on base theorem 
#conditional probability
import pandas as pd
dataset = pd.read_csv("social_network.csv")
#print(dataset)
x = dataset.iloc[:,[0,1]].values# firstly separte our model by depended or independed varible 
y = dataset.iloc[:,-1].values


from sklearn.model_selection import train_test_split # split model 
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8)

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(x_train,y_train)
print(model.predict(x_test))
print(model.predict([[22,50000]]))
print(model.score(x_test,y_test))


