#step of data preprocessing 
import numpy as np
import pandas as pd
from pandas._libs import missing
dataset=pd.read_csv("DMart.csv")
#dataset.fillna(method="ffill")# missing data fill by before one rows data 
#x = pd.read_csv("DMart.csv",usecols=["Name",])
#print(x)
y = dataset[["Price"]].values
#b = y.to_numpy()
#print(b)
x = dataset[["Name","Brand","Price","DiscountedPrice"]].values
#missing value replaced by mean of total values of this column(it work on jupyter becuse pip install sklearn.impute it not work)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
imputer = imputer.fit(x[:,2:4])
x[:,2:4] = imputer.transform(x[:,2:4])
#print(x)
from sklearn.preprocessing import LabelEncoder
label_encoder_x=LabelEncoder()
x[:,0]=label_encoder_x.fit_transform(x[:,0])
#print(x)
label_encoder_x=LabelEncoder()
x[:,1]=label_encoder_x.fit_transform(x[:,1])
#print(x)
from sklearn.preprocessing import OneHotEncoder
OneHotEncoder=OneHotEncoder()
x = OneHotEncoder.fit_transform(dataset.Name.values.reshape(-1,1)).toarray()
#print(x)
labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
x_test
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)


