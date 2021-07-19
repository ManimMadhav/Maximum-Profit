"""import libraries"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""import dataset"""
dataset = pd.read_csv("50_Startups.csv")

"""setup independant variable"""
x = dataset.iloc[:,:-1].values

"""setup dependant variable"""
y = dataset.iloc[:,-1].values

"""encode the categorical data"""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])],remainder='passthrough')
x = np.array(ct.fit_transform(x))

"""split training set and test set"""
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)


"""LinearRegression class can be used even though there are multiple variables, sklearn detects that it is a case of Multiple linear regression automatically """
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

"""predict test results"""
y_pred = regressor.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_pred),1)),1))

""" say you want to predict the profit of a company: 
1. Based in NYC
2. Has R&D spend = 170000
3. Has Admin spend = 90000
4. Has marketing spend = 350000 """
print(regressor.predict([[0,0,1,170000,90000,350000]]))
