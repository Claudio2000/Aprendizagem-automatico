import matplotlib.pyplot as plt
import pickle as p1
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
data = pd.read_csv("./Dataset/abalone.data", sep=",") ## -- ou -- sep=",")
train_data=data[:1000]
data_X=train_data.iloc[:,1:8]
data_Y=train_data.iloc[:,8:9]
#print(train_data.columns)
print(data_X)
print(data_Y)
regr = linear_model.LinearRegression()
preditor_linear_model=regr.fit(data_X, data_Y)
preditor_Pickle = open('./abaloneprevisao', 'wb')
print("Modelo criado")
