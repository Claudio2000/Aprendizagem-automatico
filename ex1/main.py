import Imports
import imports
import matplotlib.pyplot as plt
import pickle as p1
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
data = pd.read_csv("./winequality-white.csv",sep=";")
train_data=data[:1000]
data_X=train_data.iloc[:,0:11]
data_Y=train_data.iloc[:,11:12]
print(data_X)
print(data_Y)
regr = linear_model.LinearRegression()
preditor_linear_model=regr.fit(data_X, data_Y)
preditor_Pickle = open('./white-wine_quality_predictor', 'wb')
print("white-wine_quality_predictor")
p1.dump(preditor_linear_model, preditor_Pickle)
rr=regr.score(data_X, data_Y)
print("score",rr)
Imports
data = pd.read_csv("./winequality-white.csv",sep=";")
evaluation_data=data[1001:]
data_X=evaluation_data.iloc[:,0:11]
data_Y=evaluation_data.iloc[:,11:12]
print(type(evaluation_data))
print(data_X)
loaded_model = p1.load(open('./white-wine_quality_predictor', 'rb'))
print("Coefficients: \n", loaded_model.coef_)
y_pred=loaded_model.predict(data_X)
z_pred=y_pred-data_Y
print(type(z_pred))
print(type(z_pred["quality"]))
right=0
wrong=0
total=0
for x in z_pred["quality"]:
 z=int(x)
 total=total+1
 if z==0:
 right =right+1
else:
 wrong=wrong+1
print(len(data),total)
print(right)
print(wrong)
print("accuraccy= ",right/total)
print("accuraccy= ",wrong/total)
print(type(evaluation_data))
print(len(data_X))
print(loaded_model)
imports
data_x=input("introduza valores do wine\n")
data=data_x.split(";")
print(data)
fmap_data = map(float, data)
print(fmap_data)
flist_data = list(fmap_data)
print(flist_data)
data1 = pd.read_csv("./winequality-white.csv",sep=";")
data2=data1.iloc[:0,:11]
data_preparation=pd.DataFrame([flist_data],columns=list(data2))
out=data2
for x in out:
 print(x,data_preparation[x].values)
loaded_model = p1.load(open('./white-wine_quality_predictor', 'rb'))
y_pred=loaded_model.predict(data_preparation)
print("wine quality",int(y_pred))