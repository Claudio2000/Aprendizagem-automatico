import matplotlib.pyplot as plt
import pickle as p1
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.datasets import data
from sklearn.metrics import mean_squared_error, r2_score
data_x=input("introduz valores do ablalone.data: \n")
data=data_x.split(";")
print(data)
fmap_data = map(float, data)
print(fmap_data)
flist_data = list(fmap_data)
print(flist_data)
data1 = pd.read_csv("abalone.data",   sep=";")
data2=data1.iloc[:0,:11]
data_preparation=pd.DataFrame([flist_data],columns=list(data2))
out=data2
for x in out:
    print(x,data_preparation[x].values)
loaded_model = p1.load(open('abalone.data', 'rb'))
y_pred=loaded_model.predict(data_preparation)
print("abalone data ",int(y_pred))