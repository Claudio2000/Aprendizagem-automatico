import matplotlib.pyplot as plt
import pickle as p1
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
data = pd.read_csv("../VinhoB/Qualidade_vinho_B/winequality-white.csv",sep=";") ## -- ou -- sep=",")
train_data=data[:1000]
data_X=train_data.iloc[:,0:11]
data_Y=train_data.iloc[:,11:12]
#print(train_data.columns)
print(data_X)
print(data_Y)
regr = linear_model.LinearRegression()
preditor_linear_model=regr.fit(data_X, data_Y)
preditor_Pickle = open('../white-wine_quality_predictor', 'wb')
print("white-wine_quality_predictor")
p1.dump(preditor_linear_model, preditor_Pickle)