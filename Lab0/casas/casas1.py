import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model

XX = pd.read_csv("../casas/casas1.csv")
X=np.array(XX, ndmin=2) #matriz n\times2
y=X[:, 1:].T
y=y[0]
X=X[:, :1]
print(y)
print(X)
regr = linear_model.LinearRegression()
z = regr.fit(X, y)
y300 = regr.predict([[300]]).round(2)
print((y300))
yz = regr.predict(X)

yz = yz.round(3)
plt.title("Preços das casas vs Área em metros quadrados")
plt.xlabel('Área em ($m^2$)')
plt.ylabel('Preço Estimado (¿)')
plt.scatter(X, y, color="black")
plt.scatter(300,y300)
plt.plot(X, yz, color="blue", linewidth=3)
plt.show()
print(y)
print(yz)
print(regr.predict([[300]]).round(3))
