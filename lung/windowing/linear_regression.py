import csv
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression


r = open('mean_list.csv', 'r')
rdr = csv.reader(r)
hd = next(rdr)
X = []
y = []
X0 = []
y0 = []
X1 = []
y1 = []
for item in rdr:
    i = float(item[0])
    j = float(item[1])
    if i >= 80:
        X.append(i)
        y.append(j)
        if j <= 100:
            X0.append(i)
            y0.append(j)
        else:
            X1.append(i)
            y1.append(j)
X = np.array(X).reshape(-1, 1)
y = np.array(y).reshape(-1, 1)
X0 = np.array(X0).reshape(-1, 1)
y0 = np.array(y0).reshape(-1, 1)
X1 = np.array(X1).reshape(-1, 1)
y1 = np.array(y1).reshape(-1, 1)

model = LinearRegression().fit(X, y)
print(model.coef_, model.intercept_)

yhat = model.predict(X)
SS_Residual = sum((y-yhat)**2)
SS_Total = sum((y-np.mean(y))**2)
r_squared = 1 - (float(SS_Residual))/SS_Total
adjusted_r_squared = 1 - (1-r_squared)*(len(y)-1)/(len(y)-X.shape[1]-1)
print(r_squared, adjusted_r_squared)

xs = np.linspace(np.min(X), np.max(X), 100)
plt.plot(X, y, 'o')
plt.plot(xs, model.coef_[0][0]*xs + model.intercept_[0])


model = LinearRegression().fit(X0, y0)
print(model.coef_, model.intercept_)

yhat = model.predict(X0)
SS_Residual = sum((y0-yhat)**2)
SS_Total = sum((y0-np.mean(y0))**2)
r_squared = 1 - (float(SS_Residual))/SS_Total
adjusted_r_squared = 1 - (1-r_squared)*(len(y0)-1)/(len(y0)-X0.shape[1]-1)
print(r_squared, adjusted_r_squared)

xs = np.linspace(np.min(X0), np.max(X0), 100)
plt.plot(xs, model.coef_[0][0]*xs + model.intercept_[0])


model = LinearRegression().fit(X1, y1)
print(model.coef_, model.intercept_)

yhat = model.predict(X1)
SS_Residual = sum((y1-yhat)**2)
SS_Total = sum((y1-np.mean(y1))**2)
r_squared = 1 - (float(SS_Residual))/SS_Total
adjusted_r_squared = 1 - (1-r_squared)*(len(y1)-1)/(len(y1)-X1.shape[1]-1)
print(r_squared, adjusted_r_squared)

xs = np.linspace(np.min(X1), np.max(X1), 100)
plt.plot(xs, model.coef_[0][0]*xs + model.intercept_[0])
plt.show()
