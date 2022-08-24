# Name - Akash Anand
# Roll No. - B20243
# Mob No. - 7677858773
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

# Train test split
series = pd.read_csv('/Users/akashanand/Desktop/daily_covid_cases.csv', parse_dates=['Date'],index_col=['Date'],sep=',')
test_size = 0.35 # 35% for testing
X = series.values
tst_sz = math.ceil(len(X)*test_size)
train, test = X[:len(X)-tst_sz], X[len(X)-tst_sz:]

# Part A
print('2 a\n')
# Training the model
p = 5
model = AutoReg(train, lags=p)
# Fit/Train the model
model_fit = model.fit()
# Get the coefficients of AR model
coef = model_fit.params

# Printing the coefficients
print('The coefficients obtained are\n', coef)

# Part B
# Using these coefficients walk forward over time steps in test, one step each time
hist = train[len(train)-p:]
hist = [hist[i] for i in range(len(hist))]
pred = list() # List to hold the predictions, 1 step at a time
for t in range(len(test)):
  length = len(hist)
  Lag = [hist[i] for i in range(length-p,length)]
  yhat = coef[0] # Initialize to w0
  for d in range(p):
    yhat += coef[d+1] * Lag[p-d-1] # Add other values
  obs = test[t]
  pred.append(yhat) # Adding predictions to compute RMSE later
  hist.append(obs) # Adding actual test value to hist, to be used in next step

# Part B - a
# Scatter plot between Actual and Predicted values
plt.title('Actual vs Predicted Values')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.scatter(pred, test)
plt.show()

# Part B - b
# Line plot between actual and predicted values
plt.figure(figsize=(20, 10))
plt.title('Predicted and Actual values')
plt.xlabel('Days')
plt.ylabel('New Cases')
plt.plot(test)
plt.plot(pred)
plt.show()

# Part B - c
print('Part C\n')
# Computation of  RMSE
rmse_percent = (math.sqrt(mean_squared_error(test, pred))/np.mean(test))*100
print('RMSE(%):',round(rmse_percent,3))

# Computation of MAPE
mape = np.mean(np.abs((test - pred)/test))*100
print('MAPE:',round(mape,3))