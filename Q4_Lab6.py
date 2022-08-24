# Name - Akash Anand
# Roll No. - B20243
# Mob No. - 7677858773
import pandas as pd
import numpy as np
from scipy.stats.stats import pearsonr
import math
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

# Train test split
series = pd.read_csv('/Users/akashanand/Desktop/daily_covid_cases.csv', parse_dates=['Date'],index_col=['Date'],sep=',')
test_size = 0.35 # 35% for testing
X = series.values
tst_sz = math.ceil(len(X)*test_size)
train, test = X[:len(X)-tst_sz], X[len(X)-tst_sz:]

# Computation of number of optimal value of p
p = 1
while p < len(series):
  corr = pearsonr(train[p:].ravel(), train[:len(train)-p].ravel())
  if(abs(corr[0]) <= 2/math.sqrt(len(train[p:]))):
    print('The heuristic value is',p-1)
    break
  p+=1

p=p-1
# Training the model
model = AutoReg(train, lags=p)
# Fit/train the model
model_fit = model.fit()
coef = model_fit.params
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

# Computation of  RMSE
rmse_per = (math.sqrt(mean_squared_error(test, pred))/np.mean(test))*100
print('RMSE(%):',round(rmse_per,3))

# Computation of  MAPE
mape = np.mean(np.abs((test - pred)/test))*100
print('MAPE:',round(mape,3))