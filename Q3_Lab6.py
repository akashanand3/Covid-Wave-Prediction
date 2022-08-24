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

lag_val = [1,5,10,15,25]    # Lagged values
RMSE = []
MAPE = []
for l in lag_val:
  model = AutoReg(train, lags=l)
  # fit/train the model
  model_fit = model.fit()
  coef = model_fit.params
  hist = train[len(train)-l:]
  hist = [hist[i] for i in range(len(hist))]
  predicted = list() # List to hold the predictions, 1 step at a time
  for t in range(len(test)):
    length = len(hist)
    Lag = [hist[i] for i in range(length-l,length)]
    yhat = coef[0] # Initialize to w0
    for d in range(l):
      yhat += coef[d+1] * Lag[l-d-1] # Add other values
    obs = test[t]
    predicted.append(yhat) # Adding predictions to compute RMSE later
    hist.append(obs) # Adding actual test value to hist, to be used in next step

  # Computation of  RMSE
  rmse_per = (math.sqrt(mean_squared_error(test, predicted))/np.mean(test))*100
  RMSE.append(rmse_per)

  # Computation of  MAPE
  mape = np.mean(np.abs((test - predicted)/test))*100
  MAPE.append(mape)

# RMSE (%) and MAPE between predicted and original data values wrt lags in time sequence
data = {'Lag value':lag_val,'RMSE(%)':RMSE, 'MAPE' :MAPE}
print('Table 1\n',pd.DataFrame(data))

# plotting RMSE(%) vs. time lag
plt.title('RMSE(%) vs. Time lag')
plt.xlabel('Time Lag')
plt.ylabel('RMSE(%)')
plt.xticks([1,2,3,4,5],lag_val)
plt.bar([1,2,3,4,5],RMSE)
plt.show()

# plotting MAPE vs. time lag
plt.title('MAPE vs. time lag')
plt.xlabel('Time Lag')
plt.ylabel('MAPE')
plt.xticks([1,2,3,4,5],lag_val)
plt.bar([1,2,3,4,5],MAPE,color='orange')
plt.show()
