# Name - Akash Anand
# Roll No. - B20243
# Mob No. - 7677858773
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
import statsmodels.api as sm

# Part A
# Reading the csv file
series = pd.read_csv('/Users/akashanand/Desktop/daily_covid_cases.csv')
# generating the x-ticks
t = [16]
for i in range(10):
    t.append(t[i] + 60)

# Listing labels for the Month-Year
l = ['Feb-20', 'Apr-20', 'Jun-20', 'Aug-20', 'Oct-20', 'Dec-20', 'Feb-21', 'Apr-21', 'Jun-21', 'Aug-21', 'Oct-21']
original = series['new_cases']
plt.figure(figsize=(20, 10))
plt.xticks(t, l)
plt.plot(original)
plt.title('Month-Year vs Case')
plt.xlabel('Month-Year')
plt.ylabel('New Confirmed Cases')
plt.show()

# Part 1 b
print('Part-1 b')
# Generation of time series with 1 day lag
lag = series['new_cases'].shift(1)
corr = pearsonr(lag[1:], original[1:])
print("The Autocorrelation coefficient between the generated one-day lag time sequence and the given time sequence is",round(corr[0], 3))

# Part 1 c
# Scatter plot (one day lagged sequence with time sequence)
plt.xlabel('Given time sequence')
plt.ylabel('One day lagged time sequence')
plt.title('One day lagged sequence vs Given time sequence')
plt.scatter(original[1:], lag[1:])
plt.show()

# Part 1 d
print('Part-1 d')
# lag values
lag_val = [1, 2, 3, 4, 5, 6]
corr2 = []
print("The correlation coefficient between each of the generated time sequences and the given time sequence is")
for d in lag_val:
    lag = series['new_cases'].shift(d)
    corr = pearsonr(lag[d:], original[d:])
    corr2.append(corr[0])
    print(f"{d}-day =", round(corr[0],3))
# Line plot of correlation coefficients vs Lagged values
plt.title('Correlation coefficients vs Lagged Values')
plt.xlabel('Lagged Values')
plt.ylabel('Correlation coefficients')
plt.plot(lag_val, corr2)
plt.show()

# Part 1 e
sm.graphics.tsa.plot_acf(original,lags=lag_val)
plt.xlabel('Lagged Values')
plt.ylabel('Correlation coefficients')
plt.show()
