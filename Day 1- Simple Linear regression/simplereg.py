import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('../#100daysofMlcode/Day 1- Simple Linear regression/Salary_Data.csv')
x = data.iloc[0:16, :-1].values
y = data.iloc[0:16, 1].values
#print(y)
print(x)
print(data.shape)
data.head()
#mean
x_mean = np.mean(x)
y_mean = np.mean(y)

l = len(x)
nu = 0
de = 0
#formula to calculate slope m
for i in range(l):
    nu +=(x[i] - x_mean) * (y[i] - y_mean)
    de += (x[i] - x_mean) ** 2
#slope
m = nu / de
c = y_mean - (m * x_mean)
print(m,c)

#plots
x_max = np.max(x)
x_min = np.min(x)
X = [x_min, x_max]
Y = (m * X) + c
plt.plot(X, Y, color='#E69500', label='Regression Line')
plt.scatter(x, y, color = 'red', label='Data Scatter Points')
plt.xlabel('Years of Expirence')
plt.ylabel('Salary')
plt.legend()
plt.show()
