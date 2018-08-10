import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('../100daysofMlcode/Day 13- Decision Tree Regression/Position_Salaries.csv')
X = data.iloc[:, 1:2].values
y = data.iloc[:, 2].values

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

y_pred = regressor.predict(6.5)
y_pred

X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X, y, color='blue')
plt.plot(X_grid, regressor.predict(X_grid), color='green')
plt.plot()
plt.xlabel('Salary')
plt.ylabel("Position")
plt.show()
