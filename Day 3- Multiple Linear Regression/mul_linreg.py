import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run ():
    data = pd.read_csv('../100daysofMLcode/Day 3- Multiple Linear Regression/Cost_to_profit.csv')
    data.head()
    data1 =data
    data1.head()

    #feature Normalization
    data = (data - data.mean()) / data.std()
    data.head()

    #x0 = 1
    data.insert(0, 'ones', 1)
    data.head()
    cols = data.shape[1]
    cols
    x = data.iloc[:, 0:cols - 1]
    y = data.iloc[:, cols-1:cols]
    y
    x

    x = np.array(x.values)
    y = np.array(y.values)
    temp = []
    for i in range(cols - 1):
        temp.append(0)
    temp
    theta = np.matrix(np.array(temp))
    #hyperparameters
    rate = 0.001
    iteration = 300
    g, error = grad_desnt(x, y, theta, rate, iteration)
    print('Theta value and error are:')
    print(g, error)

    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(np.arange(iteration), error, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Error vs. Training Epoch')
    plt.show()

def grad_desnt(X, Y, theta1, rate1, iteration1):
    temp = np.matrix(np.zeros(theta1.shape))
    parameters = int(theta1.ravel().shape[1])
    cost = np.zeros(iteration1)

    for i in range(iteration1):
        error = (X * theta1.T) - Y

        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta1[0,j] - ((rate1 / len(X)) * np.sum(term))

        theta1 = temp
        cost[i] = computeCost(X, Y, theta1)

    return theta1, cost

def computeCost(x, y, theta):
    inner = np.power(((x * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(x))

if __name__ == '__main__':
    run()
