import sys
import numpy as np
import pandas as pd
sys.path.append('../100daysofMLcode/Day 2- Linear Regression with Gradient Decent')
import gra_desnt as gd
import cal_error as ce

def run():
    data = np.genfromtxt('../100daysofMlcode/Day 1- Simple Linear regression/Salary_Data.csv')
    #hyperparameter
    rate = 0.001
    initial_b = 0
    initial_m = 0
    iterations = 500
    [b, m] = gd.gradient_descent(data, initial_b, initial_m, rate, iterations)
    #final b and m learned
    print('Values of b and m are')
    print(b, m)
    print('Total error is ')
    print(ce.calculate_error())

if __name__ == '__main__':
    run()
