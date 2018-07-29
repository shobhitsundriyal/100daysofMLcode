import numpy as np
import pandas as pd
import Network

df = pd.read_csv('../100daysofMLcode/Day 8-9 - Trying to build a neural network manually/train.csv')
df
y=[]
for i in range(42000):
    yy=[]
    for j in range(10):
        if df[i]['label']==j:
            yy.append(j)
        else:
            yy.append(0)
    y.append(yy)

df.drop(labels='label', axis=1, inplace=True)
x=[]
for i in range(42000):
    x.append(df[i])
