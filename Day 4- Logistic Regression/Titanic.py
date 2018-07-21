import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

train_data = pd.read_csv("../100daysofMLcode/Day 4- Logistic Regression/Titanic Train Data.csv")
train_data.head()

sns.countplot(x='Survived', data=train_data)
sns.countplot(x='Survived', hue='Sex', data=train_data)
sns.countplot(x='Survived', hue='Pclass', data=train_data)
train_data['Age'].plot.hist()
train_data['Fare'].plot.hist(bins=20)

sns.countplot(x='SibSp', data=train_data)
#Data Wrangling
train_data.isnull()
train_data.isnull().sum()
sns.heatmap(train_data.isnull())
sns.boxplot(x='Pclass', y='Age', data=train_data)
train_data.drop('Cabin', axis=1, inplace=True,)
train_data.head()
train_data.dropna(inplace=True)
sns.heatmap(train_data.isnull())
gender = pd.get_dummies(train_data['Sex'], drop_first=True)
gender.head()
embark = pd.get_dummies(train_data['Embarked'], drop_first=True)
embark.head()
pclass = pd.get_dummies(train_data['Pclass'], drop_first=True)
pclass.head()

train_data = pd.concat([train_data, gender, embark, pclass], axis=1)
train_data.head()
train_data.drop(['Sex', 'Embarked', 'PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
train_data.head()
train_data.drop(['Pclass'], axis=1, inplace=True)
train_data.head()

#Training
x = train_data.drop('Survived', axis=1)
y = train_data['Survived']
from sklearn.linear_model import LogisticRegression as rg
logmodel = rg()
logmodel.fit(x, y)

#Test
test_data = pd.read_csv('../100daysofMLcode/Day 4- Logistic Regression/test.csv')
test_data.isnull().sum()
test_data.drop('Cabin', axis=1, inplace=True,)
test_data.head()
test_data.drop('Name', axis=1, inplace=True,)
#Assumming
test_data.head()
sns.heatmap(test_data.isnull())
test_data["Fare"].fillna(test_data["Fare"].mean(), inplace=True)
test_data.isnull().sum()
#t = test_data
#t.dropna(inplace=True)
#t.isnull().sum()
gender = pd.get_dummies(test_data['Sex'], drop_first=True)
gender.head()
embark = pd.get_dummies(test_data['Embarked'], drop_first=True)
embark.head()
pclass = pd.get_dummies(test_data['Pclass'], drop_first=True)
pclass.head()

test_data = pd.concat([test_data, gender, embark, pclass], axis=1)
test_data.head()
test_data.drop(['Sex', 'Embarked', 'PassengerId', 'Ticket'], axis=1, inplace=True)
test_data.head()
test_data.drop(['Pclass'], axis=1, inplace=True)
test_data.head()
train_data.head()
len(test_data)
test_data["Age"].fillna(0, inplace=True)
test_data.isnull().sum()

pred = logmodel.predict(test_data)
pred
len(pred)

t = pd.read_csv('../100daysofMLcode/Day 4- Logistic Regression/test.csv')
#final = {'PassengerId':t['PassengerId'], 'Survived':pred}
final = pd.DataFrame(data={'PassengerId':t['PassengerId'], 'Survived':pred})
final
#final.drop(final[''])
final.to_csv('../100daysofMLcode/Day 4- Logistic Regression/Submit.csv', sep=',')
