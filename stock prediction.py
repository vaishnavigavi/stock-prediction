# -*- coding: utf-8 -*-


#install the libraries
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.style.use('bmh')

#load the data
from google.colab import files
uploaded = files.upload()

#store the data into data frame
df = pd.read_csv('britannia365.csv')
df.head(10)

#get the number of trading days
df.shape

#visualize the selling price data
plt.figure(figsize=(16,8))
plt.title('britannia')
plt.xlabel('days')
plt.ylabel('prev close')
plt.plot(df['Prev Close'])
plt.show()

#get the prev close
df = df[['Prev Close']]
df.head(4)

#create a variable to predict 'x' days out into the future
future_days=25
#create a new column (target) shifted 'x' units/days up
df['Prediction'] = df[['Prev Close']].shift(-future_days)
df.tail(4)

#create the feature data set (x) and convert it to a numpy array and remove the last 'x' rows/days
x=np.array(df.drop(['Prediction'],1))[:-future_days]
print(x)

#createthe target data set (y)and convert it to a numpy array and get all of the target values except thelast 'x' rows/days
y=np.array(df['Prediction'])[:-future_days]
print(y)

#split  the data into 75% training and 25% testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)

#create models
#create the decision tree regressor model
tree = DecisionTreeRegressor().fit(x_train,y_train)
#create the linear regression model
lr = LinearRegression().fit(x_train,y_train)

#get the last 'x' rows of the feature data set
x_future = df.drop(['Prediction'],1)[:-future_days]
x_future = x_future.tail(future_days)
x_future = np.array(x_future)
x_future

#show the model tree prediction
tree_prediciton = tree.predict(x_future)
print(tree_prediciton)
print()
#show the model linear regression prediction
lr_prediction = lr.predict(x_future)
print(lr_prediction)

#visualize the data
prediction = tree_prediciton

valid= df[x.shape[0]:]
valid['Prediction'] = prediction
plt.figure(figsize=(16,8))
plt.title('model')
plt.xlabel('days')
plt.ylabel('prev close')
plt.plot(df['Prev Close'])
plt.plot(valid[['Prev Close','Prediction']])
plt.legend(['Orig','Val','Pred'])
plt.show()

#visualize the data
prediction = lr_prediction

valid= df[x.shape[0]:]
valid['Prediction'] = prediction
plt.figure(figsize=(16,8))
plt.title('model')
plt.xlabel('days')
plt.ylabel('prev close')
plt.plot(df['Prev Close'])
plt.plot(valid[['Prev Close','Prediction']])
plt.legend(['Orig','Val','Pred'])
plt.show()