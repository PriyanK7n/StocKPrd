# -*- coding: utf-8 -*-

# Recurrent Neural Network



# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:,[1] ].values

# Feature Scaling--using normalziation
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output 
X_train = []
y_train = []
for i in range(60, 1258):#i-60 previous(means 60 days previous data )
    X_train.append(training_set_scaled[i-60:i, 0]) #previous timesteps data from i-60 till i-1 taking 60 days previous data till current -1 day
    y_train.append(training_set_scaled[i, 0])# current data to predict i
X_train,y_train=np.array(X_train),np.array(y_train) #converting list to array

# Reshaping --adding dimensionality (A 3rd dimension conating 1 indicater)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN--gives continuous value o we use regressor
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))#input shape mis time steps and indicaters
regressor.add(Dropout(0.2)) #disabling 20 percent neurons 
           #return sequences means adding another LSTM layer or not


# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True)) # no need for input after 1st layer
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer='adam',loss='mean_squared_error')

# Fitting the RNN to the Training set--making conenction of rnn
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32) # 32 stock prizes we r gonna update weigths 


# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017--actual test values
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
'''we conactenate the datasetsnot the train or tst set so tht not the actual test values re scaled '''
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1)) # making 3d array
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)# inversing scaling to get  back to our actual scale of values


# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

#evaluating RNN
import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))#absolute error
rel=rmse/800
'''Our model relays smoothly to the variations of the actual stock price
but cannot judge sudden variation ..but plays smoothy to upward and downward variations 
as Future Vairations re not subject to past''' 




#we can improve accuracy by being an artist and doin parameter Tuning---K-Fold cross Validation


















