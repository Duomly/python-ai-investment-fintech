from datetime import date, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt

from keras import layers
from keras import Sequential
from sklearn.preprocessing import MinMaxScaler

def getStock(stock):
  today = date.today()
  monthAgo = today - timedelta(days=1850)
  data = pdr.get_data_yahoo(stock, start=monthAgo, end=today)
  return data

def splitData(stockPrices):
  stockPricesLastMonth = stockPrices[-20:]
  stockPricesTest = stockPrices[:-20]
  return stockPricesLastMonth, stockPricesTest

def prepareData(prices):
  stockPricesLastMonth, stockPricesTest = splitData(prices)
  cleanData = stockPricesTest.iloc[:, 1:2].values

  scaledData = scaler.fit_transform(cleanData)
  return scaledData, stockPricesTest, stockPricesLastMonth

def splitTrainTest(scaledData):
  inputs = []
  realPrice = []
  for i in range(120, len(scaledData)):
    inputs.append(scaledData[i-120:i, 0])
    realPrice.append(scaledData[i, 0])
  return inputs, realPrice

def reshapeData(inputs, realPrice):
  inputs, realPrice = np.array(inputs), np.array(realPrice)
  inputs = np.reshape(inputs, (inputs.shape[0], inputs.shape[1], 1))
  return inputs

def createModel(inputs):
  model.add(layers.LSTM(units = 30, return_sequences = True, input_shape = (inputs.shape[1], 1)))
  model.add(layers.Dropout(0.2))

  model.add(layers.LSTM(units = 30, return_sequences = True))
  model.add(layers.Dropout(0.2))

  model.add(layers.LSTM(units = 30))
  model.add(layers.Dropout(0.2))

  model.add(layers.Dense(units = 1))
  model.compile(optimizer='adam', loss='mse')

def trainModel(inputs, realPrice, epochs, batch):
  model.fit(inputs, realPrice, epochs = epochs, batch_size = batch)

#Prepare data for prediction
def preparePredictData(stockPricesLastMonth, stockPricesTest):
  #Take the all data
  dataset_total = pd.concat((stockPricesTest['Open'], stockPricesLastMonth['Open']), axis = 0)
  predictInputs = dataset_total[len(dataset_total) - len(stockPricesLastMonth) - 120:].values
  #Reshape the data
  predictInputs = predictInputs.reshape(-1,1)
  #Scale data to fit this one from training
  predictInputs = scaler.transform(predictInputs)
  testPredicts = []

  for i in range(120, 140):
      testPredicts.append(predictInputs[i-120:i, 0])
  #Make 3d arr, and reshape 
  testPredicts = np.array(testPredicts)
  testPredicts = np.reshape(testPredicts, (testPredicts.shape[0], testPredicts.shape[1], 1))

  return testPredicts

#Do Prediction
def getPrediction(dataToPredict):
  nextPrices = model.predict(dataToPredict)
  nextPrices = scaler.inverse_transform(nextPrices)
  return nextPrices

#Create graph to compare with real prices of last month
def createGraph(nextPrices, stockPricesLastMonth):
  #Separate real prices
  real = stockPricesLastMonth.iloc[:, 1:2].values
  #Create graph
  plt.plot(real, color = 'green', label = 'Real prices')
  plt.plot(nextPrices, color = 'orange', label = 'Prices from ai')
  plt.ylabel('Prices')
  plt.xlabel('Timestamp')
  plt.legend()
  plt.show()

prices = getStock('MSFT')
model = Sequential()
scaler = MinMaxScaler(feature_range=(0,1))

def trainAI(prices, epochs, batch):
  scaled, stockPricesTest, stockPricesLastMonth = prepareData(prices)
  inputs, realPrice = splitTrainTest(scaled)
  reshapedInputs = reshapeData(inputs, realPrice)
  createModel(reshapedInputs)
  trainModel(reshapedInputs, realPrice, epochs, batch)

#Predict
def predict():
  stockPricesLastMonth, stockPricesTest = splitData(prices)
  dataToPredict = preparePredictData(stockPricesLastMonth, stockPricesTest)
  predicted = getPrediction(dataToPredict)
  createGraph(predicted, stockPricesLastMonth)

trainAI(prices, 50, 64)
#Start prediction
predict()
