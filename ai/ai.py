# Import depencencies
from datetime import date, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt

# Install and import keras
from keras import layers
from keras import Sequential
from sklearn.preprocessing import MinMaxScaler

# Get stock prices from API
def getStock(stock):
  today = date.today()
  # Take data for the last 5 years
  monthAgo = today - timedelta(days=1850)
  data = pdr.get_data_yahoo(stock, start=monthAgo, end=today)
  return data

#Split data
def splitData(stockPrices):
  stockPricesLastMonth = stockPrices[-20:]
  stockPricesTest = stockPrices[:-20]
  return stockPricesLastMonth, stockPricesTest

#Prepare data
def prepareData(prices):
  stockPricesLastMonth, stockPricesTest = splitData(prices)
  # Get only open price value as the important one
  cleanData = stockPricesTest.iloc[:, 1:2].values

  # Scale data to speed up alghoritm
  scaledData = scaler.fit_transform(cleanData)
  return scaledData, stockPricesTest, stockPricesLastMonth

#Split train and result
def splitTrainTest(scaledData):
  # Take the data for the first 6 months and real prices
  inputs = []
  realPrice = []
  for i in range(120, len(scaledData)):
    inputs.append(scaledData[i-120:i, 0])
    realPrice.append(scaledData[i, 0])
  return inputs, realPrice

# Create the table of arrays and reshape to have one list
def reshapeData(inputs, realPrice):
  inputs, realPrice = np.array(inputs), np.array(realPrice)
  inputs = np.reshape(inputs, (inputs.shape[0], inputs.shape[1], 1))
  return inputs

#Create model
def createModel(inputs):
  # AI, 2 or 3 layers gives good results
  model.add(layers.LSTM(units = 30, return_sequences = True, input_shape = (inputs.shape[1], 1)))
  model.add(layers.Dropout(0.2))

  model.add(layers.LSTM(units = 30, return_sequences = True))
  model.add(layers.Dropout(0.2))

  model.add(layers.LSTM(units = 30))
  model.add(layers.Dropout(0.2))

  model.add(layers.Dense(units = 1))
  model.compile(optimizer='adam', loss='mse')

# Train model
def trainModel(inputs, realPrice, epochs, batch):
  model.fit(inputs, realPrice, epochs = epochs, batch_size = batch)

# Define the price, model and scaler
prices = getStock('MSFT')
model = Sequential()
scaler = MinMaxScaler(feature_range=(0,1))

# Train AI 
def trainAI(prices, epochs, batch):
  scaled, stockPricesTest, stockPricesLastMonth = prepareData(prices)
  inputs, realPrice = splitTrainTest(scaled)
  reshapedInputs = reshapeData(inputs, realPrice)
  createModel(reshapedInputs)
  trainModel(reshapedInputs, realPrice, epochs, batch)

trainAI(prices, 50, 64)
