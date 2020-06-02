from datetime import date, timedelta

import yfinance as yf
import pandas as pd
from pandas_datareader import data as pdr

from app import app, db
from models.prices import PriceModel

yf.pdr_override()

@app.cli.command("get-prices")
def getPrices():
  stockList = ['MSFT', 'AAPL', 'AMZN', 'FB', 'GOOGL', 'GOOG', 'GOOGL', 'INTC', 'NVDA', 'CSCO', 'ADBE', 'NFLX', 'PEP', 'PYP', 'CMCSA', 'TSLA', 'COST', 'GOOGL', 'AMGN', 'TMUS', 'AVGO', 'CHTR', 'TXN', 'GILD', 'QCOM', 'SBUX', 'INTU', 'MDLZ', 'VRTX', 'FISV', 'BKNG', 'ISRG', 'REGN', 'ADP', 'AMD', 'ATVI', 'CSX', 'BIIB', 'ILMN', 'MU', 'AMAT', 'JD', 'ADSK', 'MELI', 'ADI', 'LRCX', 'MNST', 'WBA', 'EXC', 'KHC', 'LULU', 'EA']

  for stock in stockList:
    getPrice(stock)

def getPrice(stock):
  today = date.today()
  monthAgo = today - timedelta(days=31)
  data = pdr.get_data_yahoo(stock, start=monthAgo, end=today)
  df = pd.DataFrame(data)
  for index, row in df.iterrows():
    addPrice(row, index.strftime("%Y-%m-%d"), stock)

def addPrice(price, date, stock):
  new_price = PriceModel(date=date, company=stock, openPrice=price['Open'], highPrice=price['High'], lowPrice=price['Low'], closePrice=price['Close'], volume=price['Volume'])
  db.session.add(new_price)
  db.session.commit()
