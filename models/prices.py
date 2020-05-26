from app import db

class PriceModel(db.Model):
    __tablename__ = 'prices'

    id = db.Column(db.Integer, primary_key=True)
    company = db.Column(db.String())
    date = db.Column(db.Integer())
    openPrice = db.Column(db.Integer())
    highPrice = db.Column(db.Integer())
    lowPrice = db.Column(db.Integer())
    closePrice = db.Column(db.Integer())
    volume = db.Column(db.Integer())

    def __init__(self, company, date, openPrice, highPrice, lowPrice, closePrice, volume):
        self.company = company
        self.date = date
        self.openPrice = openPrice
        self.highPrice = highPrice
        self.lowPrice = lowPrice
        self.closePrice = closePrice
        self.volume = volume

    def __repr__(self):
        return f"<Price for {self.company} in day {self.date} is {self.closePrice}>"