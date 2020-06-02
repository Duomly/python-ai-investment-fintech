import psycopg2
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

# Define app and setup db connection
app = Flask(__name__)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = "postgresql://postgres:postgres@localhost:5432/playground"

# Connect with db and import models
db = SQLAlchemy(app)
from models.prices import PriceModel
from prices.prices import getPrices
migrate = Migrate(app, db)

# App run
if __name__ == '__main__':
    app.run(debug=True)
