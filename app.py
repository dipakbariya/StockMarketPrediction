import numpy as np
import flask
from flask import Flask, request, jsonify, render_template
import pickle
from model_test import *
from functions import scraper, data_cleaner
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import snscrape.modules.twitter as sntwitter
import pandas as pd
#Initializing the application name [here, the name is app]
app = Flask(__name__)

#Loading the model created in model.py
#model = pickle.load(open('model.pkl', 'rb'))

#Starting the app by rendering the index.html page
@app.route('/')
def home():
    return render_template('index.html')



#Calling the prediction function using the POST method
@app.route('/predict',methods=['POST'])
def predict():
    since = request.form['since_date']
    until = request.form['until_date']
    hashtag = request.form['hashtag']
    data, date, hashtag = scraper(since, until, hashtag)
    df = data_cleaner(data, date, hashtag)
    pred = model(df, date)
    return render_template('index.html', prediction_text='Predicted Close Price is $ {}'.format(pred))


if __name__ == "__main__":
    app.run(debug=True)


