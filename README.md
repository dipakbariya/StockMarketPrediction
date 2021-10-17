**# Prediction of Stock Market Price Using Macro Economic Headlines**

**INTRODUCTION**

On June 2015, 2016 debt negotiations between Greek Govt and its creditors borke off abrubptly. Large market movements as a concequence of political and economic headlines are hardly uncommon, liquid markets are most suspectable to swing when the news breaks. Using VIX as a proxy for market volatality, we investigate how macroeconoic headlines affect the changes. Here, we predict equity market value using tweets from major news sources, investment banks and notable economists.


**Problem**

Twitter provides a plethora of market data. In this project we have extracted around 100,000 tweets from various accounts to predict the upward movements. Using this data we are researching how this economic news affects the market.


**Explaratory Data Analysis:**

EDA includes extracting the twitter data based on the stock names viz, Apple, Tesla, Nvidia, Paypal and Microsoft, cleaning of twitter data that were pulled i.e., removing unnecessary data from tweets. After cleaning the data, below are the plots that were plotted against the sentiments that is Positive, Negative and Neutral.

>Most common positive words
![image](https://user-images.githubusercontent.com/63631974/137613734-4f797f13-9eb9-4bd1-955b-37e5e801d48e.png)


>Most common negative words
![image](https://user-images.githubusercontent.com/63631974/137613755-a28a127d-d35d-4239-bec5-61e2ed541a63.png)





**Type of Machine Learning**

This project is Regression based problem, which is a predictive modelling technique that analyzes the relation between the target or dependent variable and independent variable in a dataset.

METRICS USED: The performance of a regression model must be reported as an error in those predictions and these error summarizes on average how close the predictions were to their expected values.

Accuracy mectrics we have used in this project are:

Root Mean Squared Error(RMSE) 
Mean Absolute Error(MAE) 
Rsquared value(r2) 

**Modelling**

We have implemented differnt ML models Linear Regression, Random Forest Regression, Decision Tree Regressor. We have choosen Random Forest Regression ML for our project as its r2 - 0.99964, rmse - 3.65. We have choosed Random Forest Regressor model as it classifies decision trees on various subsamples and uses averaging to imporve the predictive accuracy and control overfitting. So, our team decided to apply Random Forest Regression model for our project.

![image](https://user-images.githubusercontent.com/63631974/137613663-88e25be4-b40f-4876-87eb-2280d14a6ad3.png)


**Deployment**

We have deployed the model using Streamlit framework, as it is a opensource Python library that allows us to create beautiful web apps for Machine Learning. It is hosted on Heroku, as it a container based Platform As A Service(PAAS), because it is flexible and easy to host on this platform.

![image](https://user-images.githubusercontent.com/63631974/137613691-93de716f-224e-4abf-ba58-fc7a9eac38ae.png)


**Links :**
Application Link
[https://stockpriceprediction0.herokuapp.com/](Click Here)

Demo Video
[https://github.com/dipakbariya/StockMarketPrediction/blob/main/2021-10-16%2012-45-49.mkv](Click Here)


