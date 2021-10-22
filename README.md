# Prediction of Stock Market Price Using Macro Economic Headlines

###  <p align="center">  INTRODUCTION </p>  


<p align="justify"> The stock market is one of the most sensitive fields, where the sentiments of the people can change the trend of the entire market. Actually, there are many factors, affect the movement of the stock market and, the sentiments of the traders are also one of them that drive the market. </p>

<p align="justify"> The volatile nature of the stock market has equal chances for earning money and losing money as well. But if the situation can be predicted, investors can make a profit or minimize their losses. </p>

> **Actually, when a piece of news comes in the market, people start talking about and give their positive or negative opinions that show their sentiments. That can be used by the sentiment analysis experts to predict the movement of the stock market or particular stock of a company.**  

<p align="justify"> On June 2015, 2016 debt negotiations between Greek Govt and its creditors borke off abrubptly. Large market movements as a concequence of political and economic headlines are hardly uncommon, liquid markets are most suspectable to swing when the news breaks. Using VIX as a proxy for market volatality, we investigate how macroeconoic headlines affect the changes. Here, we predict equity market value using tweets from major news sources, investment banks and notable economists. </p>  

![](https://www.bing.com/images/search?view=detailV2&ccid=EzQO0Kqy&id=82F8ABFBDBEB090EF5A5905C85E7286030925F9E&thid=OIP.EzQO0KqyVH5vMxlIwA57sAAAAA&mediaurl=https%3a%2f%2fmedia4.giphy.com%2fmedia%2fAgHBbekqDik0g%2f200.gif%3fcid%3d790b7611jw0t6pxn719hhtfxo3mdyr9q6908eo8va6pjqbrn%26rid%3d200.gif&exph=200&expw=357&q=stock+market+gif&simid=608024492889482897&FORM=IRPRST&ck=ED30CE5021E16F8813B23562D5BDD09B&selectedIndex=94&ajaxhist=0&ajaxserp=0)

_________________________________________________________________________________________________________________________________________________________________________________

###  <p align="center"> **PROBLEM**  </p>  


<p align="justify"> Twitter provides a plethora of market data. In this project we have extracted around 100,000 tweets from various accounts to predict the upward movements. Using this data we are researching how this economic news affects the market. </p>

<p align="justify"> The sentiment analysis task is very much field-specific. Tweets are classified as positive, negative, and neutral based on the sentiment present.
Out of the total tweets are examined by humans and annotated as 1 for Positive, 0 for Neutral and 2 for Negative emotions. For the classification of nonhuman annotated tweets, a machine learning model is trained whose features are extracted from the human-annotated tweets. </p>

<p align="justify"> Except, in extreme or unexpected conditions, most of the time, machine learning or deep learning-based models predict at very high accuracy helping stock market investors to earn money. </p>

*********************************************************************************************************************************************************************************

### <p align="center"> **EXPLARATORY DATA ANALYSIS**  </p>  

<p align="justify"> EDA includes extracting the twitter data based on the stock names viz, Apple, Tesla, Nvidia, Paypal and Microsoft, cleaning of twitter data that were pulled i.e., removing unnecessary words from tweets. After cleaning the tweets, below are the plots that were plotted against the sentiments that is Positive, Negative and Neutral. </p>

> Most common positive words:
![image](https://user-images.githubusercontent.com/63631974/137613734-4f797f13-9eb9-4bd1-955b-37e5e801d48e.png) 


> Most common negative words:
![image](https://user-images.githubusercontent.com/63631974/137613755-a28a127d-d35d-4239-bec5-61e2ed541a63.png)

> Pie chart of sentiments:
![image](https://user-images.githubusercontent.com/63631974/137678423-64ed2362-e759-4cb1-b954-97924bca5b01.png)

> Wordcloud
![image](https://user-images.githubusercontent.com/63631974/137678505-6e071b41-c3a8-4b20-93cc-a39e0f968749.png)


---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


###  <p align="center"> **TYPE OF MACHINE LEARNING MODEL**  </p>  

<p align="justify"> This project is Regression based problem, which is a predictive modelling technique that analyzes the relation between the target or dependent variable and independent variable in a dataset. </p>

<p align="justify"> METRICS USED: The performance of a regression model must be reported as an error in those predictions and these error summarizes on average how close the predictions were to their expected values. </p>

Accuracy mectrics we have used in this project are:

* Root Mean Squared Error(RMSE) 
* Mean Absolute Error(MAE) 
* Rsquared value(r2) 

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### <p align="center"> **MODELLING**  </p>  

<p align="justify"> We have implemented differnt ML models, i.e. Linear Regression, Random Forest Regression, Decision Tree Regressor, Support vector regressor. We have choosen Random Forest Regression model for our project as its r2 - 0.99964, rmse - 0.08. We have choosed Random Forest Regressor model as it classifies decision trees on various subsamples and uses averaging to imporve the predictive accuracy and control overfitting. So, our team decided to apply Random Forest Regression model for our project. </p>

![image](https://user-images.githubusercontent.com/63631974/137613663-88e25be4-b40f-4876-87eb-2280d14a6ad3.png)

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 
### <p align="center"> **DEPLOYMENT**  </p>  

<p align="justify"> We have deployed the model using Flask framework, as it is a opensource Python library that allows us to create beautiful web apps for Machine Learning. It is hosted on Heroku server, as it a container based Platform As A Service(PAAS), because it is flexible and easy to host on this platform. </p>

![image](https://user-images.githubusercontent.com/63631974/137613691-93de716f-224e-4abf-ba58-fc7a9eac38ae.png)

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### <p align="center"> **LINKS**  </p>  
* Application Link
[Click Here](https://stockpriceprediction0.herokuapp.com/)

* Demo Video
[Click Here](https://github.com/dipakbariya/StockMarketPrediction/blob/main/2021-10-16%2012-45-49.mkv)


---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### <p align="center"> **TEAM MEMBERS & CONTACT DETAILS**  </p>  
* Dipak Bariya: [Linkedin](https://linkedin.com/in/dipak-bariya/)  [GitHub](https://github.com/dipakbariya/)
* Harshit Singh: [Linkedin](https://linkedin.com/in/harshit-singh-92ba49187) [GitHub](https://github.com/HarshuSingh)
* Meet Panchal: [Linkedin](https://www.linkedin.com/in/meet-panchal011/) [GitHub](https://github.com/Meet-011)
* Pratik Thakur: [Linkedin](https://www.linkedin.com/in/pratik-thakur-661506194/)  [GitHub](https://github.com/thakur-pratik)
