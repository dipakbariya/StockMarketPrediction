import numpy as np
import flask
from flask import Flask, request, jsonify, render_template
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
    hashtag1 = request.form['hashtag']
    
    old_df = pd.read_csv("Twitter_stock_final_dataset.csv")
    old_df["Date"] = pd.to_datetime(old_df[['Day','Month','Year']])
    old_df.index=old_df.Date
    
    sc1 = StandardScaler()
    sc2 = StandardScaler()
    sc3 = StandardScaler()
    sc4 = StandardScaler()
    sc5 = StandardScaler()
    old_df.iloc[:,9] = sc1.fit_transform(np.array(old_df.iloc[:,9]).reshape(-1,1))
    old_df.iloc[:,8] = sc2.fit_transform(np.array(old_df.iloc[:,8]).reshape(-1,1))
    old_df.iloc[:,10] = sc3.fit_transform(np.array(old_df.iloc[:,10]).reshape(-1,1))
    old_df.iloc[:,11] = sc4.fit_transform(np.array(old_df.iloc[:,11]).reshape(-1,1))
    old_df.iloc[:,12] = sc5.fit_transform(np.array(old_df.iloc[:,12]).reshape(-1,1))

    from sklearn.preprocessing import LabelEncoder
    le1 = LabelEncoder()
    le2 = LabelEncoder()
    le3 = LabelEncoder()
    old_df.Year = le1.fit_transform(old_df.Year)
    old_df.StockName = le2.fit_transform(old_df.StockName)
    old_df.Day_of_week = le3.fit_transform(old_df.Day_of_week)

    # print(old_df.iloc[0,:])
    # d = pd.DataFrame()
    # d = pd.get_dummies(old_df1.Year, prefix=None, prefix_sep='_', dummy_na=False)
    # old_df1 = pd.concat([old_df1,d], axis=1)
    # old_df1.drop(['Year'], axis=1, inplace=True)

    # d = pd.DataFrame()
    # d = pd.get_dummies(old_df1.Day_of_week, prefix=None, prefix_sep='_', dummy_na=False)
    # old_df1 = pd.concat([old_df1,d], axis=1)
    # old_df1.drop(['Day_of_week'], axis=1, inplace=True)
    old_df.drop(["Date","Total Tweets"], axis=1, inplace=True)
    X = np.array(old_df.drop(["Close"],1))
    y = np.array(old_df.Close)

    from sklearn.ensemble import RandomForestRegressor
    rf_2 = RandomForestRegressor(bootstrap=True, max_depth=80, max_features='sqrt', min_samples_leaf=3, min_samples_split=8, n_estimators=1000, random_state=1)
    rf_2.fit(X,y)
    
    

    if hashtag1=="apple":
      hashtag="AAPL"
    elif hashtag1=="microsoft":
      hashtag="MSFT"
    elif hashtag1=="nvidia":
      hashtag="NVDA"
    elif hashtag1=="paypal":
      hashtag="PYPL"
    elif hashtag1=="tesla":
      hashtag="TSLA"


    def scraper(since, until, hashtag):
      import pandas as pd
      import snscrape.modules.twitter as sntwitter
      tweets_list2 = []
      data=pd.DataFrame()
      for i,tweet in enumerate(sntwitter.TwitterSearchScraper('#'+str(hashtag)+' since:'+str(since)+' until:'+str(until)+' lang:en').get_items()):        # if i>5000:
        tweets_list2.append([tweet.date, tweet.content])

        # Creating a dataframe from the tweets list above
      data = pd.DataFrame(tweets_list2, columns=['Datetime', 'Text'])
      data["Stockname"] = str(hashtag)
      date = until
      return [data, date, hashtag]

    data, date, hashtag = scraper(since, until, hashtag)


    df = data
    def split_date_time(series):
      L1=[]
      L2=[]
      for i in range(len(series)):
        date, time = str(df["Datetime"][i]).split(' ')
        L1.append(date)
        L2.append(time)
      df_1 = pd.DataFrame()
      df_1["Date"] = L1
      df_1["Time"] = L2
      return df_1
    df_1=split_date_time(df["Datetime"])
    df = df.merge(df_1, right_index=True, left_index=True)
    df.drop('Datetime', axis=1, inplace=True)
    df.drop('Time', axis=1, inplace=True)
    def pre_process(df):
        column=df["Text"]


        column = column.str.lower()                                                       # Lower Case
        column = column.apply(lambda x: re.sub(r'https?:\/\/\S+', '', x))                 # URL links
        column = column.apply(lambda x: re.sub(r"www\.[a-z]?\.?(com)+|[a-z]+\.(com)", '', x))       # URL Links
        column = column.apply(lambda x: re.sub(r'{link}', ' ', x))                         # Placeholders
        column = column.apply(lambda x: re.sub(r"\[video\]", ' ', x))                      # Placeholders
        column = column.apply(lambda x: re.sub(r'&[a-z]+;', ' ', x))                       # HTML Functions
        column = column.apply(lambda x: re.sub(r"[^a-z\s\(\-:\)\\\/\];='#]", ' ', x))      # Non Letters
        column = column.apply(lambda x: re.sub(r'@mention', ' ', x))                       # Mentions
        column = column.apply(lambda x: re.sub(r'\n', ' ', x))                             # \n
        column = column.apply(lambda x: re.sub(r'-', '', x))                              # -
        column = column.apply(lambda x: re.sub(r'(\s)#\w+', ' ', x))      # remove word starting from hashtag
        return column

    column=pre_process(df)
    df["clean_text"] = column

    def tokenizer(df):                    


      column = df["clean_text"]
      tknzr = TweetTokenizer()
      column = column.apply(tknzr.tokenize)

      PUNCUATION_LIST = list(string.punctuation)
      def remove_punctuation(word_list):
        """Remove punctuation tokens from a list of tokens"""
        return [w for w in word_list if w not in PUNCUATION_LIST]
      df['tokens'] = column.apply(remove_punctuation)
      return df

    df = tokenizer(df)
    import nltk
    #nltk.download('stopwords')  
    from nltk.corpus import stopwords
    def remove_stopwords(x):
        return [y for y in x if y not in stopwords.words('english')]

    df['temp_list1'] = df['tokens'].apply(lambda x : remove_stopwords(x))


    from nltk.corpus import stopwords
    stopwords = set('rt')
    stopwords.update(['retweet', 'RT', 'Retweet', 'RETWEET', 'rt', 'plz','#aapl','aapl','#msft','msft', 'tsla','tesla','stock','#tsla','elonmusk','apple','#wallstreetbets','reddit','wsbchairman','aapl','#aapl','microsoft'])
    l=[]
    for i in df.temp_list1:
      t = " ".join(review for review in i)
      l.append(t)
    df["temp_list2"] = l
    # textt = " ".join(review for review in df.temp_list2)
    sid = SentimentIntensityAnalyzer()
    ss=[]
    for k in tqdm(df.temp_list2):
      # print(k)
      ss.append(sid.polarity_scores(k))
    neg=[]
    pos=[]
    neu=[]
    compound=[]
    for i in tqdm(range(df.temp_list2.shape[0])):
      neg.append(ss[i]["neg"])
      pos.append(ss[i]["pos"])
      neu.append(ss[i]["neu"])
      compound.append(ss[i]["compound"])
    sia_table = pd.DataFrame()
    sia_table["sia_pos"] = pos
    sia_table["sia_neu"] = neu
    sia_table["sia_neg"] = neg
    sia_table["sia_compound"] = compound

    sentiment=[]
    for i in ss:
      if i['compound'] >= 0.05 :
        sentiment.append("Positive")

      elif i['compound'] <= - 0.05 :
        sentiment.append("Negative")

      else :
        sentiment.append("Neutral")
    df["Sentiment"] = sentiment
    d = pd.get_dummies(df.Sentiment, prefix=None, prefix_sep='_', dummy_na=False)
    df = pd.concat([df,d], axis=1)
    df.drop(['Sentiment'], axis=1, inplace=True)
    year=[]
    month=[]
    day=[]
    D = df.groupby(by='Date').sum()
    D["Stockname"] = hashtag1
    for i in range(len(D.index)):
      year.append((str(D.index[i])).split(' ')[0].split('-')[0])
      month.append((str(D.index[i])).split(' ')[0].split('-')[1])
      day.append((str(D.index[i])).split(' ')[0].split('-')[2])
    #   year.append(a)
    #   month.append(b)
    #   day.append(c)
    # df1["Stockname"] = hashtag
    D["Year"] = year
    D["Month"] = month
    D["Day"] = day
    import yfinance as yf
    new_df = D
    df1 = yf.download(tickers=hashtag, period='30d', interval='1d', rounding ="True",   )
    year=[]
    month=[]
    day=[];
    for i in range(len(df1.index)):
      year.append((str(df1.index[i])).split(' ')[0].split('-')[0])
      month.append((str(df1.index[i])).split(' ')[0].split('-')[1])
      day.append((str(df1.index[i])).split(' ')[0].split('-')[2])
    #   year.append(a)
    #   month.append(b)
    #   day.append(c)

    df1["Stockname"] = hashtag1
    df1["Year"] = year
    df1["Month"] = month
    df1["Day"] = day

    df_all = new_df.merge(df1, how='left', on=['Day', 'Month', 'Year','Stockname'])
    df_all = df_all[['Day','Month','Year','Stockname','Positive','Neutral','Negative','Close','Volume','Open','High','Low']]
    df_all = df_all.sort_values(['Year','Month','Day'])
    df_all["Day_of_week"] = pd.to_datetime(df_all[["Year","Month","Day"]]).dt.day_name()
    df_all.iloc[:,8] = sc1.transform(np.array(df_all.iloc[:,8]).reshape(-1,1))
    df_all.iloc[:,9] = sc3.transform(np.array(df_all.iloc[:,9]).reshape(-1,1))
    df_all.iloc[:,10] = sc4.transform(np.array(df_all.iloc[:,10]).reshape(-1,1))
    df_all.iloc[:,11] = sc5.transform(np.array(df_all.iloc[:,11]).reshape(-1,1))
    df_all.Year = np.int64(df_all.Year)[0]
    df_all.Year = le1.transform(df_all.Year)
    df_all.Stockname = le2.transform(df_all.Stockname)
    df_all.Day_of_week = le3.transform(df_all.Day_of_week)
    test = df_all[["Year","Month","Day","Stockname","Positive","Negative","Neutral","Volume","Open","High","Low","Day_of_week"]]
    pred = rf_2.predict(np.array(test))
    pred = round(sc2.inverse_transform(y_pred)[0],4)

    return render_template('index.html', prediction_text='Predicted Close Price is $ {}'.format(pred))


if __name__ == "__main__":
    app.run(debug=True)


