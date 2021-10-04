from functions import scraper, data_cleaner
def model(df2, date):        #New Prediction
  from sklearn.preprocessing import StandardScaler
  import pandas as pd
  import numpy as np
  from sklearn.preprocessing import LabelEncoder
  old_df = pd.read_csv("Twitter_stock_final_dataset.csv")
  old_df["Date"] = pd.to_datetime(old_df[['Day','Month','Year']])
  old_df.index=old_df.Date
  # df.drop(['Day','Month','Year','Date','Total Tweets','Open','High','Low','Volume'], axis=1, inplace=True)
  le=LabelEncoder()
  le1=LabelEncoder()

  old_df.StockName = le.fit_transform(old_df.StockName)
  old_df.Year = le1.fit_transform(old_df.Year)

  sc1 = StandardScaler()
  sc2 = StandardScaler()
  old_df.iloc[:,9] = sc1.fit_transform(np.array(old_df.iloc[:,9]).reshape(-1,1))
  old_df.iloc[:,8] = sc2.fit_transform(np.array(old_df.iloc[:,8]).reshape(-1,1))

  old_df = old_df[["Year","StockName","Positive","Negative","Neutral","Close","Volume"]]
  d = pd.DataFrame()
  d = pd.get_dummies(old_df.StockName, prefix=None, prefix_sep='_', dummy_na=False)
  old_df1 = pd.concat([old_df,d], axis=1)
  old_df1.drop(['StockName'], axis=1, inplace=True)
  # d = pd.get_dummies(df.Year, prefix=None, prefix_sep='_', dummy_na=False)
  # df1 = pd.concat([df,d], axis=1)
  d = pd.DataFrame()
  d = pd.get_dummies(old_df1.Year, prefix=None, prefix_sep='_', dummy_na=False)
  old_df1 = pd.concat([old_df1,d], axis=1)
  old_df1.drop(['Year'], axis=1, inplace=True)

  X = np.array(old_df1.drop(["Close"],1))
  y = np.array(old_df1.Close)



















  a, b, c = date.split('-')
  pred_data = df2[df2.Day == c]
  columns=['Positive', 'Negative', 'Neutral', 'Close', 'Volume', 0, 1, 2, 3, 4, 0, 1]
  pred_data1=pd.DataFrame(columns=columns)
  pred_data1["Positive"] = pred_data["pos_count"]
  pred_data1["Negative"] = pred_data["neg_count"]
  pred_data1["Neutral"] = pred_data["neu_count"]
  pred_data1["Close"] = pred_data["Close"]
  pred_data1["Volume"] = pred_data["Volume"]
  pred_data1.index = pred_data.index

  for i in range(pred_data.shape[0]):
    pred_data.Year[i]= np.int64(pred_data.Year[i])
    try:
      if pred_data["Stockname"][i] == "AAPL":
        pred_data1.iloc[:,5] = 1

      elif pred_data["Stockname"][i] == "MSFT":
        pred_data1.iloc[:,6] = 1

      elif pred_data["Stockname"][i] == "NVDA":
        pred_data1.iloc[:,7] = 1

      elif pred_data["Stockname"][i] == "TSLA":
        pred_data1.iloc[:,9] = 1

      elif pred_data["Stockname"][i] == "PYPL":
        pred_data1.iloc[:,8] = 1
    except:
      continue
    if pred_data["Year"][i] == 2020:
      pred_data1.iloc[:,10] = 1

    elif pred_data["Year"][i] == 2021:
      pred_data1.iloc[:,11] = 1


  pred_data1.iloc[:,4] = sc1.transform(np.array(pred_data1.iloc[:,4]).reshape(-1,1))
  pred_data1 = pred_data1.replace (np.nan, 0)
  X = np.array(old_df1.drop(["Close"],1))
  y = np.array(old_df1.Close)
  

  test = pred_data1.drop('Close', axis=1)
  from sklearn.ensemble import RandomForestRegressor
  rf_2 = RandomForestRegressor(bootstrap=True, max_depth=80, max_features='sqrt', min_samples_leaf=3, min_samples_split=8, n_estimators=1000, random_state=1)
  rf_2.fit(X,y)
  y_pred = rf_2.predict(np.array(test))
  y_pred = sc2.inverse_transform([y_pred])
 

  
  return y_pred
  
