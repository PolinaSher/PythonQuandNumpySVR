import quandl
import numpy as np
from sklearn.liner_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

quandl.ApiConfig.api_key = "wuh_Rrir2hmK5BwdKdV8"
#DataFlair - Get Amazon stock data
amazon = quandl.get("WIKI/AMZN")
print(amazon.head())

#DataFlair - Get only the data for the Adjusted Close column
amazon = amazon[['Adj. Close']]
print(amazon.head())


#DataFlair - Predict for 30 days; Predicted has the data of Adj. Close shifted up by 30 rows
forecast_len=30
amazon['Predicted'] = amazon[['Adj. Close']].shift(-forecast_len)
print(amazon.tail())

#DataFlair - Drop the Predicted column, turn it into a NumPy array to create dataset
x=np.array(amazon.drop(['Predicted'],1))
#DataFlair - Remove last 30 rows
x=x[:-forecast_len]
print(x)


#DataFlair - Create dependent dataset for predicted values, remove the last 30 rows
y=np.array(amazon['Predicted'])
y=y[:-forecast_len]
print(y)

#DataFlair - Split datasets into training and test sets (80% and 20%)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

#DataFlair - Create SVR model and train it
svr_rbf=SVR(kernel='rbf',C=1e3,gamma=0.1) 
svr_rbf.fit(x_train,y_train)

#DataFlair - Get score
svr_rbf_confidence=svr_rbf.score(x_test,y_test)
print(f"SVR Confidence: {round(svr_rbf_confidence*100,2)}%")

   

#DataFlair - Create Linear Regression model and train it
lr=LinearRegression()
lr.fit(x_train,y_train)

#DataFlair - Get score for Linear Regression
lr_confidence=lr.score(x_test,y_test)
print(f"Linear Regression Confidence: {round(lr_confidence*100,2)}%")
#mydata = quandl.get("FRED/GDP", start_date="2001-12-31", end_date="2020-12-31")
#mydata = quandl.get(["NSE/OIL.1", "WIKI/AAPL.4"])
#mydata = quandl.get("WIKI/AAPL", rows=5)
#mydata = quandl.get("EIA/PET_RWTC_D", collapse="monthly")
#mydata = quandl.get("FRED/GDP", transformation="rdiff")
#mydata = quandl.get_table('ZACKS/FC', ticker='AAPL')
#mydata = quandl.get_table('ZACKS/FC', paginate=True, ticker='AAPL', qopts={'columns': ['ticker', 'per_end_date']})
#print(mydata)

