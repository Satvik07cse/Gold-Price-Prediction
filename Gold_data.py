import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
gold_data = pd.read_csv('gold_price.csv')
#print(gold_data.head())
#print(gold_data.isnull().sum()) #no empty cells
X = gold_data.drop(['Date','GLD'],axis=1) #feature variable
Y = gold_data['GLD'] #Target variable
#splitting X & Y into train & Testing variables
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=2)
regressor = RandomForestRegressor(n_estimators=100)
#Train Model
regressor.fit(X_train,Y_train)
#Model evaluation
test_data_prediction = regressor.predict(X_test)
#Calculate R squared error
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R squared error : ", error_score)
#Comparing The actual value & predicted value
Y_test = list(Y_test)
plt.plot(Y_test,color='blue',label='Actual Value')
plt.plot(test_data_prediction,color='green',label='Predicted Value')
plt.title('Actual Price vs Predicted Price')
plt.xlabel('Number of values')
plt.ylabel('GLD Price')
plt.legend()
plt.show()