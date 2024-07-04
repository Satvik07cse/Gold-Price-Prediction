## Gold Price Prediction Using Random Forest Regressor

This project involves predicting gold prices using a Random Forest Regressor. The dataset used contains historical gold prices and other related financial data. Below is a step-by-step explanation of the code:

### 1. Importing Necessary Libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
```
- numpy: For numerical operations.
- pandas: For data manipulation and analysis.
- matplotlib.pyplot: For plotting graphs.
- sklearn.model_selection: For splitting the dataset into training and testing sets.
- sklearn.ensemble: For using the Random Forest Regressor.
- sklearn.metrics: For evaluating the model.

### 2. Loading the Dataset
```python
gold_data = pd.read_csv('gold_price.csv')
```
- The dataset is loaded from a CSV file named `gold_price.csv`.

### 3. Data Preprocessing
```python
X = gold_data.drop(['Date','GLD'],axis=1) # feature variable
Y = gold_data['GLD'] # Target variable
```
- X: Feature variables (all columns except 'Date' and 'GLD').
- Y: Target variable (the 'GLD' column, representing gold prices).

### 4. Splitting the Dataset
```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=2)
```
- The dataset is split into training and testing sets.
- `test_size = 0.2` means 20% of the data is used for testing, and 80% for training.
- `random_state=2` ensures reproducibility of the results.

### 5. Training the Model
```python
regressor = RandomForestRegressor(n_estimators=100)
regressor.fit(X_train,Y_train)
```
- A Random Forest Regressor model is created with 100 estimators (trees).
- The model is trained using the training data (`X_train`, `Y_train`).

### 6. Model Evaluation
```python
test_data_prediction = regressor.predict(X_test)
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R squared error : ", error_score)
```
- Predictions are made on the testing data (`X_test`).
- The performance of the model is evaluated using the R-squared error metric.

### 7. Visualizing Actual vs. Predicted Prices
```python
Y_test = list(Y_test)
plt.plot(Y_test, color='blue', label='Actual Value')
plt.plot(test_data_prediction, color='green', label='Predicted Value')
plt.title('Actual Price vs Predicted Price')
plt.xlabel('Number of values')
plt.ylabel('GLD Price')
plt.legend()
plt.show()
```
- The actual gold prices (`Y_test`) and the predicted prices (`test_data_prediction`) are plotted.
- The blue line represents the actual prices, and the green line represents the predicted prices.

---
### Conclusion :

In this project, we successfully implemented a Random Forest Regressor to predict gold prices using historical financial data. The key steps involved loading and preprocessing the data, splitting the dataset into training and testing sets, training the model, evaluating its performance, and visualizing the results.

The R-squared error metric, which indicates how well the predictions matched the actual values, was used to assess the model's accuracy. The model achieved an impressive R-squared error of 0.98, indicating a high level of accuracy in its predictions. The final visualization illustrated the comparison between actual and predicted gold prices, demonstrating the model's effectiveness.
