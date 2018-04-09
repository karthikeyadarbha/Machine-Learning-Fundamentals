import pandas as pd
# This can be used to predict the results
from sklearn.tree import DecisionTreeRegressor
# Validate the predicted results
from sklearn.metrics import mean_absolute_error


## Data wrangling begin - Extract the required data for analysis and predictions
my_data = pd.read_csv('../input/train.csv')
columns_list = ['Id','MSSubClass']
melbourne_data['MSSubClass']
two_columns = melbourne_data[columns_list]


# Start defining Model here
from sklearn.tree import DecisionTreeRegressor
# Define model
columns_list = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']

X = melbourne_data[columns_list]
y = melbourne_data['YearBuilt']

my_model = DecisionTreeRegressor()
my_model.fit(X, y)

### Validating the model begin here

# start finding the error value from the predicted value
predicted_value = my_model.predict(X)

# This will give the error value based on the predicted value
mean_absolute_error(y, predicted_value)
## The result is 0. Hence, there is no error in the predicted result