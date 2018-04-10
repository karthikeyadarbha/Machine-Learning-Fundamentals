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


## Define a function which specifies the Mean absolute error for Train and Test Data 
def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    mae = mean_absolute_error(targ_val, preds_val)
    return(mae)

# Invoke the above function

train_X = X
train_y = y
val_y = y
val_X= X
# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
	

from sklearn.ensemble import RandomForestRegressor
forest_model = RandomForestRegressor()
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))


# Read the test data
test = pd.read_csv('../input/test.csv')

#print(test)
# Treat the test data in the same way as training data. In this case, pull same columns.
predictor_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd','MSSubClass','ScreenPorch','MiscVal']
test_X = test[predictor_cols]
# Use the model to make predictions
predicted_prices = my_model.predict(test_X)


my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
my_submission.to_csv('submission.csv', index=False)