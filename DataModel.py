import pandas as pd

my_data = pd.read_csv('../input/train.csv')

columns_list = ['Id','MSSubClass']
melbourne_data['MSSubClass']
two_columns = melbourne_data[columns_list]


# Start defining Model here
from sklearn.tree import DecisionTreeRegressor
# Define model
melbourne_model = DecisionTreeRegressor()

X = melbourne_data[columns_list]
y = melbourne_data['Id']

# Fit model
melbourne_model.fit(X, y)