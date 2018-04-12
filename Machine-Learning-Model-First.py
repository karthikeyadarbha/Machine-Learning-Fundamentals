import pandas as pd
from sklearn.tree import DecisionTreeRegressor

##melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
melbourne_file_path = '../input/melbourne-dataset/melb_data.csv'
test_data = pd.read_csv(melbourne_file_path) 
# Exclude all nan values from the columns list
melbourne_data = test_data.dropna(axis=0)
#print(melbourne_data.describe())


## Start building models and predictions

y = melbourne_data.Price
melbourne_predictors = ['Rooms','Distance','Postcode']
X = melbourne_data[melbourne_predictors]


# Defining a model
melbourne_model = DecisionTreeRegressor()
melbourne_model.fit(X,y)
melbourne_model.predict(X.head())