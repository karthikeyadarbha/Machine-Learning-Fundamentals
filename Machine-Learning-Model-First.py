import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import Imputer

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


## Data split - Test and Train Data
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)
melbourne_model = DecisionTreeRegressor()
melbourne_model.fit(train_X, train_y)
# get predicted prices on validation data. i.e., The model was built using train data and its been validating wih the test data
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))


## Data pre-processing begins here .. 
# Imputation Method to fulfil missing values

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state = 0)
imputed_X_train_plus = X_train.copy()
imputed_X_test_plus = X_test.copy()
cols_with_missing = (col for col in X_train.columns 
                                 if X_train[col].isnull().any())
for col in cols_with_missing:
    print(imputed_X_train_plus[col])
    imputed_X_train_plus[col + '_was_missing'] = imputed_X_train_plus[col].isnull()
    imputed_X_test_plus[col + '_was_missing'] = imputed_X_test_plus[col].isnull()
    
# Imputation
my_imputer = Imputer()
imputed_X_train_plus = my_imputer.fit_transform(imputed_X_train_plus)
imputed_X_test_plus = my_imputer.transform(imputed_X_test_plus)