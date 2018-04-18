import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import Imputer

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence

from sklearn.model_selection import cross_val_score


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


train_X, test_X, train_y, test_y = train_test_split(X, y,random_state = 0)
my_model = XGBRegressor()
my_model.fit(train_X, train_y, verbose=False)
predictions = my_model.predict(test_X)

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))

my_model = XGBRegressor(n_estimators=1000)
my_model.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], verbose=False)


def get_some_data():
    cols_to_use = ['Distance', 'Landsize', 'BuildingArea']
    data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')
    y = data.Price
    X = data[cols_to_use]
    my_imputer = Imputer()
    imputed_X = my_imputer.fit_transform(X)
    return imputed_X, y
    


X, y = get_some_data()

# scikit-learn originally implemented partial dependence plots only for Gradient Boosting models
my_model = GradientBoostingRegressor()

# fit the model as usual
my_model.fit(X, y)
# Here we make the plot
my_plots = plot_partial_dependence(my_model,       
                                   features=[0, 2], # column numbers of plots we want to show
                                   X=X,            # raw predictors data.
                                   feature_names=['Distance', 'Landsize', 'BuildingArea'], # labels on graphs
                                   grid_resolution=10)
imputed_X_test_plus = my_imputer.transform(imputed_X_test_plus)


my_pipeline = make_pipeline(Imputer(), RandomForestRegressor())

my_pipeline.fit(train_X, train_y)
predictions = my_pipeline.predict(test_X)

my_imputer = Imputer()
my_model = RandomForestRegressor()

imputed_train_X = my_imputer.fit_transform(train_X)
imputed_test_X = my_imputer.transform(test_X)
my_model.fit(imputed_train_X, train_y)
predictions = my_model.predict(imputed_test_X)


scores = cross_val_score(my_pipeline, X, y, scoring='neg_mean_absolute_error')
print(scores)
print('Mean Absolute Error %2f' %(-1 * scores.mean()))
