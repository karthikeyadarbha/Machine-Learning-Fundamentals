# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
## Test and Train Data has NaN values which can be replaced using imputers
from sklearn.preprocessing import Imputer




# Data visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

init_df = pd.read_csv('../input/train.csv')
column_values = ['Age','Fare']
train_df = init_df[column_values]
##train_df = init_df.dropna(axis=0)

test_df = pd.read_csv('../input/test.csv')
combine = [train_df, test_df]


## Columns will display the index of the exact names of those columns
print(train_df.columns)
print(train_df.columns.values)

## Able to see NAN values, which needs to be avoided
train_df.head()




imputed_train_df = train_df.copy()
imputed_test_df = test_df.copy()

cols_with_missing = (col for col in train_df.columns 
                                 if train_df[col].isnull().any())

for col in cols_with_missing:
    imputed_train_df[col + '_was_missing'] = imputed_train_df[col].isnull()
    imputed_test_df[col + '_was_missing'] = imputed_test_df[col].isnull()

# Imputation
my_imputer = Imputer()
imputed_train_df = my_imputer.fit_transform(imputed_train_df)
imputed_train_df = my_imputer.transform(imputed_train_df)


## Analysis
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)


## Data Visualization using seaborn

g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)
plt.show()

grid = sns.FacetGrid(train_df, col='Pclass', row='Survived')
grid.map(plt.hist, 'Age')
plt.show()
