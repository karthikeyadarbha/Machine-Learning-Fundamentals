from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import datasets

irisData = datasets.load_iris()
X = irisData.data[:,[2,3]]
Y = irisData.target
np.unique(Y) ## To print Unique value from the class - Typically the distinct values of the classifier . ie., irirs, Satasoa and one more
X_train,X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.3) ## test_sizw = 0.3 which is 30% Train Data and rest 70 percent is test Data


## Performance scaling - for Test and Train Data
from sklearn.preprocessing import StandardScaler
sd = StandardScaler()

sc.fit(X_train)
X_test_std = sc.transform(X_test)
X_train_std = sc.transform(X_train)


## Apply perceptron rule on the test and train set
from sklearn.preprocessing import Perceptron

ppn = Perceptron()
pred_val = ppn.fit(X_train_std,y_train)

## To verify the accuracy in the predicted result. Compare the predicted value
## with test class labels.
from sklearn.metrics import accuracy_score
accuracy_score(y_test,pred_val)


