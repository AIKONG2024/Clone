from imblearn.over_sampling import SMOTE
from sklearn.datasets import load_wine
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

datasets = load_wine()
x = datasets.data
y = datasets.target
print(x.shape)
print(y.shape)

print(x.shape, y.shape) #(178, 13) (178,)
print(np.unique(y, return_counts=True))
print(pd.value_counts(y))
print(y)

x = x[40:]
y = y[40:]
print(pd.value_counts(y))

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=223)
print(pd.value_counts(y_train))

x_train, y_train = SMOTE(random_state=777).fit_resample(x_train, y_train)
print(pd.value_counts(y_train))

