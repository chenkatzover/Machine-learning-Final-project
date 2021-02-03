import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

# bankdata = pd.read_csv("D:/Datasets/bill_authentication.csv")
# bankdata.shape
# bankdata.head()

sp = pd.read_csv("StudentsPerformance.csv", sep=",", header=1)
sp.columns = ["gender", "ethnicity", "c", "d", "e", "f", "g", "h"]
sp['average'] = sp[['f', 'g', 'h']].mean(axis=1)
sp["gender"].replace(["female"], 1, inplace=True)
sp["gender"].replace(["male"], -1, inplace=True)
sp["ethnicity"].replace(["group A"], 0, inplace=True)
sp["ethnicity"].replace(["group B"], 1, inplace=True)
sp["ethnicity"].replace(["group C"], 2, inplace=True)
sp["ethnicity"].replace(["group D"], 3, inplace=True)
sp["ethnicity"].replace(["group E"], 4, inplace=True)

X = sp[["ethnicity", "average"]]
y = sp["gender"]
total_sum = 0
n = 15
for _ in range(n):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
    test_y = y_test.to_numpy()

    svclassifier = SVC(kernel='linear')
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)

    sum = 0
    for i in range(len(y_pred)):
        if y_pred[i] == test_y[i]:
            sum += 1

    total_sum += sum / len(y_pred)
print(total_sum / n)
# print(confusion_matrix(y_test, y_pred))
# print("********")
# print(classification_report(y_test, y_pred))