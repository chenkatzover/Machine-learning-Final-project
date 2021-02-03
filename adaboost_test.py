from sklearn.model_selection import train_test_split
import pandas as pd
from Point import Point
from adaboost import Adaboost
import numpy as np

def run_adaboost(X, y, data1, data2):
    # init avg
    avg_train = np.zeros(8)
    avg_test = np.zeros(8)

    # runs
    n = 50
    total_avg = 0
    for _ in range(n):
        # split the data to train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=5)
        train_x = []
        test_x = []
        train_y = y_train.to_numpy()
        test_y = y_test.to_numpy()
        for i, row in X_train.iterrows():
            p = Point(row[data1], row[data2])
            train_x.append(p)

        for i, row in X_test.iterrows():
            p = Point(row[data1], row[data2])
            test_x.append(p)

        # call adaboost
        ad = Adaboost()
        ad.train(train_x, train_y)
        predicts = ad.predict(test_x, test_y)
        sum = 0
        for i in range(len(predicts)):
            if predicts[i] == test_y[i]:
                sum += 1
        total_avg += sum / len(predicts)

    print("אחוז הצלחה", total_avg / n)
# end run adaboost

# import Students Performances
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
run_adaboost(sp[["ethnicity", "average"]], sp["gender"], "ethnicity", "average")
# print(sp)
