import pandas as pd
import logistic_regression
import adaboost_test
import SVM
import KNN
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics

sp = pd.read_csv("StudentsPerformance.csv", sep=",", header=1)
sp.columns = ['gender', 'ethnicity', 'education', 'lunch', 'test_preparation_course', 'math', 'reading', 'writing']
sp['average'] = sp[['math', 'reading', 'writing']].mean(axis=1)
sp['gender'].replace(['female'], -2, inplace=True)
sp['gender'].replace(['male'], -1, inplace=True)
sp['ethnicity'].replace(["group A"], 0, inplace=True)
sp['ethnicity'].replace(["group B"], 1, inplace=True)
sp['ethnicity'].replace(["group C"], 2, inplace=True)
sp['ethnicity'].replace(["group D"], 3, inplace=True)
sp['ethnicity'].replace(["group E"], 4, inplace=True)
sp['education'].replace(['bachelor\'s degree'], 5, inplace=True)
sp['education'].replace(['associate\'s degree'], 6, inplace=True)
sp['education'].replace(['master\'s degree'], 7, inplace=True)
sp['education'].replace(['some high school'], 8, inplace=True)
sp['education'].replace(['high school'], 9, inplace=True)
sp['education'].replace(['some college'], 10, inplace=True)
sp['lunch'].replace(['standard'], 11, inplace=True)
sp['lunch'].replace(['free/reduced'], 12, inplace=True)
sp['test_preparation_course'].replace(['none'], 13, inplace=True)
sp['test_preparation_course'].replace(['completed'], 14, inplace=True)

def question1():
    X = sp[['ethnicity', 'math', 'reading', 'writing']]
    X = MinMaxScaler().fit_transform(X)
    y = sp['gender']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
    run_and_print(X_train, X_test, y_train, y_test, 'linear')
# end q1

def question2():
    sum = [0, 0, 0, 0]
    for i in range(5):
        for j in range(i+1,5):
            print("i: ", i , "j: ", j)
            S = sp
            for k in range(5):
                if k != i and k != j:
                    S = S[S['ethnicity'] != k]
            # print(S)
            X = S[['gender', 'education', 'lunch', 'test_preparation_course', 'math', 'reading', 'writing']]
            X = MinMaxScaler().fit_transform(X)
            y = S['ethnicity']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

            kernel_mode = 'linear'
            log_pred = logistic_regression.run_log(X_train, X_test, y_train)
            ad_pred = adaboost_test.run_adaboost(X_train, X_test, y_train)
            svm_pred = SVM.run_svm(X_train, X_test, y_train, kernel_mode)
            knn_pred = KNN.run_KNN(X_train, X_test, y_train)

            sum[0] += metrics.accuracy_score(y_test, log_pred)
            sum[1] += metrics.accuracy_score(y_test, ad_pred)
            sum[2] += metrics.accuracy_score(y_test, svm_pred)
            sum[3] += metrics.accuracy_score(y_test, knn_pred)

    for i in range(4):
        sum[i] = sum[i] / 10
    print(sum)
# end q2

def question3():
    S = sp
    S = S[S['education'] != 5]
    S = S[S['education'] != 7]
    S = S[S['education'] != 9]
    S = S[S['education'] != 14]
    X = S[['ethnicity', 'math', 'reading', 'writing']]
    X = MinMaxScaler().fit_transform(X)
    y = S['education']


    sum = [0, 0, 0, 0]
    for i in range(5,11):
        for j in range(i+1,11):
            print("i: ", i , "j: ", j)
            S = sp
            for k in range(6):
                if k != i and k != j:
                    S = S[S['education'] != k]
            # print(S)
            X = S[['ethnicity', 'math', 'reading', 'writing']]
            X = MinMaxScaler().fit_transform(X)
            y = S['education']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

            kernel_mode = 'linear'
            log_pred = logistic_regression.run_log(X_train, X_test, y_train)
            ad_pred = adaboost_test.run_adaboost(X_train, X_test, y_train)
            svm_pred = SVM.run_svm(X_train, X_test, y_train, kernel_mode)
            knn_pred = KNN.run_KNN(X_train, X_test, y_train)

            sum[0] += metrics.accuracy_score(y_test, log_pred)
            sum[1] += metrics.accuracy_score(y_test, ad_pred)
            sum[2] += metrics.accuracy_score(y_test, svm_pred)
            sum[3] += metrics.accuracy_score(y_test, knn_pred)

    for i in range(4):
        sum[i] = sum[i] / 15
    print(sum)
# end q3

def run_and_print(X_train, X_test, y_train, y_test, kernel_mode):
    log_pred = logistic_regression.run_log(X_train, X_test, y_train)
    ad_pred = adaboost_test.run_adaboost(X_train, X_test, y_train)
    svm_pred = SVM.run_svm(X_train, X_test, y_train, kernel_mode)
    knn_pred = KNN.run_KNN(X_train, X_test, y_train)

    print("percent of success in logistic: ", metrics.accuracy_score(y_test, log_pred))
    print("percent of success in adaboost: ", metrics.accuracy_score(y_test, ad_pred))
    print("percent of success in SVM: ", metrics.accuracy_score(y_test, svm_pred))
    print("percent of success in KNN: ", metrics.accuracy_score(y_test, knn_pred))


# question1()
# question2()
question3()
