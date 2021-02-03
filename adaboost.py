import numpy as np
import Point
import pandas as pd


# Decision stump used as weak classifier
class Classifier:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        self.m = (p1.y - p2.y) / (p1.x - p2.x + (1e-10))
        self.n = p1.y - (self.m * p1.x)
        self.alpha = None
        self.error = None

    def predict(self, X):
        predictions = np.ones(len(X))
        count = 0
        for p in X:
            if p.y < ((self.m * p.x) + self.n):
                predictions[count] = -1
            count += 1
        return predictions

    def single_predict(self, p):
        if p.y < (self.m * p.x + self.n):
            return -1
        return 1


# end Classifier

class Adaboost:
    def __init__(self):
        self.clfs = []
        self.best_clfs = []

    def train(self, X, y):
        n_samples = len(X)
        w = np.full(n_samples, (1 / n_samples))
        min_error = float('inf')

        # init clfs
        for i in range(n_samples):
            k = i + 1
            for j in range(k, n_samples):
                c = Classifier(X[i], X[j])
                # calculate error for clf
                predictions = c.predict(X)

                # update error of clf
                misclassified = w[y != predictions]
                error = sum(misclassified)  # Et(H)
                c.error = error
                # print(error)
                if error < min_error:
                    min_error = error


                # update alpha
                EPS = 1e-10
                c.alpha = 0.5 * np.log((1.0 - min_error + EPS) / (min_error + EPS))

                # update points weight
                w *= np.exp(-c.alpha * y * predictions)
                w /= np.sum(w)

                # Save classifier
                self.clfs.append(c)
            #end for clfs

        # #sort the clf
        # def sortByError(elem):
        #     return elem.error
        # self.clfs.sort(key=sortByError)
        # self.best_clfs = [self.clfs[i] for i in range(8)] # take the 8 first best clf
        # # for clf in self.clfs:
        # #     print(clf.error)
    # end fit

    def predict(self, points, true_y):
        clf_preds = [clf.alpha * clf.predict(points) for clf in self.clfs]
        y_pred = np.sum(clf_preds, axis=0)
        y_pred = np.sign(y_pred)
        return y_pred

    def sum_of_rights(self, points, y):
        avg = 0
        for c in self.clfs:
            sum = 0
            for i in range(len(points)):
                if c.single_predict(points[i]) == y[i]:
                    sum += 1
            avg += sum / len(points)
        return avg
# end adaboost
