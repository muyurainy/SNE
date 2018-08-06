#coding:utf8
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics

class Task(object):
    def __init__(self, Graph):
        self.G = Graph
        self.g = Graph.g

    def link_sign_prediction_split(self, embedding):
        x_train = []
        x_test = []
        y_train = []
        y_test = []
        for line in self.G.train_edges:
            src, tgt, sign = line
            x_train.append(np.concatenate((embedding[src], embedding[tgt])))
            y_train.append(sign)
        for line in self.G.test_edges:
            src, tgt, sign = line
            x_test.append(np.concatenate((embedding[src], embedding[tgt])))
            y_test.append(sign)
        clf = OneVsRestClassifier(LogisticRegression())
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        y_score = clf.predict_proba(x_test)
        acc = 0
        for i in range(len(y_pred)):
            if y_pred[i] == y_test[i]:
                acc += 1
        print "link_sign_prediction acc: {:.3f}, f1-micro: {:.3f}, f1-macro: {:.3f}".format(1.0 * acc/len(y_pred),
                metrics.f1_score(y_test, y_pred, average = 'micro'),
                metrics.f1_score(y_test, y_pred, average = 'macro'))

