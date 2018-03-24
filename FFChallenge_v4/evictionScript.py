import pandas as pd
import numpy as np
from sklearn import svm 
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import brier_score_loss
from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier
from sklearn.neural_network import MLPClassifier

trainX = np.genfromtxt('eBackground1.csv', delimiter=',')
trainY = np.genfromtxt('eLabels1.csv', delimiter=',')
testX = np.genfromtxt('eTest1.csv', delimiter=',')
# sample_weight = np.random.RandomState(42).rand(trainY.shape[0])
# X_new = SelectKBest(chi2, k=2).fit_transform(trainX, trainY)

X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, test_size=0.4, random_state=0)
# X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(X_new, trainY, sample_weight, test_size=0.4, random_state=42)


clf = GaussianNB()
clf.fit(X_train, y_train)
y_predict = clf.predict_proba(X_test)[:, 1]
predictions = clf.predict_proba(testX)[:, 1]
# print y_test.shape
# print y_predict.shape
clf_score = brier_score_loss(y_test, y_predict)
# b = brier_score_loss(y_test, y_predict) 
print("GB: %1.5f" % clf_score)

# clf = svm.SVC(probability=True)
# clf.fit(X_train, y_train)
# y_predict = clf.predict_proba(X_test)[:, 1]
# # print y_test.shape
# # print y_predict.shape
# clf_score = brier_score_loss(y_test, y_predict)
# # b = brier_score_loss(y_test, y_predict) 
# print("SVM: %1.5f" % clf_score)

# clf = Perceptron(penalty='l2')
# clf.fit(X_train, y_train)
# y_predict = clf.predict_proba(X_test)[:, 1]
# # print y_test.shape
# # print y_predict.shape
# clf_score = brier_score_loss(y_test, y_predict)
# # b = brier_score_loss(y_test, y_predict) 
# print("P: %1.3f" % clf_score)

# clf = MLPClassifier()
# clf.fit(X_train, y_train)
# y_predict = clf.predict_proba(X_test)[:, 1]
# # print y_test.shape
# # print y_predict.shape
# clf_score = brier_score_loss(y_test, y_predict)
# # b = brier_score_loss(y_test, y_predict) 
# print("MLP: %1.3f" % clf_score)

# clf = LogisticRegression(solver='saga')
# clf.fit(X_train, y_train)
# y_predict = clf.predict_proba(X_test)[:, 1]
# # print y_test.shape
# # print y_predict.shape
# clf_score = brier_score_loss(y_test, y_predict)
# # b = brier_score_loss(y_test, y_predict) 
# print("LR: %1.5f" % clf_score)

# clf = RidgeClassifier()
# clf.fit(X_train, y_train)
# y_predict = clf.predict_proba(X_test)[:, 1]
# # print y_test.shape
# # print y_predict.shape
# clf_score = brier_score_loss(y_test, y_predict)
# # b = brier_score_loss(y_test, y_predict) 
# print("R: %1.3f" % clf_score)


results = pd.read_csv("prediction.csv", header = 0, low_memory=False)
results["eviction"] = np.array(predictions)
results.to_csv('prediction.csv', index=False)

