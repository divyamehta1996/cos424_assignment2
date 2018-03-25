import pandas as pd
import numpy as np
from sklearn import svm 
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_regression
from sklearn.metrics import brier_score_loss
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier

#Binary: Eviction, Layoff, Jobtraining 
#Continuous: Gpa, grit, MaterialHardship

def importData(category, file):
	x = category + "Background" + file + ".csv"
	trainX = np.genfromtxt(x, delimiter=',')
	y = category + "Labels" + file + ".csv"
	trainY = np.genfromtxt(y, delimiter=',')
	a = category + "Test" + file + ".csv"
	testX = np.genfromtxt(a, delimiter=',')

	if file == "Mother" or file == "Father":
		if category == "eviction" or category == "layoff" or category == "jobTraining":
			transformer = SelectKBest(chi2, k=2)
		else:
			transformer = SelectKBest(f_regression, k=2)
		trainX =  transformer.fit_transform(trainX, trainY)
		testX = transformer.transform(testX)

	return trainX, trainY, testX

def tryClassifiers(trainX, trainY, testX, category, file):
	X_train, X_test, y_train, y_test = train_test_split(trainX , trainY, test_size=0.4, random_state=0)

	if category == "eviction" or category == "layoff" or category == "jobTraining":
		clf = GaussianNB()
		clf.fit(X_train, y_train)
		y_predict = clf.predict_proba(X_test)[:, 1]
		clf_score = brier_score_loss(y_test, y_predict)
		print(category + " GB " + file),
		print("%1.5f" % clf_score)

		clf = svm.SVC(probability=True)
		clf.fit(X_train, y_train)
		y_predict = clf.predict_proba(X_test)[:, 1]
		clf_score = brier_score_loss(y_test, y_predict)
		print(category + " SVM " + file),
		print("%1.5f" % clf_score)

		clf = LogisticRegression(solver='saga')
		clf.fit(X_train, y_train)
		y_predict = clf.predict_proba(X_test)[:, 1]
		clf_score = brier_score_loss(y_test, y_predict)
		print(category + " LR " + file),
		print("%1.5f" % clf_score)

	else:
		clf = Ridge()
		clf.fit(X_train, y_train)
		y_predict = clf.predict(X_test)
		clf_score = mean_squared_error(y_test, y_predict)
		print(category + " Ridge " + file),
		print("%1.5f" % clf_score)


def continuousPredictions(category, file):
	trainX, trainY, testX = importData(category, file)
	clf = Ridge()
	clf.fit(trainX, trainY)
	predictions = clf.predict(testX)
	writeOutput(predictions, category)


def binaryPredictions(category, file):
	trainX, trainY, testX = importData(category, file)
	clf = svm.SVC(probability=True)
	clf.fit(trainX, trainY)
	predictions = clf.predict_proba(testX)[:,1]
	writeOutput(predictions, category)



def writeOutput(predictions, category):
	results = pd.read_csv("prediction.csv", header = 0, low_memory=False)
	results[category] = np.array(predictions)
	results.to_csv('prediction.csv', index=False)



def main():
	# loop = ["gpa", "grit", "materialHardship", "eviction", "layoff", "jobTraining"]
	# for category in loop:
	# 	files = ["Mother", "Father", "Six"]
	# 	for f in files:
	# 		trainX, trainY, testX = importData(category, f)
	# 		tryClassifiers(trainX, trainY, testX, category, f)
			
    continuousPredictions("gpa", "Father")
    continuousPredictions("grit", "Father")
    continuousPredictions("materialHardship", "Father")
    binaryPredictions("eviction", "Father")
    binaryPredictions("layoff", "Father")
    binaryPredictions("jobTraining", "Father")
    

if __name__ == "__main__":
  main()