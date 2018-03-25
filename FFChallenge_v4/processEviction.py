import pandas as pd
import numpy as np
import math 
from sklearn import preprocessing
import sys


v = sys.argv[1]
s = sys.argv[2]
file = "only" + s + ".csv"
print file

train = []
labels = []

predict = pd.read_csv("train.csv", header = 0, low_memory=False)
background = pd.read_csv(file, header = 0, low_memory=False)

ids = predict.shape[0]
for i in range(ids):

    score = unicode(v, "utf-8")
    key = unicode("challengeID", "utf-8")
    val = predict.loc[predict.index[[i]], score]
    # print eviction.values
    labels.append(val.values)
    
    # # Gets corresponding data for each challengeID in training set
    challengeID = predict.loc[predict.index[[i]], key].values
    data = background.loc[background.index[[challengeID - 1]]]
    
    # # Removes from the array the id
    withoutID = data.drop(["challengeID"], axis=1)

    train.append(withoutID.values)

# print labels
trainX = []
trainY = []

#remove NANs
for i in range(len(labels)):
    if not math.isnan(labels[i][0]):
        trainX.append(train[i][0])
        trainY.append(labels[i][0])

# print trainY

for each in trainX:
    if type(each) is str:
        print each

outB = v + "Background" + s + ".csv"
outL = v + "Labels" + s + ".csv"
outT = v + "Test" + s + ".csv"

#write train data to csv files
np.savetxt(outB, trainX,  fmt='%f', delimiter=',')
np.savetxt(outL, trainY, fmt='%f', delimiter=',')


rows = background.shape[0]
allSamples = []
for i in range(rows):  
    data = background.loc[background.index[[i]]]
    allSamples.append(data.drop(["challengeID"], axis=1).values[0])


np.savetxt(outT, allSamples, fmt='%i', delimiter=',')

