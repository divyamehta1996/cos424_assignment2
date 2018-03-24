import pandas as pd
import numpy as np
import math 
from sklearn import preprocessing

train = []
labels = []

predict = pd.read_csv("train.csv", header = 0, low_memory=False)
background = pd.read_csv("onlySix.csv", header = 0, low_memory=False)

ids = predict.shape[0]
for i in range(ids):

    score = unicode("eviction", "utf-8")
    key = unicode("challengeID", "utf-8")
    eviction = predict.loc[predict.index[[i]], score]
    # print eviction.values
    labels.append(eviction.values)
    
    # # Gets corresponding data for each challengeID in training set
    challengeID = predict.loc[predict.index[[i]], key].values
    data = background.loc[background.index[[i - 1]]]
    
    # # Removes from the array the id
    withoutID = data.drop(["challengeID"], axis=1)

    train.append(withoutID.values)

trainX = []
trainY = []

#remove NANs
for i in range(len(labels)):
    if not math.isnan(labels[i][0]):
        trainX.append(train[i][0])
        trainY.append(int(labels[i][0]))


#write train data to csv files
np.savetxt('eBackground1.csv', trainX,  fmt='%f', delimiter=',')
np.savetxt('eLabels1.csv', trainY, fmt='%i', delimiter=',')


rows = background.shape[0]
allSamples = []
for i in range(rows):  
    data = background.loc[background.index[[i]]]
    allSamples.append(data.drop(["challengeID"], axis=1).values[0])


np.savetxt('eTest1.csv', allSamples, fmt='%i', delimiter=',')

