# Perform Support Vector Regression on training data with radial basis function
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats.stats import pearsonr
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.svm import SVR

from DataManager import DataManager

# Prep the data
chunkLen = 200
predGap = 1
histLen = 3
includeDst = True
dataManager = DataManager()
print("predGap = {} hours".format(predGap))
print("histLen = {} hours".format(histLen))
print("includeDst = {}".format(includeDst))
trainInput, trainTarget, validInput, validTarget, testInput, testTarget = \
	dataManager.chunkifyDataWindow(chunkLen, predGap, histLen, includeDst)

# Find the goodness for simple DST shifting as a compaison
splitDate = (2004,1,21,0)
ctrainInput, ctrainTarget, ctestInput, ctestTarget = \
	dataManager.genDataForModel(predGap, histLen, splitDate)
pRef = pearsonr(ctestTarget[predGap:], ctestTarget[0:-predGap])
RMSERef =  np.sqrt(mean_squared_error(ctestTarget[predGap:], ctestTarget[0:-predGap]))
print("Reference RMSE: {}".format(RMSERef*dataManager.dstNorm))
print("Reference p: {}".format(pRef[0]))

C = 1e3
gamma = 0.05
clf = SVR(kernel='rbf', C=C, gamma=gamma)
clf.fit(trainInput, trainTarget.ravel())

testPred = clf.predict(testInput)
testPred = np.array(testPred)
testPred = testPred.reshape(testTarget.shape)

RMSE =  sqrt(mean_squared_error(testTarget, testPred))
p = pearsonr(testTarget, testPred)[0]
print("RMSE = " + str(RMSE*dataManager.dstNorm))
print("RHO = " + str(p))