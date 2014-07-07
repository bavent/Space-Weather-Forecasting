# Build, train, and test a basic feedfoward neural network using pybrain

import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats.stats import pearsonr
from math import sqrt
import matplotlib.pyplot as plt

from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

from DataManager import DataManager

# Prep the data
chunkLen = 200
predGap = 8
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

# Network parameters
recSize = 4 if includeDst else 3
nInputs = (histLen+1)*recSize
hiddenNodes = 14
epochs = 500
learningrate = 0.002
lrdecay = 0.9999995
print("hiddenNodes: {}".format(hiddenNodes))
print("learningrate: {}".format(learningrate))
print("lrdecay: {}".format(lrdecay))

# Build a pybrain dataset
trainDataset = SupervisedDataSet(nInputs, 1)
for i in range(trainInput.shape[0]):
    trainDataset.addSample(trainInput[i,:], trainTarget[i])

testDataset = SupervisedDataSet(nInputs, 1)
for i in range(testInput.shape[0]):
    testDataset.addSample(testInput[i,:], testTarget[i])

# Build and train the network
FFnet = buildNetwork(nInputs, hiddenNodes, 1, hiddenclass=TanhLayer, bias=True)
trainer = BackpropTrainer(FFnet, trainDataset, momentum=0.9, learningrate=learningrate, lrdecay=lrdecay)

print("Training...")
trainRes = trainer.trainUntilConvergence(verbose=False, maxEpochs=epochs)
res = trainer.testOnData(dataset=testDataset, verbose=False)
print("Test results : " + str(res*dataManager.dstNorm))

# Calculate the correlation coefficient
testPred = []
for i in range(testInput.shape[0]):
    testPred.append(FFnet.activate(testInput[i,:]))
testPred = np.array(testPred)
RMSE =  sqrt(mean_squared_error(testTarget, testPred))
p = pearsonr(testTarget, testPred)[0]

print("RMSE = " + str(RMSE*dataManager.dstNorm))
print("RHO = " + str(p))

print("predGap = {} hours".format(predGap))
print("histLen = {} hours".format(histLen))
print("includeDst = {}".format(includeDst))
print("hiddenNodes: {}".format(hiddenNodes))
print("learningrate: {}".format(learningrate))
print("lrdecay: {}".format(lrdecay))

# Plot the results
startI = 0
endI = 500
plt.plot(testTarget[startI:endI]*dataManager.dstNorm, lw=3.0, alpha=0.75)   
plt.plot(testPred[startI:endI]*dataManager.dstNorm, lw=3.0, alpha=0.75)
plt.legend(['Actual','Predicted'])
plt.title("{}-hr DST Forecasting with {}-hr History".format(predGap, histLen))
plt.xlabel("Time (hrs)")
plt.ylabel("DST Index (nT)")
plt.show()
