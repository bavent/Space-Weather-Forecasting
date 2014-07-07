# Build, train, and test a Elman Recurrent neural network using pybrain

from DataManager import DataManager

import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats.stats import pearsonr
from math import sqrt
import matplotlib.pyplot as plt

from pybrain.structure import RecurrentNetwork
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, TanhLayer, BiasUnit
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer, RPropMinusTrainer
from pybrain.supervised.trainers.evolino import EvolinoTrainer
from pybrain.structure import FullConnection, IdentityConnection

# Prep the data
chunkLen = 200
predGap = 8
histLen = 0
includeDst = True
dataManager = DataManager()
print("Network: Elman")
print("predGap = {} hours".format(predGap))
print("histLen = {} hours".format(histLen))
print("includeDst = {}".format(includeDst))

trainDataset, testDataset = \
	dataManager.chunkifyDataWindowSequentialDS(chunkLen, predGap, histLen, includeDst)

# Network parameters
# Needs to be changed if the input parameters are adjusted
recSize = 5 if includeDst else 4
nInputs = (histLen+1)*recSize
hiddenNodes = 20
# Build and train the network
ERNNnet = buildNetwork(nInputs, hiddenNodes, 1, hiddenclass=TanhLayer, bias=False, recurrent=True)

''' Set up the network explictly
ERNNnet = RecurrentNetwork()

inLayer = LinearLayer(nInputs)
hiddenLayer = TanhLayer(20)
outLayer = LinearLayer(1)
biasUnit = BiasUnit()

ERNNnet.addInputModule(inLayer)
ERNNnet.addModule(hiddenLayer)
ERNNnet.addOutputModule(outLayer)
ERNNnet.addModule(biasUnit)

in_to_hidden = FullConnection(inLayer, hiddenLayer)
hidden_to_out = FullConnection(hiddenLayer, outLayer)
#reccurrentConnJordan = FullConnection(outLayer, hiddenLayer)
reccurrentConnElman = FullConnection(hiddenLayer, hiddenLayer)
biasHiddenConn = FullConnection(biasUnit, hiddenLayer)
biasOutputConn = FullConnection(biasUnit, outLayer)

ERNNnet.addConnection(in_to_hidden)
ERNNnet.addConnection(hidden_to_out)
ERNNnet.addRecurrentConnection(reccurrentConnElman)
#ERNNnet.addRecurrentConnection(reccurrentConnJordan)
ERNNnet.addConnection(biasHiddenConn)
ERNNnet.addConnection(biasOutputConn)
ERNNnet.sortModules()
'''

print(ERNNnet)
#print(inLayer)
#print(hiddenLayer)
#print(outLayer)
#print(in_to_hidden.params)
#print(hidden_to_out.params)
#print(reccurrentConn.params)

#trainer = BackpropTrainer(ERNNnet, trainDataset, momentum=0.5)
learningrate = 0.001
momentum = 0.9
lrdecay = .9999
weightdecay= 0.1
print("learningrate = " + str(learningrate))
print("momentum = " + str(momentum))
print("lrdecay = " + str(lrdecay))
print("weightdecay = " + str(weightdecay))
trainer = BackpropTrainer(ERNNnet, trainDataset, batchlearning=False, momentum=momentum, learningrate=learningrate, lrdecay=lrdecay, weightdecay=weightdecay)
#print("Using RPropMinusTrainer")
#trainer =  RPropMinusTrainer(ERNNnet)
#print("Using evolino trainer")
#trainer = EvolinoTrainer(ERNNnet, trainDataset, verbosity=True)

print("Training...")
trainRes = trainer.trainUntilConvergence(trainDataset, verbose=True, maxEpochs=1000)
print(trainRes)
res = trainer.testOnData(dataset=testDataset, verbose=False)
print("Test results : " + str(res*dataManager.dstNorm))

# Calculate the correlation coefficient
testPred = []
testTarget = []
print("num test sequences: " + str(testDataset.getNumSequences()))
for i in range(testDataset.getNumSequences()):
	seq = testDataset.getSequence(i)
	ERNNnet.reset()
	for j in range(len(seq[0])):
		#print("Activating on :" + str(seq[0][j]))
		testPred.append(ERNNnet.activate(seq[0][j]))
		testTarget.append(seq[1][j])
		#print("Network reads: " + str(testPred[-1]))
		#print("Actual       : " + str(seq[1][j]))

res = trainer.testOnData(dataset=testDataset, verbose=False)
print("Test results : " + str(res))

testPred = np.array(testPred)
testTarget = np.array(testTarget)

RMSE =  sqrt(mean_squared_error(testTarget, testPred))
p = pearsonr(testTarget, testPred)[0]

print("RMSE = " + str(RMSE*dataManager.dstNorm))
print("RHO = " + str(p))

print("Network: Elman")
print("predGap = {} hours".format(predGap))
print("histLen = {} hours".format(histLen))
print("includeDst = {}".format(includeDst))
print("learningrate = " + str(learningrate))
print("momentum = " + str(momentum))
print("lrdecay = " + str(lrdecay))
print("weightdecay = " + str(weightdecay))

# Plot the results
startI = 1
endI = 500
#print("Testtarget " + str(testTarget[startI:endI]*dataManager.dstNorm))
#print("TestPred " + str(testPred[startI:endI]*dataManager.dstNorm))
plt.plot(testTarget[startI:endI]*dataManager.dstNorm, lw=3.0, alpha=0.75)	
plt.plot(testPred[startI:endI]*dataManager.dstNorm, lw=3.0, alpha=0.75)
plt.legend(['Actual','Predicted'])
plt.title("{}-hr DST Forecasting with {}-hr History Elman".format(predGap, histLen))
plt.xlabel("Time (hrs)")
plt.ylabel("DST Index (nT)")
plt.show()