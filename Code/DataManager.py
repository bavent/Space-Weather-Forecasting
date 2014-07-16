# Parses and pre-processes selected data from various data sources
# Nicholas Sharp - nsharp3@vt.edu

import numpy as np 
import matplotlib.pyplot as plt
import random
import math

from pybrain.datasets import SequentialDataSet

# Precondition: Input reading assumes that both files contain correpsonding date ranges on identical hourly intervals
class DataManager:

    # Initialize the system by reading our entire datasets
    def __init__(self):

        self.dstNorm = 1.0
        self.numDateMap = {}
        self.dateNumMap = {}

        initDate = (2004, 1, 0, 0)

        dstFile = "../Data/DST_2004_2007_trimmed.dat"
        aceSwepamFile = "../Data/ACE_swepam_level2_data_2004_2007_trimmed.dat"
        aceMagFile = "../Data/ACE_mag_data_2004_2007_trimmed.dat"

        # Read in DST data
        # dstVals has the form [iHr, dst]
        ihr = 0
        dstVals = []
        with open(dstFile) as f:
            for line in f:

                year = 2000 + int(line[3:5])
                month = int(line[5:7])
                day = int(line[8:10])               

                for hr in range(0,24):

                    # Build out date maps
                    self.numDateMap[ihr] = (year,month,day,hr)
                    self.dateNumMap[(year,month,day,hr)] = ihr

                    dst = float(line[20+4*hr:20+4*(hr+1)])
                    dstVals.append((ihr, dst))
                    ihr += 1
                    
        self.dstVals = np.array(dstVals)

        # Read in ACE data
        # aceVals has the form [iHrs, protonDensity, Vx, Vy, Vz, speed, DynamicPressure, Bx, By, Bz]
        ihr = 0
        aceVals = []
        with open(aceSwepamFile) as f:
            for line in f:

                pDens = float(line[23:33])
                xDotGSM = float(line[66:76])
                yDotGSM = float(line[76:86]) 
                zDotGSM = float(line[86:96])

                aceVals.append((ihr, pDens, xDotGSM, yDotGSM, zDotGSM, -7, -7, -1, -1, -1))
                ihr += 1

        self.aceVals = np.array(aceVals)

        ihr = 0
        with open(aceMagFile) as f:
            for line in f:
                Bx = float(line[15:24])
                By = float(line[24:33])
                Bz = float(line[33:42])
                self.aceVals[ihr, 7] = Bx
                self.aceVals[ihr, 8] = By
                self.aceVals[ihr, 9] = Bz
                ihr += 1

        print("Data read from file.")

        self.interpolateData()

        # Calculate speed and dynamic pressure
        for i in range(self.aceVals.shape[0]):

            pDens = self.aceVals[i,1]
            xDotGSM = self.aceVals[i,2]
            yDotGSM = self.aceVals[i,3]
            zDotGSM = self.aceVals[i,4]

            speed = np.sqrt(xDotGSM*xDotGSM + yDotGSM*yDotGSM + zDotGSM*zDotGSM)
            pDyn = 1.6726e-6 * pDens * speed * speed    # See http://www.swpc.noaa.gov/SWN/sw_dials.html

            self.aceVals[i,5] = speed
            self.aceVals[i,6] = pDyn

    # Find missing data in the dataset and linearly interpolate to fill
    def  interpolateData(self):

        # Assumes that at the very least the first and last data points
        # in the set are not bad data points
    
        # DST data should all be clean. Only interpolate the ACE data,
        # verify the DST data here
        if(np.any(np.abs(self.dstVals[:,1]) > 500)):
            loc = np.where(np.abs(self.dstVals[:,1]) > 500)
            raise Exception("DST contains bad value at " + str(loc))

        # The bad values for each datapoint
        badVals = np.array([-1, -9999.9, -9999.9, -9999.9, -9999.9, -1, -1, -999.9, -999.9, -999.9])

        self.interpedVal = np.zeros(self.aceVals.shape[0], dtype=bool)

        # Check everything in the array and liearly interpolate between
        for iVal in range(len(badVals)):
            tot = 0

            # Only check the values which are read from the set (as opposed to calculated)
            if(badVals[iVal] == -1):
                continue

            length = 0
            for j in range(self.aceVals.shape[0]):
            
                if(self.aceVals[j, iVal] == badVals[iVal]):
                    # Propagate the last valid value
                    self.aceVals[j, iVal] = self.aceVals[j-1, iVal]

                    # Mark it
                    self.interpedVal[j] = True
                    length += 1
                    tot += 1

                elif(length > 0):
                    length = 0

        # Re-calculate the calculated values
        for j in  range(self.aceVals.shape[0]):
            
            pDens = self.aceVals[j, 1]
            xDotGSM = self.aceVals[j, 2]
            yDotGSM = self.aceVals[j, 3]
            zDotGSM = self.aceVals[j, 4]

            speed = np.sqrt(xDotGSM*xDotGSM + yDotGSM*yDotGSM + zDotGSM*zDotGSM)
            pDyn = 1.6726e-6 * pDens * speed * speed    # See http://www.swpc.noaa.gov/SWN/sw_dials.html

            self.aceVals[j, 5] = speed
            self.aceVals[j, 6] = pDyn

        print("Data cleaned of missing entries and interpolated")

        self.normalizeDST()

    # Generate the datasets for the machine learning models.
    # - histLen is the number of previous values for the history to include
    # - all data before splitDate are for training, all after are for testing
    def genDataForModel(self, predGap, histLen, splitDate):

        # Input sets are structured as (in time-increasing order)
        # [pDyn_0, Bz_0, DST_0, pDyn_1, Bz_1, DST_1, ... k]

        # Output sets are just vectors of DST

        iSplit = self.dateNumMap[splitDate]
        nData = self.aceVals.shape[0]

        trainInput = []
        trainTarget = []
        testInput = []
        testTarget = []

        for iData in range(nData):

            if(iData <= (histLen+predGap)):
                continue

            inputVec = np.zeros(3*histLen)

            # Look back to build the input vector
            for k in range(histLen):
                inputVec[3*k+0] = self.aceVals[iData - histLen - predGap + k + 1, 6]
                inputVec[3*k+1] = self.aceVals[iData - histLen - predGap + k + 1, 9]
                inputVec[3*k+2] = self.dstVals[iData - histLen - predGap + k + 1, 1]


            targetVal = self.dstVals[iData, 1]

            # Append the vector to the proper set
            if(iData < iSplit):
                trainInput.append(inputVec)
                trainTarget.append(targetVal)
            else:
                testInput.append(inputVec)
                testTarget.append(targetVal)


        trainInput = np.array(trainInput)
        trainTarget = np.array(trainTarget).reshape(len(trainTarget),1)
        testInput = np.array(testInput)
        testTarget = np.array(testTarget).reshape(len(testTarget),1)

        return trainInput, trainTarget, testInput, testTarget

    def normalizeDST(self):

        print("Normalizing DST")

        # Normalize DST values
        self.dstNorm = np.abs(self.dstVals[:, 1]).max()
        self.dstVals[:, 1] = self.dstVals[:, 1] / self.dstNorm

    # Find limits on each input parameter from the dataset
    def findDataLimits(self, histLen):

        histLen += 1

        lims = np.zeros((3,2))

        # Find the limits from the dataset
        lims[0, 0] = self.aceVals.min(axis=0)[6]
        lims[0, 1] = self.aceVals.max(axis=0)[6]

        lims[1, 0] = self.aceVals.min(axis=0)[9]
        lims[1, 1] = self.aceVals.max(axis=0)[9]

        lims[2, 0] = self.dstVals.min(axis=0)[1]
        lims[2, 1] = self.dstVals.max(axis=0)[1]

        # Expand each limit by a constant C 
        C = 0.10
        for i in range(lims.shape[0]):
            length = lims[i,1] - lims[i,0]
            lims[i,1] += length * C
            lims[i,0] -= length * C

        dataLims =  np.zeros((3*histLen,2))
        for i in range(histLen):
            dataLims[i*3+0, 0] = lims[0, 0]
            dataLims[i*3+0, 1] = lims[0, 1]
            dataLims[i*3+1, 0] = lims[1, 0]
            dataLims[i*3+1, 1] = lims[1, 1]
            dataLims[i*3+2, 0] = lims[2, 0]
            dataLims[i*3+2, 1] = lims[2, 1]
        
        return dataLims
        
    
    # Returns data as a shuffled list of chunks, without a window of past data
    def chunkifyDataNoWindow(self, chunkLen, predGap):

        nData = self.aceVals.shape[0]

        # List of chunks of the form [([pDyn, Bz, DST], DST_future) .... chunkLen]
        chunks = []
        currChunk = []

        # Build out the chunks
        for i in range(nData):

            if i + predGap >= nData:
                continue

            rec = ([self.aceVals[i,6],self.aceVals[i,9],self.dstVals[i,1]], self.dstVals[i+predGap,1])

            currChunk.append(rec)

            if currChunk.length == chunkLen:
                chunks.append(currChunk)
                currChunk = []


        random.shuffle(chunks)

    # Splits data in to chunks with a window, randomizes, then returns arrays
    def chunkifyDataWindow(self, chunkLen, predGap, windowLen, includeDST=True):

        nData = self.aceVals.shape[0]

        # List of chunks of the form [([pDyn, Bz, DST], DST_future) .... chunkLen]
        chunks = []
        currChunk = []

        # Build out the chunks
        for i in range(nData):

            if i + predGap >= nData or i - windowLen < 0:
                continue

            windowRec = np.ones(4*(windowLen+1))
            for j in range(windowLen+1):
                windowRec[4*j+0] = self.aceVals[i-j,6]
                windowRec[4*j+1] = self.aceVals[i-j,5]
                windowRec[4*j+2] = self.aceVals[i-j,9]
                windowRec[4*j+3] = self.dstVals[i-j,1]

            rec = (windowRec, self.dstVals[i+predGap,1])

            currChunk.append(rec)

            if len(currChunk) == chunkLen:
                chunks.append(currChunk)
                currChunk = []


        #random.shuffle(chunks)

        # Normalize
        
        norms = np.zeros(len(chunks[0][0][0]))
        for c in chunks:
            for rec in c:
                windowRec = rec[0]
                for k in range(len(windowRec)):
                    norms[k] = max(norms[k], abs(windowRec[k]))
        for c in chunks:
            for rec in c:
                windowRec = rec[0]
                for k in range(len(windowRec)):
                    windowRec[k] = windowRec[k] / norms[k]
        

        # Build out raw arrays of data from the chunks

        trainFrac = .6
        validFrac = .2
        testFrac  = .2

        trainInput = []
        trainOutput = []
        validInput = []
        validOutput = []
        testInput = []
        testOutput = []

        for iChunk in range(len(chunks)):

            # Add it to the training set
            if iChunk / (float(len(chunks))) <= trainFrac:
                for rec in chunks[iChunk]:
                    trainInput.append(rec[0])
                    trainOutput.append(rec[1])

            # Add it to the validation set
            elif iChunk / (float(len(chunks))) <= trainFrac+validFrac:
                for rec in chunks[iChunk]:
                    validInput.append(rec[0])
                    validOutput.append(rec[1])

            # Add it to the testing set
            else:
                for rec in chunks[iChunk]:
                    testInput.append(rec[0])
                    testOutput.append(rec[1])

        # Convert the lists to numpy arrays
        trainInput = np.array(trainInput)
        trainOutput = np.array(trainOutput).reshape(len(trainOutput),1)
        validInput = np.array(validInput)
        validOutput = np.array(validOutput).reshape(len(validOutput),1)
        testInput = np.array(testInput)
        testOutput = np.array(testOutput).reshape(len(testOutput),1)

        # Remove DST if it's not wanted
        if not includeDST:
            dstCol = 3
            while dstCol < trainInput.shape[1]:
                trainInput = np.delete(trainInput, dstCol, 1)
                validInput = np.delete(validInput, dstCol, 1)
                testInput = np.delete(testInput, dstCol, 1)
                dstCol += 3


        return trainInput, trainOutput, validInput, validOutput, testInput, testOutput


    # Splits data in to chunks with a window, randomizes, then returns arrays
    def chunkifyDataWindowSequentialDS(self, chunkLen, predGap, windowLen, includeDST=True):

        nData = self.aceVals.shape[0]

        # List of chunks of the form [([pDyn, Bz, DST], DST_future) .... chunkLen]
        allChunks = []
        currChunk = []

        # Build out the allChunks
        for i in range(nData):

            if i + predGap >= nData or i - windowLen < 0:
                continue

            windowRec = np.ones(5*(windowLen+1))
            for j in range(windowLen+1):
                windowRec[5*j+0] = np.sqrt(self.aceVals[i-j,6])
                windowRec[5*j+1] = self.aceVals[i-j,5]*self.aceVals[i-j,9]
                windowRec[5*j+2] = self.aceVals[i-j,5]
                windowRec[5*j+3] = self.aceVals[i-j,9]
                windowRec[5*j+4] = self.dstVals[i-j,1]

            rec = (windowRec, self.dstVals[i+predGap,1])

            currChunk.append(rec)

            if len(currChunk) == chunkLen:
                allChunks.append(currChunk)
                currChunk = []

        # Take only certain chunks
        chunks = []
        for chunk in allChunks:
            chunks.append(chunk)

        # Uncomment for shuffled records
        #random.shuffle(chunks)

        # Normalize
        
        norms = np.zeros(len(chunks[0][0][0]))
        for c in chunks:
            for rec in c:
                windowRec = rec[0]
                for k in range(len(windowRec)):
                    norms[k] = max(norms[k], abs(windowRec[k]))
        for c in chunks:
            for rec in c:
                windowRec = rec[0]
                for k in range(len(windowRec)):
                    windowRec[k] = windowRec[k] / norms[k]
        

        # Build out raw arrays of data from the chunks

        trainFrac = .8
        testFrac  = .2

        trainInput = []
        trainOutput = []
        testInput = []
        testOutput = []

        for iChunk in range(len(chunks)):

            # Add it to the training set
            if iChunk / (float(len(chunks))) <= trainFrac:
                for rec in chunks[iChunk]:
                    trainInput.append(rec[0])
                    trainOutput.append(rec[1])

            # Add it to the testing set
            else:
                for rec in chunks[iChunk]:
                    testInput.append(rec[0])
                    testOutput.append(rec[1])

        # Convert the lists to numpy arrays
        trainInput = np.array(trainInput)
        trainOutput = np.array(trainOutput).reshape(len(trainOutput),1)
        testInput = np.array(testInput)
        testOutput = np.array(testOutput).reshape(len(testOutput),1)

        # Remove DST if it's not wanted
        if not includeDST:
            dstCol = 4 # This index, and the increment below, must be kept 
            		   # consistent if the record size changes
            while dstCol < trainInput.shape[1]:
                trainInput = np.delete(trainInput, dstCol, 1)
                testInput = np.delete(testInput, dstCol, 1)
                dstCol += 4

        # Put the values in to dataset
        nInputs = trainInput.shape[1]

        trainDataset = SequentialDataSet(nInputs, 1)
        for i in range(trainInput.shape[0]):
            if i % chunkLen == 0 and i > 0:
                trainDataset.newSequence() 
            trainDataset.addSample(trainInput[i,:], trainOutput[i])

        testDataset = SequentialDataSet(nInputs, 1)
        for i in range(testInput.shape[0]):
            if i % chunkLen == 0 and i > 0:
                testDataset.newSequence() 
            testDataset.addSample(testInput[i,:], testOutput[i])

        return trainDataset, testDataset


    def calcLimsForArray(self, arr):
        nCols = arr.shape[1]
        lims = np.zeros((nCols,2))

        C = 0.1

        for i in range(nCols):
            lims[i,0] = arr[:,i].min()
            lims[i,1] = arr[:,i].max()

            # Expand the limits by a little
            length = lims[i,1] - lims[i,0]
            if(length > 0):
                lims[i,0] -= length*C
                lims[i,1] += length*C
            else:
                lims[i,0] = lims[i,0]*(1+C)
                lims[i,1] = -lims[i,0]*(1+C)

        return lims
