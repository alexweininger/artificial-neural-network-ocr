import pandas as pd
import numpy as np
from neural_network import MultiLayerPerceptron

# function to load data from csv file into numpy arrays, and to normalize columns
def loadCsv(path):
    # convert csv to an 2d array, where the labels are in col 0
    data = np.array(pd.read_csv(path, header=None))
    dataNoLabels = data[:, 1:]  # seperating labels from the data
    labels = data[:, :1]
    dataNoLabels = np.true_divide(dataNoLabels, 15)
    return dataNoLabels, labels

# function to convert an array of labels into a binary array
def labelToBinaryArray(labels, nbits=26):
    binaryLabelArray = np.zeros((labels.shape[0], nbits))
    for i in range(labels.shape[0]):
        binaryLabelArray[i, labels[i] - 1] = 1
    return binaryLabelArray

def makeMiniBatches(data, labels, batchSize):
    total = data.shape[0]  # gives the number of rows
    batchesData = []
    batchesLabels = []
    idx = 0
    while idx + batchSize <= total:
        batchesData.append(data[idx:idx+batchSize, :])
        batchesLabels.append(labelToBinaryArray(labels[idx:idx+batchSize]))
        idx += batchSize

    return batchesData, batchesLabels


allTData, allTLabels = loadCsv('./data/train.csv')
allVData, allVLabels = loadCsv('./data/valid.csv')

size = 100

tData, tLabels = makeMiniBatches(allTData, allTLabels, size)
vData, vLabels = makeMiniBatches(allVData, allVLabels, size)

mlp = MultiLayerPerceptron(layerOptions=[16, 100, 100, 26], minibatchSize=size)
mlp.start(tData, tLabels, vData, vLabels, learningRate=0.5, evalTrainEachEpoch=True)

mlp = MultiLayerPerceptron(layerOptions=[16, 100, 100, 26], minibatchSize=size)
mlp.start(tData, tLabels, vData, vLabels, learningRate=0.1, evalTrainEachEpoch=True)

mlp = MultiLayerPerceptron(layerOptions=[16, 100, 100, 26], minibatchSize=size)
mlp.start(tData, tLabels, vData, vLabels, learningRate=0.01, evalTrainEachEpoch=True)

mlp = MultiLayerPerceptron(layerOptions=[16, 100, 100, 26], minibatchSize=size)
mlp.start(tData, tLabels, vData, vLabels, learningRate=0.001, evalTrainEachEpoch=True)
