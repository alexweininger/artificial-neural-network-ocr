# Alex Weininger 2019
import numpy as np

# sigmoid activation function for the middle layers
def calculateSigmoid(X, deriv=False):
    if not deriv:
        return 1 / (1 + np.exp(-X))
    else:
        return calculateSigmoid(X)*(1 - calculateSigmoid(X))

# softmax activation function for the output layuers
def calculateSoftmax(X):
    Z = np.sum(np.exp(X), axis=1)
    Z = Z.reshape(Z.shape[0], 1)
    return np.exp(X) / Z

class PerceptronLayer:
    def __init__(self, size, minibatchSize, isInput=False, isOutput=False, activationFunction=calculateSigmoid):
        self.isInput = isInput
        self.isOutput = isOutput

        # here are the matricies, defined using the names from the equations and diagrams from class

        # Z is the matrix that holds output values
        self.Z = np.zeros((minibatchSize, size[0]))
        # The activation function is an externally defined function (with a derivative) that is stored here
        self.activation = activationFunction

        self.W = None  # W is the outgoing weight matrix for this layer

        self.S = None  # S is the matrix that holds the inputs to this layer

        self.D = None  # D is the matrix that holds the deltas for this layer

        self.Fp = None  # Fp is the matrix that holds the derivatives of the activation function

        if not isInput:
            self.S = np.zeros((minibatchSize, size[0]))
            self.D = np.zeros((minibatchSize, size[0]))

        if not isOutput:
            self.W = np.random.normal(size=size, scale=1E-4)

        if not isInput and not isOutput:
            self.Fp = np.zeros((size[0], minibatchSize))

    def forwardPropogation(self):
        if self.isInput:
            return self.Z.dot(self.W)

        self.Z = self.activation(self.S)
        if self.isOutput:
            return self.Z
        else:
            # For hidden layers, we add the bias values here
            self.Z = np.append(self.Z, np.ones((self.Z.shape[0], 1)), axis=1)
            self.Fp = self.activation(self.S, deriv=True).T
            return self.Z.dot(self.W)


class MultiLayerPerceptron:
    def __init__(self, layerOptions, minibatchSize=100):
         # layer_config=[16, 100, 100, 26]
        self.layers = []
        self.numberOfLayers = len(layerOptions)
        self.minibatchSize = minibatchSize

        for i in range(self.numberOfLayers-1):
            if i == 0:
                print(f'Initializing input layer with size {0}.', str(
                    layerOptions[i]))
                # Here, we add an additional unit at the input for the bias
                # weight.
                self.layers.append(
                    PerceptronLayer([layerOptions[i]+1, layerOptions[i+1]], minibatchSize, isInput=True))
            else:
                print(
                    f"Initializing hidden layer with size {0}.", layerOptions[i])
                # Here we add an additional unit in the hidden layers for the bias weight.
                self.layers.append(PerceptronLayer(
                    [layerOptions[i]+1, layerOptions[i+1]], minibatchSize, activationFunction=calculateSigmoid))

        print(f'Initializing output layer with size {layerOptions[-1]}.')
        self.layers.append(PerceptronLayer(
            [layerOptions[-1], None], minibatchSize, isOutput=True, activationFunction=calculateSoftmax))
        print('Finished generating output layer.')

    def forwardPropagation(self, data):
        # We need to be sure to add bias values to the input
        self.layers[0].Z = np.append(data, np.ones((data.shape[0], 1)), axis=1)

        for i in range(self.numberOfLayers-1):
            self.layers[i+1].S = self.layers[i].forwardPropogation()
        return self.layers[-1].forwardPropogation()

    def backPropagation(self, yhat, labels):
        self.layers[-1].D = (yhat - labels).T
        for i in range(self.numberOfLayers-2, 0, -1):
            # We do not calculate deltas for the bias values
            W_noBias = self.layers[i].W[0:-1, :]

            self.layers[i].D = W_noBias.dot(self.layers[i+1].D) * \
                self.layers[i].Fp

    # given the learning rate, update the weights of the current layer
    def updateWeights(self, learningRate):
        for i in range(0, self.numberOfLayers-1):
            W_gradient = -learningRate * \
                (self.layers[i+1].D.dot(self.layers[i].Z)).T

            # add the weights calculated via the gradient to the weights
            self.layers[i].W += W_gradient

    # begin the learning algorithm
    def start(self, trainingData, trainingLabels, testData, testLabels, numberOfEpochs=500, learningRate=0.05, evalTrainEachEpoch=False, evalTestEachEpoch=True):

        numberOfTrainings = len(trainingLabels)*len(trainingLabels[0])
        numberOfTests = len(testLabels)*len(testLabels[0])

        # confusion matrix to hold [input][output] of the model
        confusionMatrix = np.zeros(shape=(26, 26))

        # array to keep track of the classification rate vs. epochs
        classificationRateVsEpochs = []

        print(f'Starting training for {numberOfEpochs} epochs...')
        for t in range(0, numberOfEpochs):  # for each epoch
            outputString = f'[{t:4d}]'  # print epoch number

            # for the data and labels in each batch in this epoch
            for batchData, batchLabels in zip(trainingData, trainingLabels):
                # preform forward propagation
                output = self.forwardPropagation(batchData)
                # preform back propagation
                self.backPropagation(output, batchLabels)
                # update the weights, given the learning rate
                self.updateWeights(learningRate=learningRate)

            # if we want to evaluate the model against the training data each epoch
            if evalTrainEachEpoch:
                numberWrong = 0
                for batchData, batchLabels in zip(trainingData, trainingLabels):
                    # perform forward propagation
                    output = self.forwardPropagation(batchData)
                    # get the index of the largest value in array
                    yhat = np.argmax(output, axis=1)
                    numberWrong += np.sum(1 -
                                          batchLabels[np.arange(len(batchLabels)), yhat])

                outputString = f'{outputString} Training error: {(float(numberWrong)/numberOfTrainings):.5f}'
                classificationRateVsEpochs.append(
                    float(numberWrong)/numberOfTrainings)

            # if we want to evaluate the model against the test data each epoch
            if evalTestEachEpoch:
                numberWrong = 0
                for batchData, batchLabels in zip(testData, testLabels):
                    output = self.forwardPropagation(batchData)
                    # get the index of the largest value in array
                    yhat = np.argmax(output, axis=1)
                    numberWrong += np.sum(1 -
                                          batchLabels[np.arange(len(batchLabels)), yhat])

                    for binaryLabelArray, y in zip(batchLabels, yhat):

                        # get the index of the largest value in array (converting binary label back to a decimal value that maps to a character)
                        label = np.argmax(binaryLabelArray, axis=None)
                        confusionMatrix[label][y] += 1

                outputString = f'{outputString} Test error: {(float(numberWrong)/numberOfTests):.5f}'

            print(outputString)

        # save the confusion matrix values to a csv
        np.savetxt(
            f"./results/confusionMatrix{numberOfEpochs}e{learningRate}r.csv", confusionMatrix, delimiter=",")

        # save the classification array values to a csv
        np.savetxt(f"./results/classificationArray{numberOfEpochs}e{learningRate}r.csv",
                   classificationRateVsEpochs, delimiter=",")
