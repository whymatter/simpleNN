import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random

class NN():
    """simple implementation of a NN with one hidden layer"""

    def __init__(self, numInput, numHidden, numOutput):
        self.w1 = np.matrix(np.random.randn(numInput, numHidden))
        self.w2 = np.matrix(np.random.randn(numHidden, numOutput))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoidPrime(self, x):
        sig = self.sigmoid(x)
        return sig * (1 - sig)

    def forward(self, data):
        """forward some data through out NN"""
        data = np.matrix(data)
        self.netIn1 = np.dot(data, self.w1)
        self.out1 = self.sigmoid(self.netIn1)

        self.netIn2 = np.dot(self.out1, self.w2)
        self.out2 = self.sigmoid(self.netIn2)

        return self.out2

    def backpropagate(self, data, expectedResult, rate):
        """train network using gradient decent"""
        self.forward(data)

        data = np.matrix(data)

        error = self.out2 - expectedResult

        # calculate partial derivative with regard to w2
        sigPrimeNet2 = self.sigmoidPrime(self.netIn2.A1)
        backPropagatingError2 = np.multiply(error, sigPrimeNet2)
        partialDerivitivesW2 = np.dot(backPropagatingError2, self.out1)
        partialDerivitivesW2 = partialDerivitivesW2.T * rate

        # calculate partial derivative with regard to w2
        sigPrimeNet1 = self.sigmoidPrime(self.netIn1.A1)
        backPropagatingError1 = np.dot(
            np.dot(sigPrimeNet1, self.w2), backPropagatingError2)
        partialDerivitivesW1 = np.dot(backPropagatingError1, data)
        partialDerivitivesW1 = partialDerivitivesW1.T * rate

        # modify weights
        self.w2 = self.w2 - partialDerivitivesW2
        self.w1 = self.w1 - partialDerivitivesW1

        # the mean squared error
        return 0.5 * (error)**2

# initialize some training data
# input values
xyTrain = np.random.randint(0, 11, (50, 2))
# input combined with output
xyzTrain = np.array(
    [np.array([np.array([x, y]), np.array([(x + y) / 2])]) for x, y in xyTrain]) / 10

def normalTrain():
    """training using fixed learning rate"""
    learnRate = 0.01

    # train network 40000 times
    for n in range(40000):
        error = 0

        for xyz in xyzTrain:
            error = error + myNN.backpropagate(xyz[0], xyz[1], learnRate)

        # logging
        if n % 100 == 0:
            print("iteration " + repr(n) + " error " + repr(error) + " rate " + repr(learnRate))

def myTraining():
    """using my training algorithm (see extra paper for info about it"""

    oldError = 0
    newError = 0
    lowestRate = 0.01
    rate = lowestRate
    wantedFactor = 10000
    mvAvgValuesCount = 50
    mvAvgValues = []

    # train network 40000 times
    for n in range(40000):
        oldError = newError
        newError = 0

        for xyz in xyzTrain:
            newError = newError + myNN.backpropagate(xyz[0], xyz[1], rate)

        mvAvgValues.append(newError - oldError)
        if len(mvAvgValues) > mvAvgValuesCount:
            del mvAvgValues[0]
        mvAvg = np.sum(mvAvgValues) / len(mvAvgValues)

        actualRate = mvAvg #newError - oldError
        wantedRate = newError / wantedFactor

        # print(actualRate)

        if actualRate > 0:
            rate = lowestRate
        elif np.abs(actualRate) > wantedRate:
            rate = rate - lowestRate
            if rate < lowestRate: rate = lowestRate
        else:
            rate = rate + lowestRate

        # logging    

        if n % 100 == 0:
            print("iteration " + repr(n) + " error " + repr(newError) + " rate " + repr(rate))

def fwd(x,y):
    """forward shortcut"""
    return myNN.forward([x,y])
    
def resetNetwork():
    """reset shortcut"""
    return NN(2, 3, 1)

# plotting stuff

def plotSuccess():
    """plot results of network vs calculated results"""
    x = y = np.arange(0, 1.0, 0.1)
    X, Y = np.meshgrid(x, y)

    # by network
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    zs = np.array([myNN.forward([x, y]) for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    ax.plot_surface(X, Y, Z)

    # wanted
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    zs = np.array([(x + y) / 2 for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z2 = zs.reshape(X.shape)
    ax2.plot_surface(X, Y, Z2)

    plt.show()

# init network
myNN = resetNetwork()