import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


class NN():

    def __init__(self, numInput, numHidden, numOutput):
        self.w1 = np.matrix(np.random.rand(numInput, numHidden) * 2)
        self.w2 = np.matrix(np.random.rand(numHidden, numOutput) * 2)

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

    def backpropagate(self, data, expectedResult):
        self.forward(data)

        data = np.matrix(data)

        error = self.out2 - expectedResult

        sigPrimeNet2 = self.sigmoidPrime(self.netIn2.A1)
        backPropagatingError2 = np.multiply(error, sigPrimeNet2)
        partialDerivitivesW2 = np.dot(backPropagatingError2, self.out1)
        partialDerivitivesW2 = partialDerivitivesW2.T * 2

        sigPrimeNet1 = self.sigmoidPrime(self.netIn1.A1)
        backPropagatingError1 = np.dot(np.dot(sigPrimeNet1, self.w2), backPropagatingError2)
        partialDerivitivesW1 = np.dot(backPropagatingError1, data)
        partialDerivitivesW1 = partialDerivitivesW1.T * 2

        self.w2 = self.w2 - partialDerivitivesW2
        self.w1 = self.w1 - partialDerivitivesW1

        return 0.5 * (error)**2
        

myNN = NN(2, 3, 1)
print("W1:\n" + repr(myNN.w1))
print("W2:\n" + repr(myNN.w2))
print("Demo sigmoid of 4: " + repr(myNN.sigmoid(4)))

print("first try:")
print(myNN.forward([0.2, 0.3])) #should be 0.25
print(myNN.forward([0.8, 0.9])) #should be 0.85

for n in range(100):
    myNN.backpropagate([0.2, 0.3], [0.25])
    myNN.backpropagate([0.8, 0.9], [0.85])
    # print(myNN.forward([0.2, 0.3]))

print("another try:")
print(myNN.forward([0.2, 0.3])) #should be 0.25
print(myNN.forward([0.8, 0.9])) #should be 0.85
