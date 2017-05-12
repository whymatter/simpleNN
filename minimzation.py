import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random


def f(x):
    """a non konvex function (like the feedforward function in neuronal nets)"""
    return x + 0.98 * np.sin(x)


def f1(x):
    """derivative of the feedforward function funtion"""
    return 1 + 0.98 * np.cos(x)


def ferr(x):
    """generate a konvex error function (mean squared error)"""
    return 0.01 * f(x)**2


def ferr1(x):
    """the derivative of the error function (using chain rule)"""
    return 0.01 * 2 * f(x) * f1(x)


def defaultMinimize(f, f1, x0, nn, o=None):
    rate = 0.01

    for n in range(nn):
        oldError = ferr(x0)
        x0 = x0 - ferr1(x0) * rate
        newError = ferr(x0)

        if o != None:
            o.append(newError - oldError)

    return x0


def myMinimize(f, f1, x0, nn, o=None):
    mvAvg = 0
    oldError = 0
    newError = 0
    lowestRate = 0.01
    rate = lowestRate
    wantedFactor = 100

    for n in range(nn):
        oldError = ferr(x0)
        x0 = x0 - ferr1(x0) * rate
        newError = ferr(x0)

        actualRate = newError - oldError
        mvAvg = (mvAvg * n + (actualRate)) / (n + 1)
        wantedRate = newError / wantedFactor

        if mvAvg > 0:
            # in case of increasing error
            rate = lowestRate
        elif np.abs(mvAvg) > wantedRate:
            # in case of increasing rate
            rate = rate - lowestRate
            # do never go below lowestRate
            if rate < lowestRate:
                rate = lowestRate
        else:
            # in case of decreasing rate
            rate = rate + lowestRate

        if o != None:
            o.append(mvAvg)

    return x0


startValue = 40
iterations = 1000

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)

# plot our feedforward dummy function
x = np.arange(-(startValue + 5), (startValue + 5), 0.02)
y = [ferr(x) for x in x]
ax1.plot(x, y, label="feedforward dummy")
# plot the derivative of the feedforward dummy function
y = [ferr1(x) for x in x]
ax1.plot(x, y, label="derivative")
# plot our x0 starting point
ax1.plot(startValue, ferr(startValue), 'bo')

# minimize ferr with derivative ferr1 starting x0 with 5000 iterations
o1 = []
x0 = startValue
x0 = myMinimize(ferr, ferr1, x0, iterations, o1)
# plot finish point of approximation
ax1.plot(x0, ferr(x0), 'go')

# minimize ferr with derivative ferr1 starting x0 with 5000 iterations
o2 = []
x0 = startValue
x0 = defaultMinimize(ferr, ferr1, x0, iterations, o2)
# plot finish point of approximation
red_dot = ax1.plot(x0, ferr(x0), 'ro')

# plot the rates in which the error decreases
x = np.arange(0, iterations, 1)
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.plot(x, o1, label="myMinimize rate")
ax2.plot(x, o2, label="defaultMinimize rate")

plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)

plt.show()
