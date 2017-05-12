import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random


def f(x):
    return x + 0.98 * np.sin(x)


def ferr(x):
    return 0.01 * f(x)**2


def f1(x):
    return 1 + 0.98 * np.cos(x)


def ferr1(x):
    return 0.01 * 2 * f(x) * f1(x)


def minimize(f, f1, x0, nn, o=None):
    mvAvg = 0
    oldF = 0
    newF = 0
    rate = 0.01
    wantedFactor = 1000
    for n in range(nn):
        oldF = ferr(x0)

        x0 = x0 - rate * f1(x0)

        newF = ferr(x0)
        actualRate = np.abs(newF - oldF)
        wantedRate = np.abs(newF / 1)

        if actualRate > wantedRate:
            rate = rate / 2
        else:
            rate = rate + 0.01

        mvAvg = (mvAvg * n + actualRate) / (n + 1)
        if o != None:
            o.append(actualRate)

    return x0


x = np.arange(-20, 20, 0.02)
y = [ferr(x) for x in x]

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)

ax1.plot(x, y)
y = [ferr1(x) for x in x]
ax1.plot(x, y)
x0 = 40
ax1.plot(x0, ferr(x0), 'bo')
o = []
x0 = minimize(ferr, ferr1, x0, 5000, o)

x = np.arange(0, 5000, 1)

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.plot(x, o)

ax1.plot(x0, ferr(x0), 'bo')
plt.show()
