import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg
from numpy.linalg import inv
import matplotlib.animation as animation

#Pe = U * h / 2 / k

L = 1
points = 11
inpoints = points - 2
h = L/(points-1)

x = np.zeros([points])
for point in range(0, points):
    x[point] = h*point

#Exact Solution
def exact(Pe):
    y = np.zeros([points])
    for point in range(0, points):
        y[point] = (1 - np.exp(Pe * x[point] * 2 / h))/(1 - np.exp(Pe * L * 2 / h))
    return y    

#Matrix Formation and Solution
def Galerkin(t1, t2, t3):
    y = np.zeros([points])
    K = np.zeros([inpoints, inpoints])
    f = np.zeros([inpoints])

    for i in range(0, inpoints):
        for j in range(0, inpoints):
            if i == j+1:
                K[i][j] = t1
            elif i == j:
                K[i][j] = t2
            elif i == j-1:
                K[i][j] = t3
                break

    f[inpoints-1] = - t3
    phi = inv(K).dot(f)
    for i in range(0, points-2):
        y[i+1] = phi[i]
    y[0] = 0
    y[points-1] = 1
    return y

#Standard Galerkin
def stanGalerkin(Pe):
    t1 = - Pe - 1
    t2 = 2
    t3 = Pe - 1
    return Galerkin(t1, t2, t3)

#Full Upwind Galerkin
def fullUpwindGalerkin(Pe):
    t1 = - 2 * Pe - 1
    t2 = 2 + 2 * Pe
    t3 = -1
    return Galerkin(t1, t2, t3)

#Optimal Petrov-Galerkin Solution
def optPetrovGalerkin(Pe):
    cothPe = (np.exp(abs(Pe)) + np.exp(-abs(Pe))) / (np.exp(abs(Pe)) - np.exp(-abs(Pe)))
    alphaOpt = cothPe - 1 / abs(Pe)
    t1 = - Pe * (alphaOpt + 1) - 1
    t2 = 2 + 2 * alphaOpt * Pe
    t3 = - Pe * (alphaOpt - 1) - 1
    return Galerkin(t1, t2, t3)

#Execution / Animation
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(0, 1), ylim=(-1, 1))
ax.grid()
line1, = ax.plot([], [], '-', label='Exact', lw=2)
line2, = ax.plot([], [], '-.', label='Standard Galerkin', lw=2)
line3, = ax.plot([], [], ':', label='Full Upwind Galerkin', lw=2)
line4, = ax.plot([], [], 'o--', label='Optimal Petrov-Galerkin', lw=2)
ax.legend()
time_template = 'Pe = U*h/2/k = %.1f'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
def init():
    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    line4.set_data([], [])
    time_text.set_text('')
    return line1, line2, line3, line4, time_text
def animate(Pe):
    line1.set_data(x, exact(Pe))
    line2.set_data(x, stanGalerkin(Pe))
    line3.set_data(x, fullUpwindGalerkin(Pe))
    line4.set_data(x, optPetrovGalerkin(Pe))
    time_text.set_text(time_template % (Pe))
    return line1, line2, line3, line4, time_text
ani = animation.FuncAnimation(fig, animate, np.arange(0.01, 5.0, 0.01), interval=20, blit=True, init_func=init)
plt.show()
