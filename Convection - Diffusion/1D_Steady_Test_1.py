
#Comparison of Standard and Optimal Petrov Galerkin methods for varying Peclet number (Pe = U * h / k / 2)
#
#Equation : a * f' - f'' = 0, Boundary Condition : x = 0, f = 0 ; x = 1, f = 1

import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from solver import Solver 


x0 = 0.0
L = 1.0
nElem = 4
bc0 = 0.0
bc1 = 1.0


elemType = 0   #Linear
method = 0     #Standard Galerkin

stanLin = Solver(x0, L, nElem, bc0, bc1, elemType, method)

elemType = 0   #Linear
method = 1     #Petrov Galerkin

petrLin = Solver(x0, L, nElem, bc0, bc1, elemType, method)


#Exact Solution
def exact(u):
    y = np.zeros([stanLin.nNode])
    for i in range(0, stanLin.nNode):
        y[i] = (1 - np.exp(u * stanLin.x[i] / 1.0))/(1 - np.exp(u * L / 1.0))
    return y


fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(0, 1), ylim=(-1, 1))
ax.grid()


line1, = ax.plot([], [], '-', label='Exact', lw=2)
line2, = ax.plot([], [], '--x', label='Standard Galerkin', lw=1)
line3, = ax.plot([], [], ':o', label='Petrov Galerkin', lw=1)


ax.legend()
peclet_template = 'Pe = U*h/2/k = %.1f'
text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


def init():
    
    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    text.set_text('')
    
    return line1, line2, line3, text

def animate(u):
    
    line1.set_data(stanLin.x, exact(u))
    stanLin.setU(0, 0, u, 0)
    line2.set_data(stanLin.x, stanLin.solve())
    stanLin.resetMatrices()
    petrLin.setU(0, 0, u, 0)
    line3.set_data(petrLin.x, petrLin.solve())
    petrLin.resetMatrices()
    
    text.set_text(peclet_template % (u*stanLin.h/2/stanLin.k))
    
    return line1, line2, line3, text


ani = animation.FuncAnimation(fig, animate, np.arange(0.8, 40, 0.8), interval=100, blit=True, init_func=init)
plt.show()
