
#Comparison of Linear, Quadratic, and Cubic Petrov Galerkin methods for varying Velocity
#
#Equation : a * f' - f'' = 0, Boundary Condition : x = 0, f = 0 ; x = 1, f = 1


import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from solver import solver


x0 = 0.0
xL = 1.0
nElem = 5
bc0 = 0.0
bcL = 1.0

eType = 0

s = solver(x0, xL, eType, nElem, bc0, bcL)


#Exact Solution
def exact(u):
    y = np.zeros([s.nElem*(s.eType+1)+1])
    for i in range(0, s.nElem*(s.eType+1)+1):
        y[i] = (1 - np.exp(u * s.x[i] / 1.0))/(1 - np.exp(u * 1.0 / 1.0))
    return y


fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(0, 1), ylim=(-1, 1))
ax.grid()


line1, = ax.plot([], [], '-', label='Exact', lw=2)
line2, = ax.plot([], [], '--x', label='Linear', lw=1)
line3, = ax.plot([], [], '--o', label='Quadratic', lw=1)
line4, = ax.plot([], [], '--d', label='Cubic', lw=1)


ax.legend()
velocity_template = 'U = %.1f'
text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


def init():
    
    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    line4.set_data([], [])
    text.set_text('')
    
    return line1, line2, line3, line4, text

def animate(u):
    
    s.setU(0, 0, u, 0)
    s.eType = 0
    s.generateUniformGrid()
    line2.set_data(s.x, s.solve())
    
    s.setU(0, 0, u, 0)
    s.eType = 1
    s.generateUniformGrid()
    line3.set_data(s.x, s.solve())

    s.setU(0, 0, u, 0)
    s.eType = 2
    s.generateUniformGrid()
    line4.set_data(s.x, s.solve())

    line1.set_data(s.x, exact(u))
    
    
    text.set_text(velocity_template % (u))
    
    return line1, line2, line3, line4, text


ani = animation.FuncAnimation(fig, animate, np.arange(0.2, 20, 0.4), interval=5, blit=True, init_func=init)
plt.show()
