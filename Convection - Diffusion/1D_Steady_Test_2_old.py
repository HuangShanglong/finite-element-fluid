
#Comparison of Linear and Quadratic elements in Optimal Petrov Galerkin method for varying velocity
#
#Equation : (a / x) * f' - f'' - x^2 = 0, Boundary Condition : x = 1, f = 1 ; x = 2, f = 0

import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from solver_old import Solver 


x0 = 1.0
L = 1.0
nElem = 5
bc0 = 1.0
bc1 = 0.0

elemType = 0   #Linear
method = 1     #Petrov Galerkin

s1 = Solver(x0, L, nElem, bc0, bc1, elemType, method)
s1.setU(0, 0, 0, 60)
s1.setk(0, 0, 1, 0)
s1.setQ(-1, 0, 0, 0)

elemType = 1   #Quadratic
method = 1     #Petrov Galerkin

s2 = Solver(x0, L, nElem, bc0, bc1, elemType, method)
s2.setU(0, 0, 0, 60)
s2.setk(0, 0, 1, 0)
s2.setQ(-1, 0, 0, 0)


#Exact Solution
def exact(a):
    y = np.zeros([s1.nNode])
    c1 = - (1 + 15 / 4 / (a - 3)) * (a + 1) / (2**(a+1) - 1)
    c2 = 1 - 1 / 4 / (a - 3) - c1 / (a + 1)
    for i in range(0, s1.nNode):
        y[i] = c1 * s1.x[i]**(a + 1) / (a + 1) + s1.x[i]**4 / 4 / (a - 3) + c2
    return y


fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(1, 2), ylim=(0, 1.5))
ax.grid()


line1, = ax.plot([], [], '-', label='Exact', lw=2)
line2, = ax.plot([], [], '--x', label='Petrov Galerkin Linear', lw=1)
line3, = ax.plot([], [], ':o', label='Petrov Galerkin Quadratic', lw=1)


ax.legend()
velocity_template = 'U = %.1f / x'
text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


def init():
    
    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    text.set_text('')
    
    return line1, line2, line3, text

def animate(a):
    
    line1.set_data(s1.x, exact(a))
    s1.setU(0, 0, 0, a)
    line2.set_data(s1.x, s1.solve())
    s1.resetMatrices()
    s2.setU(0, 0, 0, a)
    line3.set_data(s2.x, s2.solve())
    s2.resetMatrices()
    
    text.set_text(velocity_template % (a))
    
    return line1, line2, line3, text


ani = animation.FuncAnimation(fig, animate, np.arange(4, 30, 1), interval=10, blit=True, init_func=init)
plt.show()
