
#Testing variable Velocity and Source against exact solution
#
#Equation : (a / x) * f' - f'' - x^2 = 0, Boundary Condition : x = 1, f = 1 ; x = 2, f = 0


import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from solver import solver


x0 = 1.0
xL = 2.0
nElem = 10
bc0 = 1.0
bcL = 0.0
eType = 0

s = solver(x0, xL, eType, nElem, bc0, bcL)

s.setQ(-1, 0, 0, 0)


#Exact Solution
def exact(a):
    y = np.zeros([(s.eType+1)*s.nElem+1])
    c1 = - (1 + 15 / 4 / (a - 3)) * (a + 1) / (2**(a+1) - 1)
    c2 = 1 - 1 / 4 / (a - 3) - c1 / (a + 1)
    for i in range(0, (s.eType+1)*s.nElem+1):
        y[i] = c1 * s.x[i]**(a + 1) / (a + 1) + s.x[i]**4 / 4 / (a - 3) + c2
    return y


fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(1, 2), ylim=(0, 1.5))
ax.grid()


line1, = ax.plot([], [], '-', label='Exact', lw=2)
line2, = ax.plot([], [], '--x', label='Petrov Galerkin Linear', lw=1)


ax.legend()
velocity_template = 'U = %.1f / x'
text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


def init():
    
    line1.set_data([], [])
    line2.set_data([], [])
    text.set_text('')
    
    return line1, line2, text

def animate(a):

    s.setU(0, 0, 0, a)
    y = s.solve()
    
    line1.set_data(s.x, exact(a))
    line2.set_data(s.x, y)
    
    text.set_text(velocity_template % (a))
    
    return line1, line2, text


ani = animation.FuncAnimation(fig, animate, np.arange(2, 30, 2), interval=10, blit=True, init_func=init)
plt.show()

