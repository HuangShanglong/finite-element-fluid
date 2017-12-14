import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg
from numpy.linalg import inv

#Settings
U = 60
x0 = 1.0
L = 1.0
steps = 20

h = L / steps
points = steps + 1
inpoints = points - 2

x = np.zeros([points])
y = np.zeros([points])

for i in range(0, points):
    x[i] = 1.0 + h * i


#Boundary conditions
y[0] = 1
y[points - 1] = 0


#Calculate optimal alpha for each internal node
alpha = np.zeros([inpoints])
Pe = np.zeros([inpoints])

for i in range(0, inpoints):
    Pe[i] = U * h / 2 / x[i+1]
    cothPe = (np.exp(abs(Pe[i])) + np.exp(-abs(Pe[i]))) / (np.exp(abs(Pe[i])) - np.exp(-abs(Pe[i])))
    alpha[i] = cothPe - 1 / abs(Pe[i])


#Calculate Stiffness matrix and Force vector
K = np.zeros([inpoints, inpoints])
f = np.zeros([inpoints])

for i in range(0, inpoints):

    t1 = - U * (alpha[i] + 1) / 2 - (x[i+1] + x[i]) / 2 / h
    t2 = U * alpha[i] + (x[i] + x[i+2]) / h
    t3 = - U * (alpha[i] - 1) / 2 - (x[i+2] + x[i+1]) / 2 / h

    for j in range(0, inpoints):
        if i == j+1:
            K[i][j] = t1
        elif i == j:
            K[i][j] = t2
        elif i == j-1:
            K[i][j] = t3
            break

    f[i] = (x[i+2]**5 - 2 * x[i+1]**5 + x[i]**5) / 5 / h
    f[i] = f[i] + ((h + x[i+1]) * (x[i+1]**4 - x[i+2]**4) + x[i] * (x[i+1]**4 - x[i]**4)) / 4 / h
    f[i] = f[i] + alpha[i] * h * (x[i+2]**4 - 2 * x[i+1]**4 + x[i]**4) / 8
    
    #Apply boundary conditions
    if i == 0:
        f[i] = f[i] + t1 * y[0]
    if i == (inpoints - 1):
        f[i] = f[i] + t3 * y[points - 1]
     

#Calculate result set
out = inv(K).dot(-f)
for i in range(0, inpoints):
    y[i+1] = out[i]


#Exact solution calculation
z = np.zeros([points])
for i in range(0, points):
    c1 = ((y[points - 1] - y[0]) - (x[points - 1]**4 - x[0]**4) / 228) / (x[points - 1]**61 - x[0]**61)
    c2 = y[0] - x[0]**4 / 228 - c1 * x[0]**61
    z[i] = c1 * x[i]**61 + c2 + x[i]**4 / 228


#Plotting
plt.plot(x, z, '-', lw=2, label='Exact')
plt.plot(x, y, ':o', lw=1, label='Optimal Petrov-Galerkin')
plt.legend()
plt.show()
