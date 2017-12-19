

#Solver for 1D Steady Convection-Diffusion problems using Standard / Petrov Galerkin methods and 3-point Linear / Quadratic elements
#
#Velocity, Diffusion Coefficient, and Source can be set as functions of the form : a*x^2 + b*x + c + d/x
#
#Main inputs are :
# 1. Starting x coordinate (x0)
# 2. Length of domain
# 3. Number of elements
# 4. Bounday conditions at x = x0, L
# 5. Element type (0 - Linear, 1 - Quadratic)
# 6. Method (0 - Standard Galerkin, 1 - Optimal Petrov Galerkin)


import numpy as np
from numpy import linalg
from numpy.linalg import inv
import scipy as sp
from scipy import integrate
import sympy
from sympy import *
import matplotlib.pyplot as plt


class Solver:


    #Initializing function
    def __init__(s, x0, L, nElem, bc0, bcL, elemType, method):
        
        s.x0 = x0                #Starting Point
        s.L = L                  #Length of Domain
        s.nElem = nElem          #Number of Elements
        s.bc0 = bc0              #Boundary Condition at x = x0
        s.bcL = bcL              #Boundary Condition at x = L
        s.elemType = elemType    #Element Type
        s.method = method        #Method
        
        s.nNode = 3 + (s.nElem - 1) * 2
        s.nInNode = s.nNode - 2
        s.h = s.L / s.nElem / 2
        
        s.x = np.zeros([s.nNode])
        s.K = np.zeros([s.nInNode, s.nInNode])
        s.f = np.zeros([s.nInNode])
        
        for i in range(0, s.nNode):
            s.x[i] = s.x0 + s.h * i
            
        s.setU(0, 0, 8, 0)       #Default settings for Velocity, Diffusion Coefficient, and Source
        s.setk(0, 0, 1, 0)
        s.setQ(0, 0, 0, 0)


    #Allowing U, k, Q to be set as functions of the form : a*x^2 + b*x + c + d/x
    def setU(s, a, b, c, d):
        x = symbols('x')
        xe = symbols('xe')      #Elemental center - Required for integration between -h and h
        s.U = a*(x+xe)**2 + b*(x+xe) + c + d/(x+xe)

    def setk(s, a, b, c, d):
        x = symbols('x')
        xe = symbols('xe')
        s.k = a*(x+xe)**2 + b*(x+xe) + c + d/(x+xe)

    def setQ(s, a, b, c, d):
        x = symbols('x')
        xe = symbols('xe')
        s.Q = a*(x+xe)**2 + b*(x+xe) + c + d/(x+xe)
                

    #Setting the elemental stiffness matrix and force vector
    def elem(s, e):
        
        kElem = np.zeros([3, 3])
        fElem = np.zeros([3])
        
        x = symbols('x')
        xe = symbols('xe')
        U = lambdify(xe, s.U)(s.x[2*e+1])
        k = lambdify(xe, s.k)(s.x[2*e+1])
        Q = lambdify(xe, s.Q)(s.x[2*e+1])
        
        if s.elemType == 0:     #3-Point Linear Element
            
            n1 = - x / s.h      #Shape functions for Standard Galerkin
            n21 = 1 + x / s.h
            n22 = 1 - x / s.h
            n3 = x / s.h
            
            if s.method == 1:       #Modifying shape functions for Optimal Petrov Galerkin
                Pe = lambdify(x, U)(0.0) * s.h / 2 / lambdify(x, k)(0.0)
                cothPe = (exp(abs(Pe)) + exp(-abs(Pe))) / (exp(abs(Pe)) - exp(-abs(Pe)))
                alpha = cothPe - 1 / abs(Pe)
                n1 += alpha * s.h * sign(lambdify(x, U)(0)) * diff(n1) / 2
                n21 += alpha * s.h * sign(lambdify(x, U)(0)) * diff(n21) / 2
                n22 += alpha * s.h * sign(lambdify(x, U)(0)) * diff(n22) / 2
                n3 += alpha * s.h * sign(lambdify(x, U)(0)) * diff(n3) / 2
                
            if s.method == 0 or s.method == 1:
                kElem[0, 0] = sp.integrate.quad(lambdify(x, (U * n1 * diff(n1) + k * diff(n1) * diff(n1))), -s.h, 0)[0]
                kElem[0, 1] = sp.integrate.quad(lambdify(x, (U * n1 * diff(n21) + k * diff(n1) * diff(n21))), -s.h, 0)[0]
                kElem[0, 2] = 0.0
                kElem[1, 0] = sp.integrate.quad(lambdify(x, (U * n21 * diff(n1) + k * diff(n21) * diff(n1))), -s.h, 0)[0]
                kElem[1, 1] = sp.integrate.quad(lambdify(x, (U * n21 * diff(n21) + k * diff(n21) * diff(n21))), -s.h, 0)[0] + sp.integrate.quad(lambdify(x, (U * n22 * diff(n22) + k * diff(n22) * diff(n22))), 0, s.h)[0]
                kElem[1, 2] = sp.integrate.quad(lambdify(x, (U * n22 * diff(n3) + k * diff(n22) * diff(n3))), 0, s.h)[0]
                kElem[2, 0] = 0.0
                kElem[2, 1] = sp.integrate.quad(lambdify(x, (U * n3 * diff(n22) + k * diff(n3) * diff(n22))), 0, s.h)[0]
                kElem[2, 2] = sp.integrate.quad(lambdify(x, (U * n3 * diff(n3) + k * diff(n3) * diff(n3))), 0, s.h)[0]
                fElem[0] = sp.integrate.quad(lambdify(x, (Q * n1)), -s.h, 0)[0]
                fElem[1] = sp.integrate.quad(lambdify(x, (Q * n21)), -s.h, 0)[0] + sp.integrate.quad(lambdify(x, (Q * n22)), 0, s.h)[0]
                fElem[2] = sp.integrate.quad(lambdify(x, (Q * n3)), 0, s.h)[0]
                
            if s.method == 2:               #Variational form using Standard Galerkin (Not functional yet)
                p = exp(- lambdify(x, U)(0.0) * x / lambdify(x, k)(0.0))
                kElem[0, 0] = sp.integrate.quad(lambdify(x, (diff(n1) * k * p * diff(n1))), -s.h, 0)[0]
                kElem[0, 1] = sp.integrate.quad(lambdify(x, (diff(n1) * k * p * diff(n21))), -s.h, 0)[0]
                kElem[0, 2] = 0.0
                kElem[1, 0] = sp.integrate.quad(lambdify(x, (diff(n21) * k * p * diff(n1))), -s.h, 0)[0]
                kElem[1, 1] = sp.integrate.quad(lambdify(x, (diff(n21) * k * p * diff(n21))), -s.h, 0)[0] + sp.integrate.quad(lambdify(x, (diff(n22) * k * p * diff(n22))), 0, s.h)[0]
                kElem[1, 2] = sp.integrate.quad(lambdify(x, (diff(n22) * k * p * diff(n3))), 0, s.h)[0]
                kElem[2, 0] = 0.0
                kElem[2, 1] = sp.integrate.quad(lambdify(x, (diff(n3) * k * p * diff(n22))), 0, s.h)[0]
                kElem[2, 2] = sp.integrate.quad(lambdify(x, (diff(n3) * k * p * diff(n3))), 0, s.h)[0]
                fElem[0] = sp.integrate.quad(lambdify(x, (n1 * p * Q)), -s.h, 0)[0]
                fElem[1] = sp.integrate.quad(lambdify(x, (n21 * p * Q)), -s.h, 0)[0] + sp.integrate.quad(lambdify(x, (n22 * p * Q)), 0, s.h)[0]
                fElem[2] = sp.integrate.quad(lambdify(x, (n3 * p * Q)), 0, s.h)[0]
                
        elif s.elemType == 1:       #3-Point Quadratic Element
            
            n1 = (x - s.h / 2)**2 / 2 / s.h**2 - 1 / 8     #Shape functions for Standard Galerkin
            n2 = - x**2 / s.h**2 + 1
            n3 = (x + s.h / 2)**2 / 2 / s.h**2 - 1 / 8
            
            if s.method == 1:                   #Modifying shape functions for Optimal Petrov Galerkin
                Pe = lambdify(x, U)(0.0) * s.h / 2 / lambdify(x, k)(0.0)
                cothPe = (exp(abs(Pe)) + exp(-abs(Pe))) / (exp(abs(Pe)) - exp(-abs(Pe)))
                alpha = cothPe - 1 / abs(Pe)
                n1 += alpha * s.h * sign(lambdify(x, U)(0)) * diff(n1) / 2
                n2 += alpha * s.h * sign(lambdify(x, U)(0)) * diff(n2) / 2
                n3 += alpha * s.h * sign(lambdify(x, U)(0)) * diff(n3) / 2
                
            if s.method == 0 or s.method == 1:
                kElem[0, 0] = sp.integrate.quad(lambdify(x, (U * n1 * diff(n1) + k * diff(n1) * diff(n1))), -s.h, s.h)[0]
                kElem[0, 1] = sp.integrate.quad(lambdify(x, (U * n1 * diff(n2) + k * diff(n1) * diff(n2))), -s.h, s.h)[0]
                kElem[0, 2] = sp.integrate.quad(lambdify(x, (U * n1 * diff(n3) + k * diff(n1) * diff(n3))), -s.h, s.h)[0]
                kElem[1, 0] = sp.integrate.quad(lambdify(x, (U * n2 * diff(n1) + k * diff(n2) * diff(n1))), -s.h, s.h)[0]
                kElem[1, 1] = sp.integrate.quad(lambdify(x, (U * n2 * diff(n2) + k * diff(n2) * diff(n2))), -s.h, s.h)[0]
                kElem[1, 2] = sp.integrate.quad(lambdify(x, (U * n2 * diff(n3) + k * diff(n2) * diff(n3))), -s.h, s.h)[0]
                kElem[2, 0] = sp.integrate.quad(lambdify(x, (U * n3 * diff(n1) + k * diff(n3) * diff(n1))), -s.h, s.h)[0]
                kElem[2, 1] = sp.integrate.quad(lambdify(x, (U * n3 * diff(n2) + k * diff(n3) * diff(n2))), -s.h, s.h)[0]
                kElem[2, 2] = sp.integrate.quad(lambdify(x, (U * n3 * diff(n3) + k * diff(n3) * diff(n3))), -s.h, s.h)[0]
                fElem[0] = sp.integrate.quad(lambdify(x, (Q * n1)), -s.h, s.h)[0]
                fElem[1] = sp.integrate.quad(lambdify(x, (Q * n2)), -s.h, s.h)[0]
                fElem[2] = sp.integrate.quad(lambdify(x, (Q * n3)), -s.h, s.h)[0]
                
            if s.method == 2:                   #Variational form using Standard Galerkin (Not functional yet)
                p = exp(- lambdify(x, U)(0.0) * x / lambdify(x, k)(0.0))
                kElem[0, 0] = sp.integrate.quad(lambdify(x, (diff(n1) * k * p * diff(n1))), -s.h, s.h)[0]
                kElem[0, 1] = sp.integrate.quad(lambdify(x, (diff(n1) * k * p * diff(n2))), -s.h, s.h)[0]
                kElem[0, 2] = sp.integrate.quad(lambdify(x, (diff(n1) * k * p * diff(n3))), -s.h, s.h)[0]
                kElem[1, 0] = sp.integrate.quad(lambdify(x, (diff(n2) * k * p * diff(n1))), -s.h, s.h)[0]
                kElem[1, 1] = sp.integrate.quad(lambdify(x, (diff(n2) * k * p * diff(n2))), -s.h, s.h)[0]
                kElem[1, 2] = sp.integrate.quad(lambdify(x, (diff(n2) * k * p * diff(n3))), -s.h, s.h)[0]
                kElem[2, 0] = sp.integrate.quad(lambdify(x, (diff(n3) * k * p * diff(n1))), -s.h, s.h)[0]
                kElem[2, 1] = sp.integrate.quad(lambdify(x, (diff(n3) * k * p * diff(n2))), -s.h, s.h)[0]
                kElem[2, 2] = sp.integrate.quad(lambdify(x, (diff(n3) * k * p * diff(n3))), -s.h, s.h)[0]
                fElem[0] = sp.integrate.quad(lambdify(x, (n1 * p * Q)), -s.h, s.h)[0]
                fElem[1] = sp.integrate.quad(lambdify(x, (n2 * p * Q)), -s.h, s.h)[0]
                fElem[2] = sp.integrate.quad(lambdify(x, (n3 * p * Q)), -s.h, s.h)[0]
                
        return (kElem, fElem)                


    #Assembling global stiffness matrix and force vector
    def assemble(s, e, kElem, fElem):
        
        if e == 0:
            for i in range(0, 2):
                for j in range(0, 2):
                    s.K[i, j] += kElem[i+1, j+1]
                s.f[i] += fElem[i+1] + s.bc0 * kElem[i+1, 0]
                
        elif e == (s.nElem - 1):
            for i in range(0, 2):
                for j in range(0, 2):
                    s.K[(2 * e - 1 + i), (2 * e - 1 + j)] += kElem[i, j]
                s.f[(2 * e - 1 + i)] += fElem[i] + s.bcL * kElem[i, 2]
                
        else:
            for i in range(0, 3):
                for j in range(0, 3):
                    s.K[(2 * e - 1 + i), (2 * e - 1 + j)] += kElem[i, j]
                s.f[(2 * e - 1 + i)] += fElem[i]


    #Calculating solution vector
    def solve(s):
        
        for e in range(0, s.nElem):
            elem = s.elem(e)
            s.assemble(e, elem[0], elem[1])
            
        y = np.zeros([s.nNode])
        y[0] = s.bc0
        y[s.nNode - 1] = s.bcL
        
        out = inv(s.K).dot(-s.f)
        
        for i in range(0, s.nInNode):
            y[i+1] = out[i]
            
        return y


    #Resetting stiffness matrix and force vector to zero
    def resetMatrices(s):
        
        s.K = np.zeros([s.nInNode, s.nInNode])
        s.f = np.zeros([s.nInNode])


    def exact(s, a, b, c, d):
        y = np.zeros([s.nNode])
        k1 = (1 + d*(a+2*c)/2/a/(a-b))*(a+b)/((b+c)**(a/b+1)-c**(a/b+1))
        k2 = -k1*c**(a/b+1)/(a+b)
        #c1 = -20303 / 12000000 / (exp(200) - 1)
        #c2 = - c1
        for i in range(0, s.nNode):
            y[i] = -d*s.x[i]*(a*s.x[i]+2*c)/2/a/(a-b) + k1*(b*s.x[i]+c)**(a/b+1)/(a+b) + k2
            #y[i] = c1 * exp(200 * s.x[i]) + c2 + s.x[i]**3 / 600 + s.x[i]**2 / 40000 + s.x[i] / 4000000
        return y
    
#lin = Solver(1.0, 1.0, 10, 1.0, 0.0, 1, 1)
#lin.setU(0, 0, 0, 60)
#lin.setk(0, 0, 1, 0)
#lin.setQ(-1, 0, 0, 0)
#lin2 = Solver(1.0, 1.0, 10, 1.0, 0.0, 1, 2)
#lin2.setU(0, 0, 0, 60)
#lin2.setk(0, 0, 1, 0)
#lin2.setQ(-1, 0, 0, 0)
#rLin = lin.solve()
#plt.plot(lin.x, lin.exact(4, 0, 1, -2))
#plt.plot(lin.x, lin.solve(), '-')
#plt.plot(lin2.x, lin2.solve(), ':x')
#pet = Solver(0.0, 1.0, 4, 0.0, 1.0, 1, 2)
#rPet = pet.solve()
#plt.plot(pet.x, rPet)
#plt.show()
