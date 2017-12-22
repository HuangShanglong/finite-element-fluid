
#Solver for 1D Steady Convection-Diffusion problems using Petrov Galerkin finite element method
#
#Capable of Linear, Quadratic, and Cubic element types
#
#Uses Legendre-Gauss quadrature integral approx.
#
#Velocity, Diffusion Coefficient, and Source can be set as functions of the form : a*x^2 + b*x + c + d/x


import numpy as np
from numpy import linalg
from numpy.linalg import inv
import scipy as sp
from scipy import integrate
import sympy
from sympy import *
import matplotlib.pyplot as plt


class solver:


    def __init__(s, x0, xL, eType, nElem, bc0, bcL):
        
        s.x0 = x0               #Starting point
        s.xL = xL               #Ending point
        s.eType = eType         #Type of element (0 - Linear, 1 - Quadratic, 2 - Cubic)
        s.nElem = nElem         #Number of elements
        s.nIntPoints = 3        #Number of points for Legendre-Gauss quadrature integral approx.

        s.bc0 = bc0             #Boundary condition at x = x0
        s.bcL = bcL             #Boundary condition at x = xL

        s.setU(0, 0, 10, 0)     #Default settings for U, k, Q
        s.setk(0, 0, 1, 0)
        s.setQ(0, 0, 0, 0)

        s.K = np.zeros([s.nElem*(s.eType+1)-1, s.nElem*(s.eType+1)-1])
        s.F = np.zeros([s.nElem*(s.eType+1)-1])

        s.generateUniformGrid()


    #Setting U, k, Q as function of form : a*x^2 + b*x + c + d/x
    def setU(s, a, b, c, d):
        s.U = lambda x: a*x**2 + b*x + c + d/x

    def setk(s, a, b, c, d):
        s.k = lambda x: a*x**2 + b*x + c + d/x

    def setQ(s, a, b, c, d):
        s.Q = lambda x: a*x**2 + b*x + c + d/x


    #Generating a uniform gird
    def generateUniformGrid(s):
        s.h = np.zeros([s.nElem])
        s.x = np.zeros([s.nElem*(s.eType+1)+1])
        for i in range(0, s.nElem):
            s.h[i] = (s.xL - s.x0) / s.nElem
            for j in range(0, (s.eType+1)):
                s.x[i*(s.eType+1)+j] = s.x0 + s.h[i] * i + s.h[i] * j / (s.eType+1)
        s.x[s.nElem*(s.eType+1)] = s.xL


    #Forming shape functions based on element type
    def getShapeFuncs(s):
        funcs = []
        x = symbols('x')
        d = 2 / (s.eType+1)
        for i in range (0, (s.eType+2)):
            func = 1
            for j in range(0, (s.eType+2)):
                if i != j:
                    func *= (x + 1 - j*d)
            func *= 1 / lambdify(x, func)(i*d - 1)
            funcs.append(func)
        return funcs

    #Forming shape function derivatives
    def getShapeFuncDers(s):
        funcs = s.getShapeFuncs()
        dersList = []
        for i in range(0, len(funcs)):
            ders = []
            ders.append(funcs[i])
            ders.append(diff(funcs[i]))
            ders.append(diff(diff(funcs[i])))
            dersList.append(ders)
        return dersList


    #Abscissae and weights based on Legendre-Gauss quadrature integration type
    def getIntPointsAbscissae(s):
        if s.nIntPoints == 3:
            return [-0.77459667, 0.0, 0.77459667]
        if s.nIntPoints == 5:
            return [-0.90617984, -0.53846931, 0.0, 0.53846931, 0.90617984]
        
    def getIntPointsWeights(s):
        if s.nIntPoints == 3:
            return [0.55555555, 0.88888888, 0.55555555]
        if s.nIntPoints == 5:
            return [0.23692688, 0.47862867, 0.56888888, 0.47862867, 0.23692688]


    #Precalculating shape function and derivative values at integration points
    def calcDersIntPoints(s):
        dersList = s.getShapeFuncDers()
        abscissae = s.getIntPointsAbscissae()
        s.shapeFuncDersIntPoints = np.zeros([s.eType+2, 3, np.size(abscissae)])
        x = symbols('x')
        for i in range(0, s.eType+2):
            for j in range(0, 3):
                for k in range(0, np.size(abscissae)):
                    s.shapeFuncDersIntPoints[i][j][k] = lambdify(x, dersList[i][j])(abscissae[k])


    #Calculating elemental stiffness matrix and force vector
    def calcElemMatrices(s, e):
        x0 = s.x[e*(s.eType + 1)]
        xL = s.x[e*(s.eType + 1)] + s.h[e]
        Jinv = 2 / (xL - x0)
        abscissae = s.getIntPointsAbscissae()
        weights = s.getIntPointsWeights()
        kElem = np.zeros([s.eType + 2, s.eType + 2])
        fElem = np.zeros([s.eType + 2])
        for k in range(0, s.nIntPoints):
            xc = x0*(1-abscissae[k])/2 + xL*(1+abscissae[k])/2
            Ue = s.U(xc)
            he = s.h[e]
            ke = s.k(xc)
            Qe = s.Q(xc)
            Pe = Ue * he / 2 / ke
            cothPe = (exp(abs(Pe)) + exp(-abs(Pe))) / (exp(abs(Pe)) - exp(-abs(Pe)))
            alpha = cothPe - 1 / abs(Pe)
            for i in range(0, (s.eType + 2)):
                for j in range(0, (s.eType + 2)):
                    kElem[i][j] += Ue * s.shapeFuncDersIntPoints[i][0][k] * s.shapeFuncDersIntPoints[j][1][k] * weights[k]   #Convective term - Standard Galerkin
                    kElem[i][j] += ke * s.shapeFuncDersIntPoints[i][1][k] * s.shapeFuncDersIntPoints[j][1][k] * Jinv * weights[k]    #Diffusive term - Standard Galerkin
                    kElem[i][j] += Ue * alpha * he / (s.eType + 1) / 2 * sign(Ue) * s.shapeFuncDersIntPoints[i][1][k] * s.shapeFuncDersIntPoints[j][1][k] * Jinv * weights[k]  #Convective term - Additional Petrov
                    kElem[i][j] += ke * alpha * he / (s.eType + 1) / 2 * sign(Ue) * s.shapeFuncDersIntPoints[i][2][k] * s.shapeFuncDersIntPoints[j][1][k] * Jinv**2 * weights[k] #Diffusive term - Additional Petrov
                fElem[i] += Qe * s.shapeFuncDersIntPoints[i][0][k] / Jinv * weights[k]
                fElem[i] += Qe * alpha * he / 2 * sign(Ue) * s.shapeFuncDersIntPoints[i][1][k] * weights[k]
        return (kElem, fElem)


    #Assembling elemental matrices into global matrices
    def assemble(s):
        for e in range(0, s.nElem):
            elem = s.calcElemMatrices(e)
            kElem = elem[0]
            fElem = elem[1]
            if e == 0:
                for i in range(0, s.eType + 1):
                    for j in range(0, s.eType + 1):
                        s.K[i, j] += kElem[i+1, j+1]
                    s.F[i] += fElem[i+1] + s.bc0 * kElem[i+1, 0]
                
            elif e == (s.nElem - 1):
                for i in range(0, s.eType + 1):
                    for j in range(0, s.eType + 1):
                        s.K[((s.eType + 1) * e - 1 + i), ((s.eType + 1) * e - 1 + j)] += kElem[i, j]
                    s.F[((s.eType + 1) * e - 1 + i)] += fElem[i] + s.bcL * kElem[i, (s.eType + 1)]
                
            else:
                for i in range(0, s.eType + 2):
                    for j in range(0, s.eType + 2):
                        s.K[((s.eType + 1) * e - 1 + i), ((s.eType + 1) * e - 1 + j)] += kElem[i, j]
                    s.F[((s.eType + 1) * e - 1 + i)] += fElem[i]


    #Main solve function
    def solve(s):
        s.reset()
        s.generateUniformGrid()
        s.calcDersIntPoints()
        s.assemble()
        y = np.zeros([s.nElem*(s.eType+1)+1])
        y[0] = s.bc0
        y[s.nElem*(s.eType+1)] = s.bcL
        out = inv(s.K).dot(-s.F)
        for i in range(0, s.nElem*(s.eType+1)-1):
            y[i+1] = out[i]
        return y


    #Resetting stiffness matrix and force vector
    def reset(s):
        s.K = np.zeros([s.nElem*(s.eType+1)-1, s.nElem*(s.eType+1)-1])
        s.F = np.zeros([s.nElem*(s.eType+1)-1])
