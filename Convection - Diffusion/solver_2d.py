import numpy as np
from numpy import linalg
from numpy.linalg import inv
import matplotlib.pyplot as plt
import math
import json

with open('config.json', 'r') as f:
    config = json.load(f)

x0 = config['X_START']
xL = config['X_END']
y0 = config['Y_START']
yL = config['Y_END']
nEleX = config['X_ELEMENTS']
nEleY = config['Y_ELEMENTS']
bcWall = []
bcWall.append(config['BC']['BOTTOM'])
bcWall.append(config['BC']['RIGHT'])
bcWall.append(config['BC']['TOP'])
bcWall.append(config['BC']['LEFT'])
xVel = config['X_VELOCITY']
yVel = config['Y_VELOCITY']
diff = config['DIFFUSION_COEFFICIENT']
source = config['SOURCE']


def setPoints():
    
    nPointX = nEleX + 1
    nPointY = nEleY + 1
    hX = (xL - x0) / nEleX
    hY = (yL - y0) / nEleY

    neq = 0                      # Equation number of point corresponds to
    
    points = np.zeros([nPointX, nPointY, 4])     # Point info : X, Y, BC(value / -1), Equation number
    for j in range(0, nPointY):
        for i in range(0, nPointX):
            
            x = x0 + hX*i
            y = y0 + hY*j
            points[i][j][0] = x
            points[i][j][1] = y

            points[i][j][3] = -1
            
            if bcWall[0] != -1 and y == y0:
                points[i][j][2] = bcWall[0]
            elif bcWall[1] != -1 and x == xL:
                points[i][j][2] = bcWall[1]
            elif bcWall[2] != -1 and y == yL:
                points[i][j][2] = bcWall[2]
            elif bcWall[3] != -1 and x == x0:
                points[i][j][2] = bcWall[3]
            else:
                points[i][j][2] = -1        
                points[i][j][3] = neq         # If there is no BC, then this is the position it corresponds to in matrices
                neq += 1

    return neq, points


def bilinearQuad(coords):       #Returns matrix of shape function and derivatives at all Gauss points
    
    nShape = 4

    N = []

    for i in range(0, len(coords)):
        x = coords[i][0]
        y = coords[i][1]
        N.append([[(-x + 1)*(-y + 1)/4, y/4 - 1/4, x/4 - 1/4, 0, 1/4, 0],
             [(x + 1)*(-y + 1)/4, -y/4 + 1/4, -x/4 - 1/4, 0, -1/4, 0],
             [(x + 1)*(y + 1)/4, y/4 + 1/4, x/4 + 1/4, 0, 1/4, 0],
             [(-x + 1)*(y + 1)/4, -y/4 - 1/4, -x/4 + 1/4, 0, -1/4, 0]])      # [ n, dn/dx, dn/dy, d2n/dx2, d2n/dxdy, d2n/dy2 ] Counter clockwise from node (-1, -1)
        
    return N

def gaussIntg(nGP):

    if nGP == 9:
        
        a = math.sqrt(3) / math.sqrt(5)
        coords = [[-a, -a], [0, -a], [a, -a], [-a, 0], [0, 0], [a, 0], [-a, a], [0, a], [a, a]]
        
        a = 5 / 9
        b = 8 / 9
        weights = [a**2, a*b, a**2, a*b, b*b, a*b, a**2, a*b, a**2]

    return coords, weights


def getU(x, y):
    return xVel

def getV(x, y):
    return yVel

def getk(x, y):
    return diff

def getQ(x, y):
    return source


def assemble():
    
    nNodes = 4       # Using bilinearQuad element

    nGP = 9          # Number of Gauss points for integration

    neq, points = setPoints()       # Getting number of equations and nodal details

    K = np.zeros([neq, neq])
    f = np.zeros([neq])

    coords, weights = gaussIntg(nGP)    #Getting coordinates and weights for Gauss Integration

    N = bilinearQuad(coords)      # Getting matrix of shape function values and their derivatives at Gauss point
    
    for j in range(0, nEleY):
        for i in range(0, nEleX):
            
            lPoints = []                    # Relating the local points in counter clockwise order to global ones
            lPoints.append(points[i][j])
            lPoints.append(points[i+1][j])
            lPoints.append(points[i+1][j+1])
            lPoints.append(points[i][j+1])
            
            for k in range(0, nGP):

                x_glob = y_glob = 0.0
                J = np.zeros([2, 2])
                
                for l in range(0, nNodes):      # Summing over element nodal shape functions to calculate global coordinates and J at Gauss point
                    
                    x_glob += N[k][l][0] * lPoints[l][0]
                    y_glob += N[k][l][0] * lPoints[l][1]
                    
                    J[0][0] += N[k][l][1] * lPoints[l][0]
                    J[0][1] += N[k][l][1] * lPoints[l][1]
                    J[1][0] += N[k][l][2] * lPoints[l][0]
                    J[1][1] += N[k][l][2] * lPoints[l][1]

                U_GP = getU(x_glob, y_glob)        # Find U, k, Q at Gauss point
                V_GP = getV(x_glob, y_glob)
                k_GP = getk(x_glob, y_glob)
                Q_GP = getQ(x_glob, y_glob)

                U_vec = np.array([U_GP, V_GP])

                if U_GP >= V_GP:
                    he = (xL-x0)/nEleX/math.cos(math.atan(V_GP/U_GP))
                else:
                    he = (yL-y0)/nEleY/math.cos(math.atan(U_GP/V_GP))

                U_abs = math.sqrt(U_GP**2+V_GP**2)
                Pe = U_abs*he/2/k_GP
                cothPe = (math.exp(abs(Pe)) + math.exp(-abs(Pe))) / (math.exp(abs(Pe)) - math.exp(-abs(Pe)))
                alpha = cothPe - 1 / abs(Pe)

                w = weights[k]

                Jinv = inv(J)

                for m in range(0, nNodes):      # Double loop over nodes to calculate element matrices - m : weighting func, n : shape func

                    bc_m = lPoints[m][2]
                    
                    if bc_m != -1:              # If BC at node m exists ( bc != -1), then skip mth equation
                        continue

                    neq_m = int(lPoints[m][3])
                    
                    for n in range(0, nNodes):

                        k_mn = N[k][m][0]*(U_vec.dot(Jinv).dot([N[k][n][1], N[k][n][2]]))*(J[0][0]*J[1][1]+J[0][1]*J[1][0])*w   # Pure convection

                        k_mn += alpha*he/2/U_abs*(U_vec.dot(Jinv).dot([N[k][m][1], N[k][m][2]]))*(U_vec.dot(Jinv).dot([N[k][n][1], N[k][n][2]]))*(J[0][0]*J[1][1]+J[0][1]*J[1][0])*w

                        k_mn += k_GP*Jinv.dot([N[k][m][1], N[k][m][2]]).dot(Jinv.dot([N[k][n][1], N[k][n][2]]))*(J[0][0]*J[1][1]+J[0][1]*J[1][0])*w   # Pure Diffusion

                        k_mn += k_GP*alpha*he/2/U_abs*(U_vec.dot((Jinv.dot(Jinv)).transpose()).dot([[N[k][m][3], N[k][m][4]], [N[k][m][4], N[k][m][5]]])).dot(Jinv.dot([N[k][n][1], N[k][n][2]]))*(J[0][0]*J[1][1]+J[0][1]*J[1][0])*w
                        
                        bc_n = lPoints[n][2]
                                                                            
                        if bc_n != -1:              # If BC at node n exists ( bc != -1), then multiply with k_mn value and add to force vector
                            
                            f[neq_m] += k_mn * bc_n
                            
                        else:                       # Else proceed as usual, add k_mn value to stiffness matrix
                            
                            neq_n = int(lPoints[n][3])
                            K[neq_m, neq_n] += k_mn

                    f_m = Q_GP*N[k][m][0]*(J[0][0]*J[1][1]+J[0][1]*J[1][0])*w

                    f_m += Q_GP*alpha*he/2/U_abs*(U_vec.dot(Jinv).dot([N[k][m][1], N[k][m][2]]))*(J[0][0]*J[1][1]+J[0][1]*J[1][0])*w

                    f[neq_m] += f_m
                            
    return K, f, points
                                                                            
def solve():
    K, f, points = assemble()
    out = inv(K).dot(-f)
    return out, points

out, points = solve()

x = np.zeros([(nEleX+1), (nEleY+1)])
y = np.zeros([(nEleX+1), (nEleY+1)])
z = np.zeros([(nEleX+1), (nEleY+1)])

for j in range(0, nEleY+1):

    for i in range(0, nEleX+1):

        x[i][j] = points[i][j][0]
        y[i][j] = points[i][j][1]
        if points[i][j][2] == -1:
            z[i][j] = out[int(points[i][j][3])]
        else:
            z[i][j] = points[i][j][2]

plt.figure(1)
plt.contourf(x, y, z)
plt.colorbar()
plt.xlabel('X')
plt.ylabel('Y')
plt.text(0.1, 1.03, '%d x %d Grid' % (nEleX, nEleY))
plt.show()
