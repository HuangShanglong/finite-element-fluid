# finite-element-fluid

Finite Element Method approximations to Fluid Dynamics


Convection - Diffusion > solver.py :
Solver for 1D Steady Convection-Diffusion problems using Petrov Galerkin finite element method.
Capable of Linear, Quadratic, and Cubic element types.
Uses Legendre-Gauss quadrature integral approx.
Velocity, Diffusion Coefficient, and Source can be set as functions of the form : a*x^2 + b*x + c + d/x.


Convection - Diffusion > 1D_Steady_Test_1.py :
Comparison of Linear, Quadratic, and Cubic Petrov Galerkin methods for varying Velocity.
Equation : a * f' - f'' = 0, 
Boundary Condition : x = 0, f = 0 ; x = 1, f = 1.


Convection - Diffusion > 1D_Steady_Test_2.py :
Testing variable Velocity and Source against exact solution.
Equation : (a / x) * f' - f'' - x^2 = 0, 
Boundary Condition : x = 1, f = 1 ; x = 2, f = 0


2D Solver :

Test :
Bottom and Right BC = 1.0
Top and Left BC = 0.0
X Velocity = 20.0
Y Velocity = 20.0
Diffusion Coefficient = 10.0
Source = 0.0
50 x 50 Grid