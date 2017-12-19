# finite-element-fluid

Finite Element Method approximations to Fluid Dynamics


Convection - Diffusion > 1D_Steady_Solver.py :
Solver for 1D Steady Convection-Diffusion problems using Standard / Petrov Galerkin methods and 3-point Linear / Quadratic elements
Velocity, Diffusion Coefficient, and Source can be set as functions of the form : a*x^2 + b*x + c + d/x
Main inputs are :
 1. Starting x coordinate (x0)
 2. Length of domain
 3. Number of elements
 4. Bounday conditions at x = x0, L
 5. Element type (0 - Linear, 1 - Quadratic)
 6. Method (0 - Standard Galerkin, 1 - Optimal Petrov Galerkin)


Convection - Diffusion > 1D_Steady_Test_1.py :
Comparison of Standard and Optimal Petrov Galerkin methods for varying Peclet number (Pe = U * h / k / 2)
Equation : a * f' - f'' = 0
Boundary Conditions : x = 0, f = 0 ; x = 1, f = 1


Convection - Diffusion > 1D_Steady_Test_2.py :
Comparison of Linear and Quadratic elements for Optimal Petrov Galerkin method for varying Velocity
Equation : (a / x) * f' - f'' - x^2 = 0
Boundary Conditions : x = 1, f = 1 ; x = 2, f = 0