from scipy.optimize import fsolve
import numpy as np

def newton_raphson_method(f, J, x0, tol=1e-6, max_iter=100):
    """
    Implement Newton-Raphson method for solving non-linear equations.

    :param f: Function representing the system of equations.
    :param J: Jacobian matrix of the function f.
    :param x0: Initial guess for the roots.
    :param tol: Tolerance for convergence.
    :param max_iter: Maximum number of iterations.
    :return: Solution vector.
    """
    x = x0
    for _ in range(max_iter):
        x_new = x - np.linalg.inv(J(x)) @ f(x)
        if np.linalg.norm(x_new - x) < tol:
            return x_new
        x = x_new
    return x  # Return the last iteration if no convergence within max_iter

# Defining the system of equations
def f(x):
    x1, x2 = x
    return np.array([
        x1 * (x1**2 + x2**2 + 1)**(1/3) - 9/2,
        x2 * (x1**2 + x2**2 + 1)**(1/4) + 5/2
    ])

# Defining the Jacobian matrix
def J(x):
    x1, x2 = x
    return np.array([
        [(x1**2 + x2**2 + 1)**(1/3) + x1 * (1/3) * (x1**2 + x2**2 + 1)**(-2/3) * 2 * x1, 
         x1 * (1/3) * (x1**2 + x2**2 + 1)**(-2/3) * 2 * x2],
        [x2 * (1/4) * (x1**2 + x2**2 + 1)**(-3/4) * 2 * x1,
         (x1**2 + x2**2 + 1)**(1/4) + x2 * (1/4) * (x1**2 + x2**2 + 1)**(-3/4) * 2 * x2]
    ])

# Initial guess
initial_guess = np.array([1, -1])

# Applying Newton-Raphson Method
solution_nr = newton_raphson_method(f, J, initial_guess)
print(solution_nr)

