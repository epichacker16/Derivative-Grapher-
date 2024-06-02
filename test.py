import sympy as sp
import numpy as np
import numpy as np
from scipy.optimize import fsolve
from matplotlib import pyplot as plt

def sp_der(func_str: str, input: np.ndarray):
    # Define the variable used in the function
    x = sp.symbols('x')

    # Parse the function string into a sympy expression
    func = sp.sympify(func_str)

    # Calculate the derivative
    derivative = sp.diff(func, x)

    # Convert the symbolic derivative into a function that can be evaluated with numpy arrays
    derivative_func = sp.lambdify(x, derivative, "numpy")

    # Evaluate the derivative at the points in the input array
    derivative_values = derivative_func(input)

    return derivative_values

def compare_values(sp_values, my_values):
    # assert equivalence
    assert np.allclose(sp_values, my_values), "Values are not equivalent"


