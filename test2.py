import python_site as p
import numpy as np
import sympy as sp
from sympy.solvers import solve
# Define the string expression
expr_str = "1/x"
x = sp.symbols('x')
# Convert the string expression to a SymPy expression
expr = sp.sympify(expr_str)
# Convert the SymPy expression to a numerical function
func_numeric = sp.lambdify(x, expr, 'numpy')


        
 




if expr_str is not None:
    f_x, f_x_prime, f_2x_prime, fig, x_input = p.plot_graph(func_numeric, 5, -5, 51,expr_str)
    print("start find_sig\n")
    p.Find_sig(expr_str,x_input)
    print("x_input = ",x_input,"\n")
    print("end find_sig\n")
    p.st.pyplot(fig)