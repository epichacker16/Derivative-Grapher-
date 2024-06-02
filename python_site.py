import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy
from sympy.utilities.lambdify import lambdify
from sympy.parsing.sympy_parser import parse_expr
import pandas as pd
from scipy.misc import derivative
from scipy.optimize import fsolve
from test import sp_der, compare_values
import numpy as np
import sympy as sp
from sympy.solvers import solve

def find_critical_points(func, arrInput,min,max):
    # Initialize an empty list to store critical points
    critical_points = [ ]
    
    # Define the range of x values to search for critical points
    
    # Iterate over the range of x values
    for x in arrInput:
        # Use numerical optimization to find roots of the derivative function
        root = fsolve(func, x)
        
        # Check if the root is real and within the specified range
        if np.isreal(root) and np.abs(func(root)) < 1e-6 and min <= root <= max:
            critical_points.append(root)
    st.write(critical_points)
    appended_array = np.append(arrInput,critical_points)
    return appended_array


# Define a function to convert a string expression to a callable function
def parse_function(func_str):
    x = sympy.symbols("x")
    
    # Replace '^' with '**'
    func_str = func_str.replace('^', '**')

    # Check if the input is a single variable and it's "x"
    if func_str.strip() == "x":
        return lambdify(x, x, modules="numpy")  # Default function is identity function x -> x
    elif x not in parse_expr(func_str).free_symbols:
        st.error("Enter a function with 'x' as the variable.")
        raise ValueError("Variable must be 'x'")

    try:
        expr = parse_expr(func_str)
        return lambdify(x, expr, modules="numpy")
    except ValueError:
        st.error("Enter a valid function (e.g., x**2)")
        return None

# Define a function to get the function value at a specific point
def get_function_value(func, x):
    try:
        return func(x)
    except ZeroDivisionError:
        return np.nan

# Calculate the numerical derivative of a function at a given point x
def numerical_derivative(func, x, delta=1e-5):
    try:
        result = (get_function_value(func, x + delta) - get_function_value(func, x)) / delta
        return np.nan if np.isnan(result).all() else result
    except:
        return np.nan

# Create a copy of an array with NaN values at specific indices
def create_nan_array_copy(arr, arr_idx):
    nan_array = np.full(shape=arr.shape, fill_value=np.nan) # Create an array with the shape of arr filled with nans
    nan_array[arr_idx] = arr[arr_idx]
    return nan_array

# Helper function to get extrema and offset
def calculate_extrema_and_offset(function, percent):
    max_y, min_y = np.nanmax(function[np.isfinite(function)]), np.nanmin(function[np.isfinite(function)])
    offset = np.abs(percent * ((max_y - min_y) / 2))
    return max_y, min_y, offset

# Replace infinite values in an array with NaNs
def replace_infs_with_nans(array):
    inf_indices = np.where(array == np.inf)[0]
    array[inf_indices] = np.nan
    neg_inf_indices = np.where(array == -np.inf)[0]
    array[neg_inf_indices] = np.nan
    return array

# Replace singularities in a function's output with NaNs
def replace_singularity_with_nans(fx, arr_input):
    result = np.empty_like(arr_input, dtype=float)
    epsilon = 1e-1  # Tolerance for singularity
    for i, x_value in enumerate(arr_input):
        try:
            func_result = get_function_value(fx, x_value)
            
            # Calculate the numerical derivative at the current point
            derivative_at_x = numerical_derivative(fx, x_value)

            # Check if the result is NaN or infinite
            if np.isnan(func_result) or np.isinf(func_result):

                # Check if the singularity is within epsilon distance or the derivative is close to zero
                if np.abs(derivative_at_x) < epsilon:
                    # Insert NaN between consecutive elements whose sum is near zero
                    result[i] = func_result
                    result = np.insert(result, i+1, np.nan)
                else:
                    result[i] = np.nan
            else:
                result[i] = func_result
        except (ZeroDivisionError, ValueError):
            result[i] = np.nan

    return result


## to deal with situations when you have an undefined derivative... for example 1/(x-1)... we need a function that finds at which x the derivative is not defined.
## one way to do this is to find at which point the denominator is zero... write a functino that finds this point by solving the equation...
## vertical asymptotes is another case to consider... occurs when denominator is positive or negavice infinity

def zero_checker(min_val, max_val, fx, arr_input):
    delta = 0.01
    epsilon = 1e-5
    x_test_vals = np.arange(min_val, max_val + delta, delta, dtype=float)
    for x in x_test_vals:
        try:
            # Calculate the derivative at x
            derivative_at_x = derivative(fx, x, dx=1e-6)
            y = fx(x)
            # Check if derivative is close to zero or undefined (division by zero)
            if np.abs(derivative_at_x) < epsilon or np.isinf(y):
                print(x)
                arr_input = np.concatenate((arr_input, np.array([x]))) if arr_input is not None else np.array([x])
        except (ValueError, ZeroDivisionError):
            print(x)
            # Catch exceptions such as division by zero or undefined functions
            arr_input = np.concatenate((arr_input, np.array([x]))) if arr_input is not None else np.array([x])
    return arr_input


def Find_sig(function: str,inputArr: np.array):
    undefined_derivative_points = []
    # Define the symbolic variable
    x = sp.symbols('x')
    #get diravtaive of symbolic form
    sp_f_prime = sp.diff(function, x)
    # Convert the string expression to a SymPy expression
    sp_function = sp.sympify(function)
    #solving for points where the function = 0
    zero = solve(sp_function,0)
    print("the x values that make the function zero is ",zero,"\n")

    #evalulate limits from + and - side
    for x_in in inputArr:
        left_hand_limit = sp.limit(sp_f_prime,x,x_in,dir="-")
        right_hand_limit = sp.limit(sp_f_prime,x,x_in,dir="+")
        # print("the value ",x_in," aproaches left hand lim of ", left_hand_limit,"\n")
        # print("the value ",x_in," aproaches right hand lim of ", right_hand_limit,"\n")
        if left_hand_limit != right_hand_limit:
            print("The limit at ",x_in," does not exist\n")
        # Check if the limit is positive or negative infinity
        if right_hand_limit == sp.oo or left_hand_limit == sp.oo:
            undefined_derivative_points.append(x_in)
        elif right_hand_limit == -sp.oo or left_hand_limit == -sp.oo:
            undefined_derivative_points.append(x_in)
 
    
    # Step 3: Find points where the derivative might be undefined
    # Get the denominator of the derivative if it is a fraction
    if isinstance(sp_f_prime, sp.Rational):
        denominator = sp.denom(sp_f_prime)
    else:
        denominator = sp_f_prime.as_numer_denom()[1]
    
    # Solve for points where the denominator is zero
    critical_points = sp.solveset(denominator, x, domain=sp.S.Reals)
    
    # Combine with points where the original function is undefined (if any)
    expr_denominator = sp_function.as_numer_denom()[1]
    undefined_points = sp.solveset(expr_denominator, x, domain=sp.S.Reals)
    critical_points = critical_points.union(undefined_points)
    
    # Convert the critical points to numerical values
    critical_points = [point.evalf() for point in critical_points]

     # Step 4: Numerical verification
    func_numeric = sp.lambdify(x, sp_function, 'numpy')
    derivative_numeric = sp.lambdify(x, sp_f_prime, 'numpy')
    
    for point in critical_points:
        try:
            point_value = float(point)
            derivative_value = derivative_numeric(point_value)
            if np.isnan(derivative_value) or np.isinf(derivative_value):
                undefined_derivative_points.append(point_value)
        except:
            undefined_derivative_points.append(point_value)
    for zeros in zero:
        undefined_derivative_points.append(zeros)
    output = np.append(inputArr,undefined_derivative_points)
    print("undefined points of the second dirvative is ", undefined_derivative_points,"\n")
    return output


# Plot the graph of a function and its first and second derivatives
def plot_graph(function, max_domain, min_domain, step_size,sympy_funcion: str):
    if function is None:
        return None
    
    # Generate x values for the given domain and step size
    x_values = np.linspace(min_domain, max_domain, step_size)
    x_values = np.sort(x_values)
    # print("x_values: ")
    # print(x_values)
    new_x_values = find_critical_points(function,x_values,min_domain,max_domain)
    # print("new_x_values: ")
    # print(new_x_values)
    newer_x_values = Find_sig(sympy_funcion,new_x_values)
    # print("newer_x_values after Find_sig: ")
    # print(newer_x_values)
    # Calculate function values
    f_x = function(newer_x_values)
    newer_x_values = zero_checker(min_domain,max_domain,function,newer_x_values)
    newer_x_values = np.sort(newer_x_values)
    # print("newer_x_values after zero_checker: ")
    # print(newer_x_values)
    f_x = replace_infs_with_nans(f_x)
    f_x = replace_singularity_with_nans(function, newer_x_values)
    
    # Calculate first derivative
    f_x_prime = numerical_derivative(function, newer_x_values)
    f_x_prime = replace_infs_with_nans(f_x_prime)
    
    # Calculate second derivative using numpy gradient
    f_2x_prime = np.gradient(f_x_prime, newer_x_values)
    f_2x_prime = replace_infs_with_nans(f_2x_prime)

    # Get extrema and offset for the function
    max_y, min_y, offset = calculate_extrema_and_offset(f_x, 0.1)

    # Create subplots for the function and its derivatives
    fig, axes = plt.subplots(3, sharex=True)
    axes[0].plot(newer_x_values, f_x)
    axes[0].set(ylabel='$f(x)$')

    # Plot the first derivative with positive and negative parts in different colors
    neg_indices = f_x_prime < 0
    temp = create_nan_array_copy(f_x_prime, neg_indices)
    axes[1].plot(newer_x_values, f_x_prime, color="black")
    axes[1].plot(newer_x_values, temp, color='red')
    temp = create_nan_array_copy(f_x_prime, ~neg_indices)
    axes[1].plot(newer_x_values, temp, color='blue')
    axes[1].set(ylabel="$first der \' f(x)$")

    # Set limits to the y-axis of the subplots
    max_y, min_y, offset = calculate_extrema_and_offset(f_x_prime, 0.1)
    axes[1].set_ylim(min_y - offset, max_y + offset)

    # Plot the second derivative
    f_2x_prime[np.abs(f_2x_prime) < 1e-5] = 0
    axes[2].set(ylabel='$second der\'f(x)$')
    axes[2].plot(newer_x_values, f_2x_prime, color='green', linestyle='solid')

    # Set limits to the y-axis of the subplots
    max_y, min_y, offset = calculate_extrema_and_offset(f_2x_prime, 0.5)
    axes[2].set_ylim(min_y - offset, max_y + offset)

    return f_x, f_x_prime, f_2x_prime, fig, newer_x_values

# Receive user input for the domain
def get_domain():
    max_domain = st.text_input('Insert a number (max domain): ', key="max", value="1")
    min_domain = st.text_input('Insert a number (min domain): ', key="min", value="-1")
    
    try:
        if max_domain  == min_domain:
            st.error("Max and min domain cannot be equal.")
            return -1, 1
    
        max_domain, min_domain = float(max_domain), float(min_domain)

        # Check if both inputs are empty or zero
        if max_domain == 0 and min_domain == 0:
            st.error("Both max and min domain cannot be zero.")
            return -1, 1
        elif max_domain == min_domain:
            st.error("Max and min domain cannot be equal.")
            return -1, 1
        else:
            return float(max_domain), min_domain
    except ValueError:
        # Handle the case where the input cannot be converted to float
        st.error("Invalid input. Please enter valid numerical values for the domain.")
        return -1, 1

# Streamlit app title
st.title("Graph of derivative")

# User input for the domain and function
max_domain, min_domain = get_domain()
st.write('The current domain is [', min_domain, ' , ', max_domain, ']')
step_size = st.number_input(label="Number of Points", min_value=50, max_value=50000)
function_str = str(st.text_input("Function: "))
if not function_str.strip():
    # function_str = "x**3"
    function_str = "1/(x-0.99)"

# Calculate and plot the graph
user_function = parse_function(function_str)
if user_function is not None:
    f_x, f_x_prime, f_2x_prime, fig, x_input = plot_graph(user_function, max_domain, min_domain, step_size,function_str)
    st.pyplot(fig)

f_x_prime_sp = sp_der(function_str, x_input)

# compare_values(f_x_prime_sp, f_x_prime)

data = {
    'function': function_str,
    'x_values': x_input,
    'f_x': f_x,
    'f_x_prime': f_x_prime,
    'f_x_prime_sp': f_x_prime_sp,
    'f_x_prime_diff': f_x_prime - f_x_prime_sp,
    'f_2x_prime': f_2x_prime
}

df = pd.DataFrame(data)
st.write(df)