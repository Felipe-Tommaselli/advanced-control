import numpy as np
import re 
from numpy import cos, sin

def PlotExpr(n_values, expr, LIM=10):
    """
    Converts a UnitImpulse or UnitStep expression from LCapy to plot in matplotlib.

    Parameters:
    expr (Expr): LCapy's UnitImpulse(n - k) expression, for example
    n_values (array): Array of n values to evaluate the expression and plot
    LIM (int): Limit to plot the graph
    """

    expr_str = str(expr)
    impulse = np.zeros(len(n_values))
    step = np.zeros(len(n_values))
    
    # Check if the string contains 'UnitImpulse'
    if 'UnitImpulse' in expr_str:
        # Extract the impulse shift from the string
        shift_str = expr_str.split('UnitImpulse(')[1].split(')')[0].strip()
        try:
            # Convert the shift to an integer
            shift_str = shift_str.replace('n', '0')
            shift = int(eval(shift_str))
        except ValueError:
            print("Error: Shift is not a valid integer.")
            return
        
        impulse = np.zeros(len(n_values))
        
        # Place the impulse at the correct position based on the shift
        if (shift + LIM) < len(impulse) and (shift + LIM) >= 0:
            impulse[shift + LIM] = 1
        
    if 'UnitStep' in expr_str:
        # Extract the step shift from the string
        shift_str = expr_str.split('UnitStep(')[1].split(')')[0].strip()
        try:
            # Convert the shift to an integer
            shift_str = shift_str.replace('n', '0')
            shift = int(eval(shift_str))
        except ValueError:
            print("Error: Shift is not a valid integer.")
            return np.zeros(len(n_values))
        
        # Create the unit step function vector
        step = np.zeros(len(n_values))
        
        # Fill the unit step function based on the shift
        step[n_values >= shift] = 1
    
    # Replace all integers with floats, but ignore numbers that are already floats
    expr_str = re.sub(r'(?<!\.)\b(\d+)\b(?!\.\d)', r'\1.0', expr_str)

    # Replace negative powers to ensure they are applied to floats
    expr_str = re.sub(r'(\d+)\*\*(\-?\d+)', r'(\1.0)**(\2)', expr_str)

    # Check if the expression contains 'UnitImpulse' or 'UnitStep'
    expr_str = re.sub(r'UnitImpulse\([^)]*\)', 'impulse', expr_str)
    expr_str = re.sub(r'UnitStep\([^)]*\)', 'step', expr_str)

    # Replace n by 0 
    expr_str = expr_str.replace('n', 'n_values')

    # Create a context for evaluation based on the variables present in the expression
    context = {}
    context_test = {
        'T': 1,
        'impulse': impulse,
        'step': step,
        'cos': cos,
        'sin_values': sin,
        'atan_values': np.arctan,
        'sqrt': np.sqrt,
        'n_values': n_values,
    }
    # Add variables to the context based on the dictionary
    for key, value in context_test.items():
        if key in expr_str:
            context[key] = value

    try:
        # Evaluate the expression
        signal = eval(expr_str, context)
    except Exception as e:
        print(f"Error evaluating the expression: {e}")
        raise e

    return signal