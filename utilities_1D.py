import os
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Math
import imageio
import shutil

def print_pde(w, rhs_description, ut='u_t', precision=5, threshold=8e-2):
    """
    Formats and prints a PDE in a human-readable or LaTeX form.

    Parameters:
        w (array-like): Coefficients of the terms in the PDE.
        rhs_description (list): Descriptions of the terms corresponding to the coefficients.
        ut (str, optional): Left-hand side of the PDE (default: 'u_t').
        precision (int, optional): Number of decimal places to display (default: 5).
        threshold (float, optional): Threshold below which terms are considered zero (default: 5e-2).

    Returns:
        None. Displays the PDE in a properly formatted LaTeX format if in Jupyter.
    """
    terms = []
    latex_terms = []
    w = np.atleast_1d(w).flatten()  # Ensure w is a 1D array
    
    for coeff, term in zip(w, rhs_description):
        coeff = np.real(coeff).item()  # Convert to a scalar
        
        if abs(coeff) > threshold:  # Skip small coefficients
            # Console formatting: Ensures alignment, space after `-`
            formatted_coeff = f"{coeff:+.{precision}f}".replace("+", " ")  # Space for alignment
            terms.append(f"{formatted_coeff}   {term}")  # Extra spaces for readability

            # LaTeX formatting: Use double backslashes `\\;` to properly escape the space command
            formatted_latex_coeff = f"{coeff:+.{precision}f}"
            latex_terms.append(f"{formatted_latex_coeff} \\; {term}")  # Correctly formatted LaTeX space

    if not terms:
        pde = f"{ut} = 0"  # If all terms are zero
        latex_pde = f"{ut} = 0"
    else:
        pde = f"{ut} =\n    " + "\n    ".join(terms)  # Newline for better readability
        latex_pde = f"{ut} = " + " ".join(latex_terms)  # Keep `+` signs in LaTeX

    # Ensure LaTeX syntax consistency
    latex_pde = latex_pde.replace("u_x", "u_{x}").replace("u_xx", "u_{xx}").replace("u_xxx", "u_{xxx}")
    latex_pde = latex_pde.replace("u_t", "u_{t}")  # Consistent notation

    try:
        display(Math(latex_pde))  # Display LaTeX output properly
    except:
        pass  # If not in Jupyter, just print the formatted text
    
    print(pde)  # Print for fallback in non-Jupyter environments



def get_pde_string(w, rhs_description, ut='u_t', precision=5, threshold=8e-2):
    """
    Returns a LaTeX-formatted string representing the PDE identified by sparse regression.
    
    Parameters:
        w (ndarray): Coefficient vector.
        rhs_description (list): List of term descriptions.
        ut (str): Left-hand side variable (default 'u_t').
        precision (int): Number of decimal places.
        threshold (float): Terms with |coefficient| below this are omitted.
    
    Returns:
        str: A LaTeX string for the PDE (without the ut and equals sign).
    """
    w = np.atleast_1d(w).flatten()
    latex_terms = []
    for coeff, term in zip(w, rhs_description):
        coeff = np.real(coeff).item()
        if abs(coeff) > threshold:
            formatted_latex_coeff = f"{coeff:+.{precision}f}"
            latex_terms.append(f"{formatted_latex_coeff} \\; {term}")
    if not latex_terms:
        latex_pde = "0"
    else:
        latex_pde = " ".join(latex_terms)
    return latex_pde


def normalize_data(data):
    """Normalize a numpy array to the [0,1] range."""
    return (data - data.min()) / (data.max() - data.min())
