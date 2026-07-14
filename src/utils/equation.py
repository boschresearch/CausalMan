# Code for the CausalMan: A Digital-Twin for Large-Scale Causality
# Copyright (c) 2022 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from sympy import Symbol, symbols, Eq
from sympy.stats import Normal, Uniform
from sympy.stats.rv import RandomSymbol


# FIXME: WHy these dummy equations? Rewrite using the actual equations.
def create_distrib_fixed(
    variableName: str,
    constantValue: float,
) -> Eq:
    """
    Create a fixed distribution equation.

    Args:
        variableName (str): The name of the variable.
        constantValue: The constant value for the equation.

    Returns:
        Equation: The equation representing the fixed distribution.
    """
    # dummy equation to justify import
    Eq(symbols("dummy"), 2.0)

    Equ_str = f"Eq(symbols('{variableName}'), {constantValue})"
    Equ = eval(Equ_str)
    return Equ


def create_distrib_normal(
    variableName: str,
    mean: float,
    std: float,
) -> Eq:
    """
    Create a normal distribution equation.

    Args:
        variableName (str): The name of the variable.
        mean (float): The mean of the normal distribution.
        std (float): The standard deviation of the normal distribution.

    Returns:
        Eq: The equation representing the normal distribution.
    """
    # dummy equation to justify import
    Eq(symbols("dummy"), Normal("dummy", 0, 1))

    #distrib = Normal(variableName, mean, std)
    Equ_str = f"Eq(symbols('{variableName}'), Normal('{variableName}', {mean}, {std}))"
    Equ = eval(Equ_str)
    return Equ


def create_distrib_uniform(
    variableName: str,
    lower_limit: float,
    upper_limit: float,
) -> Eq:
    """
    Create a uniform distribution equation.

    Args:
        variableName (str): The name of the variable.
        lower_limit (float): The lower limit of the uniform distribution.
        upper_limit (float): The upper limit of the uniform distribution.

    Returns:
        Eq: The equation representing the uniform distribution.
    """
    # dummy equation to justify import
    Eq(symbols("dummy"), Uniform("dummy", 0, 1))

    Equ_str = (
        f"Eq(symbols('{variableName}'), Uniform('{variableName}', "
        f"{lower_limit}, {upper_limit}))"
    )
    Equ = eval(Equ_str)
    return Equ


def create_distrib_categorical(
    variableName: str,
    categories: list,
    probabilities: list,
) -> Eq:
    """
    Create a categorical distribution equation.

    Args:
        variableName (str): The name of the variable.
        categories (list): The categories of the categorical distribution.
        probabilities (list): The probabilities of the categories.

    Returns:
        Eq: The equation representing the categorical distribution.
    """

    if len(categories) != len(probabilities):
        raise ValueError(
            "The number of categories and probabilities must be equal."
        )
    if np.sum(probabilities) != 1:
        raise ValueError("The sum of probabilities must be equal to 1.")
    
    # dummy equation to justify import
    Eq(symbols("dummy"), Uniform("dummy", 0, 1))

    density = {}
    for cat in categories:
        density.update({cat: probabilities[cat]})

    #distrib = Categorical(variableName, categories, probabilities)
    Equ_str = (
        f"Eq(symbols('{variableName}'), FiniteRV('{variableName}', "
        f"{density}))"
    )

    Equ = eval(Equ_str)
    return Equ

def intervention_str2dict(int_str: str)-> dict[Symbol, RandomSymbol | float]:
    """Given a string describing the intervention, convert it to a dictionary 
    containing sympy equations.
    The intervention string should be in this format:
    "A=0,B=1,C=3"

    Args:
        int_str (str): Intervention described as a string.

    Returns:
        dict[Symbol, RandomSymbol | float]: Dictionary of interventions.
    """
    intervention_dict = {}

    # split the string by comma
    int_str = int_str.split(';')
    for intervention in int_str:
        if intervention == '':
            continue
        # split the string by colon
        intervention = intervention.split('=')
        
        # Initialize RV.
        Equ = eval(intervention[1])
        # add the intervention to the dictionary
        intervention_dict[Symbol(intervention[0])] = Equ

    return intervention_dict