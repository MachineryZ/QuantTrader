"""
This python file contains msci barra's cne5s, which are 10 mostly used A stock
factors:
BETA
MOMENTUM,
SIZE,
EARNYILD,
RESVOL,
GROWTH,
BTOP,
LEVERAGE,
LIQUIDTY,
SIZENL
"""
import pandas
import numpy as np
# Some data processing methods:
def exponental():
    """Exponential weight addition
    
    w_t = w_t / sum_{i=0}^{T-1}w^t

    """
    pass

def orthogonolization():
    """L,B are two factor matrix
    L = cB + E
    E is the orthogonalized factor matrix
    We can also use Gram-Schmidt Process to 
    get orthogonalized factor matrix
    """
    pass

def winsorized():
    """
    Cut of the quantile into the boundary
    """
    pass

# 10 barra risk factor:
def beta():
    """
    r_t - r_ft = alpha + beta R_t + e_t
    253 trading days
    63 half life exponential weight regression
    """
    pass

def momentum():
    """
    momentum = sum_{t=21}^525 w_t(ln(1 + r_t) - ln(1 + r_ft))
    w_t is the exponential weight
    half life is 126
    """

