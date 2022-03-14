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
    pass

def size():
    """
    size = ln capitalization
    """
    pass

def earnyild():
    """
    earnyild = 0.68 * epibs + 0.11 * etop + 0.21 * cetop

    epibs: analyst's predict ep(earning to price)
    etop: ttm-ep, 12-months profit / capitalization
    cetop: 12-months cash flow / capitalization
    """
    pass

def resvol():
    """
    Residual Volatility:
    resvol = 0.74 * dastd + 0.16 * cmra + 0.1 * hsigma
    
    dastd: 252 trading days extra return weighted volatility, half-life 42
    cmra: 12 months highest return minus lowest return
    hsigma: beta residual, 252 trading days, half-life 63
    """
    pass

def growth():
    """
    growth = 0.47 + sgro + 0.24 * egro + 0.18 * egibs + 0.11 * egibs_s

    sgro: 5 years sale growth rate, (g1 + g2 + g3 + g4 + g5)/5
    egro: 5 years earning growth rate, (e1 + e2 + e3 + e4 + e5)/5
    egibs: earning growth in long period
    egibs_s: earning growth in short period
    """
    pass

def bp():
    """
    book-to-price
    """
    pass

def leverage():
    """
    leverage = 0.38 * mlev + 0.35 * dtoa + 0.27 * blev

    mlev: market capitalization leverage
    dtoa: total debt / total asset
    belv: net asset + priority stock capitalization + long-term debt / net asset
    """
    pass

def liquidity():
    """
    liquidty = 0.35 * stom + 0.35 * stoq + 0.3 * stoa
    
    stom: turnover of month ST(1)
    stoq: turnover of 3 months ST(3)
    stoa: turnover of 12 months ST(12)
    v_t: volume in t
    s_t: shares in t
    ST(T) = ln(1/T sum_{t=1}^{21T} V_t / S_t)
    """
    pass

def sizenl():
    """
    non-linear size: size ** 3
    1. size weight sum
    2. winsorize
    3. std
    """
    pass
