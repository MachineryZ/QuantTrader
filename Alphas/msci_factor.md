MSCI Equity Factor Framework

1. Volatility
2. Quality
3. Momentum
4. Value
5. Sentiment
6. Yield
7. Size
8. Growth
9. Liquidity
10. Macro

1. Volatility:
    1. Beta 
        1. HBETA: History BETA
    2. Residual Volatility 
        1. HSIGMA: History sigma
        2. DASTD: Daily STD, daily excess yield
        3. CMRA: Cummulative Return 
2. Quality:
    1. Leverage:
        1. DTOA
        2. MLEV
        3. BLEV
    2. Profitability:
        1. ATO
        2. GP
        3. GPM
        4. ROA
        5. ROE
    3. Earning Variability:
        1. VSAL
        2. VERN
        3. VFLO
        4. VEPS
    4. Earning Quality
        1. CEE
        2. ABS
        3. ACF
    5. Investment Quality
        1. AGRO
        2. IGRO
        3. CXGRO
3. Momentum
    1. Sentiment
        1. RRIBS
        2. EPIBSC
        3. EARNC
4. Value
    1. Book-to-Price
        1. BTOP
    2. Earning Yield
        1. ETOP
        2. ETOPF
        3. CETOP
        4. EM
    3. Long-Term Reversal
        1. LSTRSTR
        2. LSTHALPHA
    4. OTHER
        1. STOP
        2. CFTOP
5. Sentiment
    1. Sentiment
        1. RRIBS
        2. EPIBS
        3. EARNC
6. Yield
    1. Dividend Yield
        1. DTOP
        2. DTOPF
7. Size
    1. Size
        1. LNSIZE
    2. Mid Cap
        1. NLSIZE
8. Growth
    1. Growth
        1. EGRLF
        2. SGRO
        3. EGRO
9. Liquidity
    1. Liquidity
        1. STOM
        2. STOQ
        3. STOA
        4. ATVR

Validity Test:
1. Section Regression

in sample x_t as variable, future return r_{t+1} as target.
Do ols:
r_{t+1} = a_t + f_t * x_t + epsilon_t
for all section t, we calculate the average |a_t|

2. IC
IC_t = corr(x_t, r_{t+1})
IC means factor exposure is more related to return, the higher the better

Volitility Test:
1.

