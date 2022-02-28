import pandas
import numpy as np

# ==================== Auxilary functions ====================
# Refer to: https://github.com/wpwpwpwpwpwpwpwpwp/Alpha-101-GTJA-191/blob/master/101Alpha_code_1.py
# 
def correlation(x, y, window=10):
    """ Wrapper function correlation

    Args:
        x (pandas.DataFrame): input x
        y (pandas.DataFrame): input y
        window (int, optional): rolling window size. Defaults to 10.

    Returns:
        _type_: return rolling correlation result
    """
    return x.rolling(window).corr(y)

def rank(df):
    """Wrapper function rank

    Args:
        df (pandas.DataFrame): 

    Returns:
        pandas.DataFrame: 
        Example:
        s = [1, 2, 3, 2, 1]
        res = [0.3, 0.7, 1.0, 0.7, 0.3],
        s = [1, 2, 3, 4, 100, 9]
        res = [1/6, 2/6, 3/6, 4/6, 6/6, 5/6]
    """
    return df.rank(pct=True)

def delta(df, period=1):
    """_summary_

    Args:
        df (pandas.DataFrame): 
        period (int, optional): Defaults to 1.
    """


class WorldQuant_101_Alphas(object):
    def __init__(self, df_data):
        self.open = df_data['open']
        self.close = df_data['close']
        self.high = df_data['high']
        self.low = df_data['low']
        self.volume = df_data['volume']
        self.returns = df_data['returns']
        self.amount = df_data['amount']
        self.vwap = self.amount * 1000 / (self.volume + 1e-8)

    # alpha_001: (rank(ts_argmax(signedpower((returns < 0) > stddev(returns, 20) : close), 2.), 5)) - 0.5)
    def alpha_001(self):
        pass

    # alpha_002: (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))
    def alpha_002(self):
        df = -1 * correlation(rank(delta(log(self.volume), 2)), rank((self.close - self.open) / self.open), 6)
