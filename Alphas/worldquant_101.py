import pandas
import numpy as np

# ==================== Auxilary functions ====================
# 
# 
def correlation(x, y, window=10):
    """
    Wrappe function to estimate rolling correlations
    x: pandas.DataFrame
    y: pandas.DataFrame,
    returns: pandas.DataFrame
    """
    return x.rolling(window).corr(y)

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
