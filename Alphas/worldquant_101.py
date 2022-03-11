import pandas
import numpy as np
from numpy import abs
from numpy import log
from numpy import sign
from scipy.stats import rankdata

# ==================== Auxilary functions ====================
# Refer to: https://github.com/wpwpwpwpwpwpwpwpwp/Alpha-101-GTJA-191/blob/master/101Alpha_code_1.py
# use google docstring

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
    """Wrapper function delta(difference)

    Args:
        df (pandas.DataFrame): input
        period (int, optional): . Defaults to 1.

    Returns:
        _type_: _description_
    """
    return df.diff(period)

def log(df):
    """Wrapper function log(base e)

    Args:
        input (pandas.DataFrame, np.ndarray, optional): input

    Returns:
        np.ndarray: returns
    """
    return np.log(df)

def ts_rank(df, window=10):
    """Wrapper function to compute the last element
    in a time series 

    Args:
        df (pandas.dataFrame): 
        window (int, optional): Defaults to 10.

    Returns:
        pandas.DataFrame: 
    
    Example:
    input:
    [1    2    2    3   1], period = 2:
    process:
     1    2
          1.5  1.5 (the same)
               1    2
                    2   1
    return:
    [nan  2    1.5  2   1]
    """
    return df.rolling(window).apply(rolling_rank)

def rolling_rank(df):
    return rankdata(df)[-1]

def sma(df, window=10):
    """Wrapper function to estimate SMA(simplest moving average)
    Also can be renamed as ts_mean function, which means rolling mean

    Args:
        df (pandas.DataFrame):
        window (int, optional): Defaults to 10.

    Returns:
        pandas.DataFrame: 
    """
    return df.rolling(window).mean()

def ts_sum(df, window=10):
    """Wrapper function to estimate rolling sum

    Args:
        df (pandas.DataFrame()): 
        window (int, optional): Defaults to 10.

    Returns:
        pandas.DataFrame(): _description_
    """
    return df.rolling(window).sum()

def delay(df, period=1):
    """Wrapper function to estimate lag

    Args:
        df (pandas.DataFrame):
        period (int, optional): Defaults to 1.

    Returns:
        pandas.DataFrame:
    
    Example:
    input: [1, 2, 3, 2, 1], 1
    output: [NaN, 1, 2, 3, 2]
    """
    return df.shift(period)

def ts_min(df, window=10):
    """Wrapper function to estimate rolling min

    Args:
        df (pandas.DataFrame):
        window (int, optional): Defaults to 1.

    Returns:
        pandas.DataFrame: 
    """
    return df.rolling(window).min()

def ts_max(df, window=10):
    """Wrapper function to estimate rolling max

    Args:
        df (pandas.DataFrame()): 
        window (int, optional): Defaults to 10.

    Returns:
        pandas.DataFrame: 
    """
    return df.rolling(window).max()


def covariance(x, y, window=10):
    """Wrapper function to estimate rolling covariance

    Args:
        x (pandas.DataFrame): 
        y (pandas.DataFrame): 
        window (int, optional): Defaults to 10.

    Returns:
        pandas.DataFrame: 
    """
    return x.rolling(window).cov(y)

def stddev(df, window=10):
    """Wrapper function to estimate rolling standard deviation

    Args:
        df (pandas.DataFrame): 
        window (int, optional): Defaults to 10.

    Returns:
        pandas.DataFrame: 
    """
    return df.rolling(window).std()

def scale(df, k=1):
    """Wrapper function to scale in time series

    Args:
        df (pandas.DataFrame): 
        k (int, optional): Defaults to 1.

    Returns:
        pandas.DataFrame: multiply k times, divided by the abs summation
    """
    return df.mul(k).div(np.abs(df).sum())

def product(df, window=10):
    """_summary_

    Args:
        df (_type_): _description_
        window (int, optional): _description_. Defaults to 10.

    Returns:
        _type_: _description_
    """
    return df.rolling(window).apply(rolling_prod)

def rolling_prod(x):
    """Auxiliary function to be used in pd.rolling_apply

    Args:
        x (np.ndarray): 

    Returns:
        np.ndarray: 
    """
    return np.prod(x)

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
        # df.replace: replace -np.inf, np.inf with 0
        # df.fillna: fill np.nan with 0
        # another way:
        # return df.replace([-np.inf, np.inf, np.nan], 0)
        return df.replace([-np.inf, np.inf], 0).fillna(value=0)

    # alpha_003: (-1 * correlation(rank(open), rank(volume), 10))
    def alpha_003(self):
        df = -1 * correlation(rank(self.open), rank(self.volume), 10)
        return df.replace([-np.inf, np.inf], 0).fillna(value=0)

    # alpha_004: (-1 * ts_rank(rank(low), 9))
    def alpha_004(self):
        return -1 * ts_rank(rank(self.low), 9)

    # alpha_005: (rank((open - (sum(vwap, 10)/10))) * (-1 * abs(rank((close - vwap)))))
    def alpha_005(self):
        return (rank(self.open - (sum(self.vwap, 10) / 10))) * (-1 * abs(rank(self.close - self.vwap)))

    # alpha_006: -1 * correlation(open, volume, 10)
    def alpha_006(self):
        return (-1 * correlation(self.open, self.volume, 10))

    # alpha_007: ((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1 * 1))
    def alpha_007(self):
        adv20 = sma(self.volume, 20)
        alpha = -1 * ts_rank(abs(delta(self.close, 7)), 60) * sign(delta(self.close, 7))
        alpha[adv20 >= self.volume] = -1
        return alpha
    
    # alpha_008: (-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10))))
    def alpha_008(self):
        return -1 * (rank(((ts_sum(self.open, 5) * ts_sum(self.returns, 5)) -
                delay((ts_sum(self.open, 5) * ts_sum(self.returns, 5)), 10))))

    # alpha_009: ((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ? 
    # delta(close, 1) : (-1 *delta(close, 1))))
    def alpha_009(self):
        alpha = -1 * delta(self.close, 1)
        cond_1 = ts_min(delta(self.close, 1), 5) > 0
        cond_2 = ts_max(delta(self.close, 1), 5) < 0
        alpha[cond_1 | cond_2] = delta(self.close, 1)
        return alpha

    # alpha_010: rank((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : 
    # ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1):(-1 * delta(close, 1))))
    def alpha_010(self):
        alpha = - delta(self.close, 1)
        cond_1 = 0 < ts_min(delta(self.close, 1), 4)
        cond_2 = ts_max(delta(self.close, 1), 4) < 0
        alpha[cond_1 | cond_2] = delta(self.close, 1)
        # Notice: https://github.com/wpwpwpwpwpwpwpwpwp/Alpha-101-GTJA-191/blob/master/101Alpha_code_1.py#L341
        # should add rank function according to the origin paper
        return rank(alpha)

    # alpha_011: ((rank(ts_max(vwap - close), 3)) + rank(ts_min((vwap - close, 3))) * rank(delta(volume, 3))
    def alpha_011(self):
        return ((rank(ts_max((self.vwap - self.close), 3)) + 
            rank(ts_min((self.vwap - self.close), 3))) * rank(delta(self.volume, 3)))

    # alpha_012: (sign(delta(volume, 1)) * (-1 * delta(close, 1))
    def alpha_012(self):
        return sign(delta(self.volume, 1)) * (-1 * delta(self.close, 1))

    # alpha_013: (-1 * rank(covariance(rank(close), rank(volume), 5)))
    def alpha_013(self):
        return -1 * rank(covariance(rank(self.close), rank(self.volume), 5))

    # alpha_014: ((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))
    def alpha_014(self):
        return (-1 * rank(delta(self.returns, 3))) * correlation(self.open, self.volume, 10)

    # alpha_015: (-1 * sum(rank(correlation(rank(high, rank(volume), 3)), 3))
    def alpha_015(self):
        return -1 * sum(rank(correlation(rank(self.high), rank(self.volume), 3)), 3)

    # alpha_016: (-1 * rank(covariance(rank(high), rank(volume), 5)))
    def alpha_016(self):
        return -1 * rank(covariance(rank(self.high), rank(self.volume), 5))

    # alpha_017: (((-1 * rank(ts_rank(close, 10))) * rank(delta(delta, close, 1), 1))) *
    # rank(ts_rank((volume / adv20), 5)))
    def alpha_017(self):
        adv20 = sma(self.volume, 20)
        return -1 * ((rank(ts_rank(self.close, 10))) * 
                    rank(delta(self.close, 1), 1) * 
                    rank(ts_rank((self.volume / adv20), 5)))
    
    # alpha_018: (-1 * rank(((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open, 10))))
    def alpha_018(self):
        df = correlation(self.close, self.open, 10)
        df = df.replace([-np.inf, np.inf, np.nan], 0)
        return -1 * (rank((stddev(abs((self.close - self.open)), 5) + (self.close - self.open)) + df))

    # alpha_019: ((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * (1 + rank((1 + sum(returns, 250)))))
    def alpha_019(self):
        return ((-1 * sign(((self.close - delay(self.close, 7)) + delta(self.close, 7)))) * (1 + rank((1 + sum(self.returns, 250)))))
    
    # alpha_020: (((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * rank((open - delay(low, 1))))
    def alpha_020(self):
        return (((-1 * rank((self.open - delay(self.high, 1)))) * rank((self.open - delay(self.close, 1)))) * rank((self.open - delay(self.low, 1))))

    # alpha_021: ((((sum(close, 8)/8) + stddev(close, 8)) < (sum(close, 2) / 2)) ? (-1 * 1) : (((sum(close,
    # 2) / 2) < ((sum(close, 8) / 8) - stddev(close, 8))) ? 1 : (((1 < (volume / adv20)) || ((volume / adv20) == 1)) ? 1 : (-1*1))))
    def alpha_021(self):
        cond_1 = (sum(self.close, 8) / 8 + stddev(self.close, 8)) < (sum(self.close, 2) / 2)
        cond_2 = sma(self.volume, 20) / self.volume < 1
        alpha = pandas.DataFrame(np.ones_like(self.close), index=self.close.index)
        alpha[cond_1 | cond_2] = -1
        return alpha

    # alpha_022: (-1 * delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))))
    def alpha_022(self):
        return (-1 * (delta(correlation(self.high, self.volume, 5), 5) * rank(stddev(self.close, 20))))

    # alpha_023: (((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)
    def alpha_023(self):
        cond = (sum(self.high, 20) / 20) < self.high
        alpha = pandas.DataFrame(np.zeros_like(self.close), index=self.close.index, columns=['close'])
        alpha.at[cond, 'close'] = -1 * delta(self.high, 2).fillna(value=0)
        return alpha

    # alpha_024: ((((delta((sum(close, 100) / 100), 100) / delay(close, 100)) < 0.05) ||
    # ((delta((sum(close, 100 / 100), 100) / delay(close, 100)) == 0.05)) > (-1 * (close - ts_min(close,
    # 100))) : (-1 * delta(close, 3)))
    def alpha_024(self):
        cond = ((delta(sum(self.close, 100) / 100), 100) / delay(self.close, 100) <= 0.05)
        alpha = -1 * delta(self.close, 3)
        alpha[cond] = -1 * (self.close - ts_min(self.close, 100))
        return alpha  
    
    # alpha_025: rank(((((-1 * returns) * adv20) * vwap) * (high - close)))
    def alpha_025(self):
        adv20 = sma(self.volume, 20)
        return rank(((((-1 * self.returns) * adv20) * self.vwap) * (self.high - self.close)))

    # alpha_026: (-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))
    def alpha_026(self):
        return (-1 * ts_max(correlation(ts_rank(self.volume, 5), ts_rank(self.high, 5), 5), 3))

    # alpha_027: ((0.5 < rank((sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0))) ? (-1): 1)
    def alpha_027(self):
        alpha = rank((ts_sum(correlation(rank(self.volume), rank(self.vwap), 6), 2) / 2.0))
        alpha[alpha < 0.5] = -1
        alpha[alpha >= 0.5] = 1
        return alpha

    # alpha_028: scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))
    def alpha_028(self):
        adv20 = sma(self.volume, 20)
        return scale(((correlation(adv20, self.low, 5) + ((self.high + self.low) / 2)) - self.close))

    # alpha_029: (min(product(rank(rank(scale(log(sum(ts_min(rank(rank((-1 * rank(delta((close- 1),
    # 5))))), 2), 1))))), 1), 5) + ts_rank(delay((-1 * returns), 6), 5))
    def alpha_029(self):


        

def create_fake_date():
    return 0

if __name__ == '__main__':
    stock_df = create_fake_date()