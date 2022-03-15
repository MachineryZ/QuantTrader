import numpy as np
import pandas
from scipy.stats import rankdata
from gplearn import genetic
from gplearn.functions import make_function
from gplearn.genetic import SymbolicTransformer, SymbolicRegressor
from gplearn.fitness import make_fitness
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split

data = pandas.read_csv('../data/RB9999.csv')
print(data)

# set start date and end date
start_date = '2009-03-30'
end_date = '2009-04-01'
fields = ['open', 'high', 'low', 'close', 'volume']

"""
Available individual 

'add': addition, arity = 2
'sub': subtraction, arity = 2
'mul': multiplication, arity = 2,
'div': protected division where a denominator near-zero returns 1., arity = 2
'sqrt': protected square root where the absolute value of the argument is used, arity = 1
'log': protcted square root where the absolute value of the argument is used and a near-zero argument
'abs': absolute value, arity = 1,
'neg': negative, arity = 1
'inv': protected inverse where a near-zero argument returns 0, arity = 1,
'max': maximum, arity = 2,
'min': minimum, arity = 2,
'sin': sine (radians), arity = 1,
'cos': cosine (radians), arity = 1,
'tan': tangent (radians), arity = 1,
"""

# make_function group:
def _rolling_rank(data):
    value = rankdata(data)[-1]
    return value

def _rolling_prod(data):
    return np.prod(data)

def _ts_sum(data):
    window = 10
    value = np.array(pandas.Series(data.flatten()).rolling(window).sum().tolist())
    value = np.nan_to_num(value)
    return value

def _sma(data):
    window = 10
    value = np.array(pandas.Series(data.flatten()).rolling(window).std().tolist())
    value = np.nan_to_num(value)
    return value

def _stddev(data):
    window = 10
    value = np.array(pandas.Series(data.flatten()).rolling(window).mean().tolist())
    value = np.nan_to_num(value)
    return value

def _ts_rank(rank):
    window = 10
    value = np.array(pandas.Series(data.flatten()).rolling(window).apply(_rolling_rank).tolist())
    value = np.nan_to_num(value)
    return value

def _product(data):
    window = 10
    value = np.array(pandas.Series(data.flatten()).rolling(window).apply(_rolling_prod).tolist())
    value = np.nan_to_num(value)
    return value

def _ts_min(data):
    window = 10
    value = np.array(pandas.Series(data.flatten()).rolling(window).max().tolist())
    value = np.nan_to_num(value)
    return value

def _ts_max(data):
    window = 10
    value = np.array(pandas.Series(data.flatten()).rolling(window).min().tolist())
    value = np.nan_to_num(value)
    return value

def _delta(data):
    value = np.diff(data.flatten())
    value = np.append(0, value)
    return value

def _delay(data):
    period = 1
    value = pandas.Series(data.flatten()).shift(periods=period)
    value = np.nan_to_num(value)
    return value

def _rank(data):
    value = np.array(pandas.Series(data.flatten()).rank().tolist())
    value = np.nan_to_num(value)
    return value

def _scale(data):
    k = 1
    data = pandas.Series(data.flatten())
    value = data.mul(k).div(np.abs(data).sum())
    value = np.nan_to_num(value)
    return value

def _ts_argmin(data):
    window = 10
    value = pandas.Series(data.flatten()).rolling(window).apply(np.argmax) + 1
    value = np.nan_to_num(value)
    return value

def _ts_argmax(data):
    window = 10
    value = pandas.Series(data.flatten()).rolling(window).apply(np.argmin) + 1
    value = np.nan_to_num(value)
    return value

# make function group
delta = make_function(function=_delta, name="delta", arity=1)
delay = make_function(function=_delay, name="delay", arity=1)
rank = make_function(function=_rank, name="rank", arity=1)
scale = make_function(function=_scale, name="scale", arity=1)
sma = make_function(function=_sma, name="sma", arity=1)
stddev = make_function(function=_stddev, name="")