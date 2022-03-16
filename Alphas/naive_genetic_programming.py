import numpy as np
import pandas
from scipy.stats import rankdata
from gplearn import genetic
from gplearn.functions import make_function
from gplearn.genetic import SymbolicTransformer, SymbolicRegressor
from gplearn.fitness import make_fitness
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
import pickle
import graphviz


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

def _ts_rank(data):
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
stddev = make_function(function=_stddev, name="stddev", arity=1)
product = make_function(function=_product, name="product", arity=1)
ts_rank = make_function(function=_ts_rank, name="ts_rank", arity=1)
ts_min = make_function(function=_ts_min, name="ts_min", arity=1)
ts_max = make_function(function=_ts_max, name="ts_max", arity=1)
ts_argmax = make_function(function=_ts_argmax, name="ts_argmax", arity=1)
ts_argmin = make_function(function=_ts_argmin, name="ts_argmin", arity=1)
ts_sum = make_function(function=_ts_sum, name="ts_sum", arity=1)

user_function = [delta, delay, rank, scale, sma, stddev, product, ts_rank, ts_min, 
    ts_max, ts_argmax, ts_argmin, ts_sum]

init_function = ['add', 'sub', 'mul', 'div']


# Define Metric Function
def _my_metric(y, yhat, w):
    # sum:
    # value = np.sum(np.abs(y)) +  np.sum(np.abs(y_hat))
    value = np.dot(y, yhat) / np.sqrt(np.power(y, 2).sum() * np.power(yhat, 2).sum())
    return value


my_metric = make_fitness(function=_my_metric, greater_is_better=True)

# Generate Expression
generations = 5
function_set = init_function + user_function
metric = my_metric
population_size = 100
random_state = 0
tournament_size = 20

est_gp = SymbolicTransformer(
    feature_names=fields,
    function_set=function_set,
    generations=generations,
    metric=metric,
    population_size=population_size,
    tournament_size=tournament_size,
    random_state=random_state,
)
train_length = 100
test_length = 20
x_train = np.random.randn(train_length, 5) # open, close, high, low, volume
x_test = np.random.randn(test_length, 5)
y_train = np.random.randn(train_length, )
y_test = np.random.randn(test_length, )

# Fit into gplearn to get the expression
est_gp.fit(x_train, y_train)
with open('naive_gp_model.pkl', 'wb') as f:
    pickle.dump(est_gp, f)
print(est_gp)
best_programs = est_gp._best_programs
best_programs_dict = {}

for p in best_programs:
    factor_name = 'alpha_' + str(best_programs.index(p) + 1)
    best_programs_dict[factor_name] = {"fitness": p.fitness_, "expression": str(p), 
                                        "depth": p.depth_, "length": p.length_}
best_programs_dict = pandas.DataFrame(best_programs_dict).T
best_programs_dict = best_programs_dict.sort_values(by="fitness")
print(best_programs_dict)

def alpha_factor_graph(num):
    factor = best_programs[num-1]
    print(factor)
    print(f"fitnes: {factor.fitness_} \
        depth: {factor.depth_} \
        length: {factor.length_}")
    dot_data = factor.export_graphviz()
    graph = graphviz.Source(dot_data)
    # graph.render('naive_alpha_factor_graph', format='png', cleanup=True)
    return graph

graph10 = alpha_factor_graph(10)
print(graph10)