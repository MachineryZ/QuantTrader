import numpy as np
import pandas
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

