import torch
import numpy as np

class PseudoData(object):
    def __init__():
        pass

    def generate_xy(self, num_stocks, time_step, feature_size):
        return np.randn(num_stocks, time_step, feature_size), \
            np.randn(num_stocks, time_step, 1)

    def generate_graph(self, num_stock, time_step, feature_size, num_tags):
        x = np.randn(num_stock, time_step, feature_size)
        y = np.randn(num_stock, time_step)
        graph = np.random.randint(0, num_tags, (num_stock, ))
        return x, y, graph

if __name__ == '__main__':
    pass