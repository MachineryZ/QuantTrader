"""
Refer to:
https://github.com/trevorstephens/gplearn/blob/master/doc/gp_examples.ipynb
This file aims to get better comprehend and understand gplearn
1. Example of symbolic regressor
2. Example of symbolic transformer
3. Example of customizing your programs
4. Example of classification
5. 
"""
def SymbolicRegressorExample():
    from gplearn.genetic import SymbolicRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.utils.random import check_random_state
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import numpy as np
    import graphviz
    
    # Ground Truth
    x0 = np.arange(-1, 1, .1)
    x1 = np.arange(-1, 1, .1)
    x0, x1 = np.meshgrid(x0, x1)
    y_truth = x0**2 - x1**2 + x1 - 1

    ax = plt.figure().gca(projection="3d")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xticks(np.arange(-1, 1.01, .5))
    ax.set_yticks(np.arange(-1, 1.01, .5))
    surf = ax.plot_surface(x0, x1, y_truth, rstride=1, cstride=1, color='green', alpha=0.5)
    plt.show()

    rng = check_random_state(0)

    # Training Samples
    x_train = rng.uniform(-1, 1, 100).reshape(50, 2)
    y_train = x_train[:, 0] ** 2 - x_train[:, 1] ** 2 + x_train[:, 1] - 1

    # Testing samples
    x_test = rng.uniform(-1, 1, 100).reshape(50, 2)
    y_test = x_test[:, 0] ** 2 - x_train[:, 1] ** 2 + x_test[:, 1] - 1

    # Initialize symbolicregressor
    # 1. population_size: the number ofr programs in each generation
    # 2. generations: the number of generations to evolve
    # 3. tournament_size: the number of programs that will compete to become part of the next generation
    # 4. stopping_criteria: the required metric value required in order to stop evolution early
    # 5. const_range: tuple of two floats, or None, default(-1, 1), the range of constants to include in formulas
    # 6. init_depth: the range of tree depths for the initial population of naive formulas
    # 7. init_method: 'grow', 'full', 'half and half', 
    # 8. function_set: default=('add', 'sub', 'mul', 'div') the functions to use when building and evolving programs
    # 9. metric: the name of the raw fitness metric, 'mean absolute error', 'mse', 'rmse', 'pearson', 'spearman')
    # 10. parsimony_coefficient: constant penalizes large programs by adjusting their fitness to be less favorable for selection
    #       like occam razor, avoid meaningless complexity increasing
    # 11. p_crossover: select a tournament winner to substitute its subtree into random tree's subtree
    # 12. p_subtree_mutation: select a tournament winner to substitute its subtree into a random whole tree
    # 13. p_hoist_mutation: select a tournament winner to do hoist mutation (raise the node's level)
    # 14. p_point_replace: select a tournament winner to replace a random nodes 
    # 15. max_samples: the fraction of samples to draw from x to evaluate each program on, default 1.0
    # 16. feature_names: column name
    # 17.
    est_gp = SymbolicRegressor(
        population_size=5000,
        generations=20,
        stopping_criteria=0.01,
        p_crossover=0.7,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.05,
        p_point_mutation=0.1,
        max_samples=0.9,
        verbose=1,
        parsimony_coefficient=0.01,
        random_state=0,
    )
    est_gp.fit(x_train, y_train)
    print(est_gp._program)



if __name__ == "__main__":
    SymbolicRegressorExample()

