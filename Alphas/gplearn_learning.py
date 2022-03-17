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
    # 4. 
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

