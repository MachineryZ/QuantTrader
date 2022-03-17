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
from gplearn.genetic import SymbolicRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.random import check_random_state
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import graphviz
from gplearn.functions import make_function
from gplearn.genetic import SymbolicTransformer
from sklearn.utils import check_random_state
from sklearn.datasets import load_boston
import numpy as np

from gplearn.genetic import SymbolicClassifier
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.datasets import load_breast_cancer


def SymbolicRegressorExample():
    
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
    plt.savefig("gplearn_result/example1.png")

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
    # 17. warm_start: whether reuse the solution of the previous call to fit and add more generations
    # 18. low_memory: only the current generation is retained. Parent information is discarded
    # 19. n_jobs: the number of jobs to run in parallel for 'fit'
    # 20. verbose: controls the verbosity of the evolution building process
    # 21. random_state: int: seed used by the random number generator, randomstate instance:, None, use np.random as randomstate number generator
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

    # DecisionTreeRegressor
    est_tree = DecisionTreeRegressor()
    est_tree.fit(x_train, y_train)
    est_rf = RandomForestRegressor(n_estimators=10)
    est_rf.fit(x_train, y_train)

    # Visualize y
    # np.ndarray.ravel: a differnt kind of np.ndarray.reshape(:, -1), in different order
    y_gp = est_gp.predict(np.c_[x0.ravel(), x1.ravel()]).reshape(x0.shape)
    score_gp = est_gp.score(x_test, y_test)
    y_tree = est_tree.predict(np.c_[x0.ravel(), x1.ravel()]).reshape(x0.shape)
    score_tree = est_tree.score(x_test, y_test)
    y_rf = est_rf.predict(np.c_[x0.ravel(), x1.ravel()]).reshape(x0.shape)
    score_rf = est_rf.score(x_test, y_test)

    fig = plt.figure(figsize=(12,10))
    for i, (y, score, title) in enumerate([
        (y_truth, None, "Ground Truth"),
        (y_gp, score_gp, "SymbolicRegressor"),
        (y_tree, score_tree, "DecisionTreeRegressor"),
        (y_rf, score_rf, "RandomForestRegressor"),
    ]):
        ax = fig.add_subplot(2,2, i+1, projection='3d')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_xticks(np.arange(-1, 1.01, .5))
        ax.set_yticks(np.arange(-1, 1.01, .5))
        surf = ax.plot_surface(x0, x1, y, rstride=1, cstride=1, color="green", alpha=0.5)
        points = ax.scatter(x_train[:, 0], x_train[:, 1], y_train)
        if score is not None:
            score = ax.text(-0.7, 1, .2, "$R^2 = \/ %.6f$" % score, 'x', fontsize=14)
        plt.title(title)
    plt.savefig("gplearn_result/example2.png")

    # Research into the construction of a certain tree
    dot_data = est_gp._program.export_graphviz()
    # graph = graphviz.Source(dot_data)
    # graph.render('gplearn_result/example3.png', format="png", cleanup=True)
    print(est_gp._program.parents)

    idx = est_gp._program.parents['donor_idx']
    fade_nodes = est_gp._program.parents['donor_nodes']
    print(est_gp._programs[-2][idx])
    print("Fitness:", est_gp._programs[-2][idx].fitness_)
    # dot_data = est_gp._programs[-2][idx].export_graphviz(fade_nodes=fade_nodes)
    # graph = graphviz.Source(dot_data)
    # print(graph)
    
    idx = est_gp._program.parents['parent_idx']
    fade_nodes = est_gp._program.parents['parent_nodes']
    print(est_gp._programs[-2][idx])
    print('Fitness:', est_gp._programs[-2][idx].fitness_)
    # dot_data = est_gp._programs[-2][idx].export_graphviz(fade_nodes=fade_nodes)
    # graph = graphviz.Source(dot_data)
    # print(graph)

def SymbolicTransformerExample(datasets="boston"):

    if datasets == "boston":
        rng = check_random_state(0)
        housing = load_boston()
        perm = rng.permutation(housing.target.size)
        housing.data = housing.data[perm]
        housing.target = housing.target[perm]

    # Due to the ethical problem load_boston is deprecated in 1.0
    # and will be removed in 1.2
    # The alternative dataset is California housing dataset

    if datasets == "california":
        from sklearn.datasets import fetch_california_housing
        rng = check_random_state(0)
        housing = fetch_california_housing()
        perm = rng.permutation(housing.target.size)
        perm = rng.permutation(housing.target.size)
        housing.data = housing.data[perm]
        housing.target = housing.target[perm]


    from sklearn.linear_model import Ridge
    est = Ridge()
    est.fit(housing.data[:300, :], housing.target[:300])
    print(est.score(housing.data[300:, :], housing.target[300:]))

    function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'log',
                    'abs', 'neg', 'inv', 'max', 'min']
    # Different Parameters:
    # 1.hall_of_fame
    gp = SymbolicTransformer(
        generations=20,
        population_size=2000,
        hall_of_fame=100,
        n_components=10,
        function_set=function_set,
        parsimony_coefficient=0.0005,
        max_samples=0.9,
        verbose=1,
        random_state=0,
    )    
    gp.fit(housing.data[:300, :], housing.target[:300])
    gp_features = gp.transform(housing.data)
    new_housing = np.hstack((housing.data, gp_features))

    est = Ridge()
    est.fit(new_housing[:300, :], housing.target[:300])
    print(est.score(new_housing[300:, :], housing.target[300:]))

def Customize(datasets='california'):
    def logic(x1, x2, x3, x4):
        return np.where(x1 > x2, x3, x4)
    logical = make_function(
        function=logic,
        name="logical",
        arity=4,
    )
    function_set = ['add', 'sub', 'mul', 'div', logical]
    gp = SymbolicTransformer(
        generations=2,
        population_size=2000,
        hall_of_fame=100,
        n_components=10,
        function_set=function_set,
        parsimony_coefficient=0.0005,
        max_samples=0.9,
        verbose=1,
        random_state=0
    )
    if datasets == "boston":
        rng = check_random_state(0)
        housing = load_boston()
        perm = rng.permutation(housing.target.size)
        housing.data = housing.data[perm]
        housing.target = housing.target[perm]
    elif datasets == "california":
        from sklearn.datasets import fetch_california_housing
        rng = check_random_state(0)
        housing = fetch_california_housing()
        perm = rng.permutation(housing.target.size)
        perm = rng.permutation(housing.target.size)
        housing.data = housing.data[perm]
        housing.target = housing.target[perm]
    gp.fit(housing.data[:300, :], housing.target[:300])
    print(gp._programs[0][996])

def Classification():
    h = 0.02
    names = [
        "Nearest Neighbors",
        "Linear SVM",
        "RBF SVM",
        "Gaussian Process",
        "Decision Tree",
        "Random Forest", 
        "Neural Net",
        "AdaBoost",
        "naive Bayes", 
        "QDA",
        "SymbolicClassifier",
    ]
    # Define a list of classifiers
    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1, tol=0.001),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
        SymbolicClassifier(random_state=0),
    ]
    x, y = make_classification(
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=1,
        n_clusters_per_class=1,
    )
    rng = np.random.RandomState(2)
    x += 2 * rng.uniform(size=x.shape)
    linearly_separable = (x, y)
    datasets = [
        make_moons(noise=0.3, random_state=0),
        make_circles(noise=0.2, factor=0.5, random_state=1),
        linearly_separable,
    ]
    figure = plt.figure(figsize=(27, 9))
    i = 1
    for ds_cnt, ds in enumerate(datasets):
        x, y = ds
        x = StandardScaler().fit_transform(x)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)
        x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
        y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                np.arange(y_min, y_max, h))
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        if ds_cnt == 0:
            ax.set_title("Input data")
        ax.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cm_bright, edgecolors=None)
        ax.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors=None)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        i += 1
        for name, clf in zip(names, classifiers):
            ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
            clf.fit(x_train, y_train)
            score = clf.score(x_test, y_test)
            if hasattr(clf, "decision_function"):
                Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            else:
                Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, cmap=cm, alpha=0.8)
            ax.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
            ax.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=cm_bright, edgecolors='k', alpha=0.6)
            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
            if ds_cnt == 0:
                ax.set_title(name)
            ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                    size=15, horizontalalignment='right')
            i += 1
    plt.tight_layout()
    plt.savefig("gplearn_result/example3.png")

    

if __name__ == "__main__":
    SymbolicRegressorExample()
    SymbolicTransformerExample(datasets="california")
    Customize(datasets="california")
    Classification()
