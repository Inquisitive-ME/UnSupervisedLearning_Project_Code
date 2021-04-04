import numpy as np
import matplotlib.pyplot as plt
from numpy.ma import mvoid
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.model_selection import ShuffleSplit
from data.wine.Wine_Quality_Data import get_wine_dataset
from sklearn import tree
from sklearn.svm import SVC
from textwrap import wrap

title_fontsize = 24
fontsize = 24
legend_fontsize = 18
default_figure_size = (15, 8)

def get_cv():
    return ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)


def perform_learning_curve(estimator, X, y, scoring, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 10)):
    """
    Reference: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    train_sizes, train_scores, test_scores, fit_times, score_times = \
        learning_curve(estimator, X, y, scoring=scoring, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    return train_sizes, train_scores, test_scores, fit_times, score_times


def plot_learning_curve(train_scores, test_scores, train_sizes, title,  ylim=None, save_fig_name=None, show_plot=True, figsize=default_figure_size, legend_loc='best'):
    """
    Reference: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py

    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    title : str
        Title for the chart.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot learning curve
    plt.rcParams["figure.figsize"] = figsize
    plt.grid()
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    plt.legend(loc=legend_loc, fontsize=legend_fontsize)

    plt.title(title, fontsize=title_fontsize, fontweight='bold')
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xticks(train_sizes)
    plt.xlabel("Training examples", fontsize=fontsize, fontweight='bold')
    plt.ylabel("Score: Accuracy", fontsize=fontsize, fontweight='bold')

    if save_fig_name is not None:
        plt.savefig(save_fig_name)

    if show_plot:
        plt.tight_layout()
        plt.show()


def plot_scalability_curve(fit_times, score_times, train_sizes, title, save_fig_name=None, show_plot=True, ylim=None, figsize=default_figure_size, legend_loc='best'):
    """
    Reference: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """

    fig = plt.figure(figsize=figsize)
    axes = fig.add_subplot(111)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    axes.grid()

    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)
    score_times_mean = np.mean(score_times, axis=1)
    score_times_std = np.std(score_times, axis=1)


    # Plot n_samples vs fit_times
    axes.plot(train_sizes, fit_times_mean, 'o-', label="train time")
    axes.fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)

    axes.plot(train_sizes, score_times_mean, 'o-', label="predict time")
    axes.fill_between(train_sizes, score_times_mean - score_times_std,
                         score_times_mean + score_times_std, alpha=0.1)

    if ylim is not None:
        axes.set_ylim(*ylim)

    axes.set_xlabel("Training examples", fontsize=fontsize, fontweight='bold')
    axes.set_ylabel("time (s)", fontsize=fontsize, fontweight='bold')
    axes.set_title(title, fontsize=title_fontsize, fontweight='bold')
    axes.legend(loc=legend_loc, fontsize=legend_fontsize)

    if save_fig_name is not None:
        plt.savefig(save_fig_name)

    if show_plot:
        plt.tight_layout()
        plt.show()


def plot_performance_curve(test_scores, fit_times, title, ylim=None, save_fig_name=None, show_plot=True, figsize=default_figure_size):
    """
    Reference: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py

    Parameters
    ----------
    title : str
        Title for the chart.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    fig = plt.figure(figsize=figsize)
    axes = fig.add_subplot(111)
    axes.grid()
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    # Plot fit_time vs score
    axes.grid()
    axes.plot(fit_times_mean, test_scores_mean, 'o-')
    axes.fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes.set_xlabel("fit_times", fontsize=14, fontweight='bold')
    axes.set_ylabel("Score", fontsize=14, fontweight='bold')
    axes.set_title(title, fontsize=14, fontweight='bold')
    if ylim is not None:
        axes.set_ylim(*ylim)

    if save_fig_name is not None:
        plt.savefig(save_fig_name)

    if show_plot:
        plt.tight_layout()
        plt.show()


def perform_validation_curve(estimator, X, y, param_name, param_range, scoring, n_jobs=-1, cv=None, verbose=1):
    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range, scoring=scoring, n_jobs=n_jobs, cv=cv, verbose=verbose)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    return train_scores_mean, train_scores_std, test_scores_mean, test_scores_std


def plot_validation_curve(train_scores_mean, train_scores_std, test_scores_mean, test_scores_std, param_name, param_range, title="Validation Curve", scoring='accuracy', ylim=None, tick_spacing=1, rotation='horizontal', figsize=default_figure_size, legend_loc='best'):
    """
    Reference: https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html#sphx-glr-auto-examples-model-selection-plot-validation-curve-py
    :param estimator:
    :param X:
    :param y:
    :param param_name:
    :param param_range:
    :param title:
    :param scoring:
    :param ylim:
    :param cv:
    :param n_jobs:
    :return:
    """

    plt.rcParams["figure.figsize"] = figsize
    plt.title(title, fontsize=title_fontsize, fontweight='bold')
    plt.xlabel(param_name, fontsize=fontsize, fontweight='bold')
    plt.ylabel("Score: {}".format(scoring), fontsize=fontsize, fontweight='bold')
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.margins(x=0, y=0)

    plt.plot(param_range, train_scores_mean, label="Training score",
                 color="darkorange", marker=".")
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=2)
    plt.plot(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", marker=".")
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=2)
    plt.xticks(param_range[::tick_spacing], rotation=rotation)
    plt.grid()
    plt.legend(fontsize=legend_fontsize, loc=legend_loc)
    plt.tight_layout()
    plt.show()


def full_learning_curve(estimator, X, y, title, scoring="accuracy", cv=None, n_jobs=-1, ylim=None, train_sizes=np.linspace(.1, 1.0, 10), save_fig_name=None, show_plot=True):
    train_sizes, train_scores, test_scores, fit_times = perform_learning_curve(estimator, X, y, scoring, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    plot_learning_curve(train_scores, test_scores, train_sizes, title,  ylim=ylim, save_fig_name=save_fig_name, show_plot=show_plot)

    return train_sizes, train_scores, test_scores, fit_times


def full_validaiton_curve(estimator, X, y, param_name, param_range, scoring, n_jobs=-1, cv=None, title="Validation Curve", ylim=None, tick_spacing=1):
    train_scores_mean, train_scores_std, test_scores_mean, test_scores_std = perform_validation_curve(estimator, X, y, param_name, param_range, scoring, n_jobs=n_jobs, cv=cv)
    try:
        plot_validation_curve(train_scores_mean, train_scores_std, test_scores_mean, test_scores_std, param_name, param_range, title=title, scoring=scoring, ylim=ylim, tick_spacing=tick_spacing)
    except:
        pass
    return train_scores_mean, train_scores_std, test_scores_mean, test_scores_std


if __name__ == "__main__":
    # Examples on wine dataset
    # Reference https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py
    X_train, X_test, y_train, y_test = get_wine_dataset()

    criterion = 'entropy'
    min_samples_split = 2
    min_samples_leaf = 1
    max_features = 'auto'
    ccp_alpha = 0.00
    max_depth = 5

    ylim=(0.45, 1.05)
    title = "Learning Curves Decision Tree criterion = {} min_samples_split = {} min_samples_leaf = {} max_feature = {} ccp_alpha = {}".format(criterion, min_samples_split, min_samples_leaf, max_features, ccp_alpha)
    title_formatted = "\n".join(wrap(title, 70))
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    estimator = tree.DecisionTreeClassifier(criterion=criterion, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features, ccp_alpha=ccp_alpha)

    train_sizes, train_scores, test_scores, fit_times = perform_learning_curve(estimator, X_train, y_train, scoring="balanced_accuracy", cv=cv, n_jobs=4)

    plot_learning_curve(train_scores, test_scores, train_sizes, title_formatted, ylim=ylim)
    plot_scalability_curve(fit_times, train_sizes, "Scalability")
    plot_performance_curve(test_scores, fit_times, "Performance of the model", ylim=ylim)

    ccp_alpha = np.arange(0.0, 0.01, 0.0001)

    criterion = ['gini', 'entropy']
    min_samples_split = [i for i in range(2, 50, 2)]
    min_samples_leaf = [i for i in range(1, 50, 2)]
    max_features = ['auto', 'sqrt', 'log2']
    max_depth = [i for i in range(1, 100, 1)]

    parameters = {'criterion': criterion, 'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf, 'max_features': max_features}

    train_scores_mean, train_scores_std, test_scores_mean, test_scores_std = perform_validation_curve(estimator, X_train, y_train, "max_depth", max_depth, 'balanced_accuracy', cv=cv)
    plot_validation_curve(train_scores_mean, train_scores_std, test_scores_mean, test_scores_std, "max_depth", max_depth, title="Validation Curve", scoring="balanced_accuracy", ylim=ylim, tick_spacing=5)


    # title = r"Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
    #
    # # SVC is more expensive so we do a lower number of CV iterations:
    # cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # estimator = SVC()
    # train_sizes, train_scores, test_scores, fit_times = perform_learning_curve(estimator, X_train, y_train, scoring="balanced_accuracy", cv=cv, n_jobs=4)
    # plot_learning_curve(train_scores, test_scores, train_sizes, title_formatted, ylim=ylim)
    # plot_scalability_curve(fit_times, train_sizes, "Scalability")
    # plot_performance_curve(test_scores, fit_times, "Performance of the model", ylim=ylim)