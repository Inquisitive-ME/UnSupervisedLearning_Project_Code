import pandas as pd
import numpy as np

from sklearn import svm, tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold, cross_val_score
import time
import IPython.display as disp

# TODO Figure out why adaboost hangs for faces
def kfolds_basic_test_all_classifiers(X_train, y_train, svm_type = 'rbf', scoring='accuracy'):
    KFold_Score = pd.DataFrame()
    time_taken = []
    classifiers = [svm_type + ' SVM', 'NeuralNetwork', 'DecisionTree', 'KNeighborsClassifier', 'RandomForestClassifier', 'AdaBoost']
    models = [svm.SVC(kernel=svm_type),
              MLPClassifier(random_state=0),
              tree.DecisionTreeClassifier(random_state=0),
              KNeighborsClassifier(n_neighbors=20),
              RandomForestClassifier(n_estimators=200, random_state=0),
              AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=2, random_state=1), random_state=0)
              ]

    start_time = time.time()
    increment_time = start_time
    j = 0
    for i in models:
        print("Running: {}".format(classifiers[j]))
        model = i
        cv = KFold(n_splits=5, random_state=0, shuffle=True)
        KFold_Score[classifiers[j]] = (cross_val_score(model, X_train, np.ravel(y_train), scoring=scoring, cv=cv))
        j = j+1
        print("Time Taken: {}".format(time.time() - increment_time))
        time_taken.append(time.time() - increment_time)
        increment_time = time.time()
    print("Final Time elapsed {}".format(time.time() - start_time))
    return KFold_Score, classifiers, time_taken


def print_kfolds_basic_test_results(KFold_Score, classifiers, y_train=None):

    mean = pd.DataFrame(KFold_Score.mean(), index= classifiers)
    KFold_Score_with_mean = pd.concat([KFold_Score,mean.T])
    KFold_Score_with_mean.index=['Fold 1','Fold 2','Fold 3','Fold 4','Fold 5','Mean']
    disp.display(KFold_Score_with_mean.T.sort_values(by=['Mean'], ascending = False))

    if y_train is not None:
        unique, frequency = np.unique(y_train, return_counts = True)
        print(frequency / len(y_train))