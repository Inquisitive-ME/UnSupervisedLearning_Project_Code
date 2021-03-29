import numpy as np
from sklearn.datasets import make_classification
import sklearn
import sklearn.model_selection
import random

def generate_noisy_nonlinear_dataset():
    n_features = 10
    n_informative = 10
    n_redundant = 0
    n_clusters_per_class = 4
    hypercube = True
    random_state = 7
    x, y = make_classification(n_samples=3500, n_features=n_features, n_redundant=n_redundant,
                               n_informative=n_informative, n_classes=2, n_clusters_per_class=n_clusters_per_class,
                               flip_y=0.3, hypercube=hypercube, random_state=random_state, class_sep=4)
    x = np.apply_along_axis(lambda x: 2 ** x, 0, x)
    x = x / x.max(axis=0)
    return x,y


def test_add_noise(labels, percent_to_flip, random_state):
    random.seed(random_state)
    noise_added_labels = []
    for i in labels:
        if random.random() < percent_to_flip:
            if i == 1:
                noise_added_labels.append(0)
            if i == 0:
                noise_added_labels.append(1)
            # noise_added_labels.append(np.random.choice(np.unique(labels)))
        else:
            noise_added_labels.append(i)
    return noise_added_labels


def generate_noisy_nonlinear_dataset_with_non_noisy_labels():
    n_features = 10
    n_informative = 10
    n_redundant = 0
    n_clusters_per_class = 4
    hypercube = True
    random_state = 7
    x, y = make_classification(n_samples=3500, n_features=n_features, n_redundant=n_redundant,
                               n_informative=n_informative, n_classes=2, n_clusters_per_class=n_clusters_per_class,
                               flip_y=0.0, hypercube=hypercube, random_state=random_state, class_sep=2, shuffle=True)
    x = np.apply_along_axis(lambda x: 2 ** x, 0, x)

    return x, test_add_noise(y, 0.2, random_state), y


def get_noisy_nonlinear():
    X2, Y2 = generate_noisy_nonlinear_dataset()

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X2, Y2, test_size=0.2, random_state=42)

    X_train = X_train / X_train.max(axis=0)
    X_test = X_test / X_test.max(axis=0)

    return X_train, X_test, y_train, y_test

def get_noisy_nonlinear_with_non_noisy_labels():
    X2, Y_noise, Y2 = generate_noisy_nonlinear_dataset_with_non_noisy_labels()

    X_train, X_test, y_train, y_test, indices_train, indices_test = sklearn.model_selection.train_test_split(X2, Y_noise, [i for i in range(len(Y_noise))], test_size=0.2, random_state=42)

    X_train = X_train / X_train.max(axis=0)
    X_test = X_test / X_test.max(axis=0)

    return X_train, X_test, y_train, y_test, Y2[indices_test]

def generate_large_num_features_dataset():
    n_features = 200
    n_informative = 50
    n_redundant = 50
    n_clusters_per_class = 4
    hypercube = True
    X2, Y2 = make_classification(n_samples=3500, n_features=n_features, n_redundant=n_redundant,
                                 n_informative=n_informative, n_classes=2, n_clusters_per_class=n_clusters_per_class,
                                 flip_y=0.0, hypercube=hypercube, class_sep=1)
    return X2, Y2

def get_large_num_features_dataset():
    X2, Y2 = generate_large_num_features_dataset()

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X2, Y2, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    get_noisy_nonlinear_with_non_noisy_labels()