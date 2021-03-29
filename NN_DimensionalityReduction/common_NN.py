from sklearn import svm
from sklearn.model_selection import GridSearchCV
import joblib
import os
from sklearn.model_selection import ShuffleSplit
from data.wine.Wine_Quality_Data import get_wine_dataset
import TrainingCurves
from sklearn.neural_network import MLPClassifier


class my_MLPClassifier(MLPClassifier):
    def __init__(self, num_hidden_layers, num_nodes_per_layer, solver='adam', learning_rate='constant', learning_rate_init=0.001, max_iter=200, batch_size='auto'):
        hidden_layer_sizes = []
        for i in range(num_hidden_layers):
            hidden_layer_sizes.append(num_nodes_per_layer)
        self.num_hidden_layers = num_hidden_layers
        self.num_nodes_per_layer = num_nodes_per_layer
        print("num_hidden_layers: ", self.num_hidden_layers, "num_nodes_per_layer: ", self.num_nodes_per_layer)

        super().__init__(hidden_layer_sizes=hidden_layer_sizes, solver=solver, learning_rate=learning_rate,
                         learning_rate_init=learning_rate_init, max_iter=max_iter, batch_size=batch_size,
                         verbose=10, random_state=1)


def perform_grid_search(parameters, X_train, y_train, scoring, GS_FILE_NAME_PREFIX, n_jobs=-1,  default_parameters={}):
    GS_FILE_NAME = GS_FILE_NAME_PREFIX
    for key, value in parameters.items():
        GS_FILE_NAME += ("_" + key)
        try:
            GS_FILE_NAME += ("_" + str(value[0]) + "-" + str(value[-1]))
        except ValueError:
            GS_FILE_NAME += ("_" + value[0] + "-" + value[-1])
    GS_FILE_NAME += ".pickle"

    if os.path.exists(GS_FILE_NAME):
        print("WARNING: file ", GS_FILE_NAME, " already exists")
        print("NOT performing Grid Search")
        gs = joblib.load(GS_FILE_NAME)
    else:
        print("Grid Search Will be Saved to ", GS_FILE_NAME)

        gs = GridSearchCV(MLPClassifier(**default_parameters), parameters, scoring=scoring, return_train_score=True,
                          verbose=10, n_jobs=n_jobs)
        gs.fit(X_train, y_train)

        joblib.dump(gs, GS_FILE_NAME)
        print("Saved ", GS_FILE_NAME)
    return gs


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_wine_dataset()
    num_features = X_train.shape[1]

    number_nodes = [i for i in range(1, 30, 2)]
    number_layers = [i for i in range(1, 20, 2)]

    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

    mlp = MLPClassifier(hidden_layer_sizes=[100, 100], learning_rate='adaptive',
                        solver='adam', verbose=10, random_state=1,
                        learning_rate_init=.01, max_iter=300, batch_size=32)

    mlp = my_MLPClassifier(num_hidden_layers=2, num_nodes_per_layer=50, learning_rate='adaptive', solver='sgd',
                           learning_rate_init=.01, max_iter=300, batch_size=32)

    train_sizes, train_scores, test_scores, fit_times = TrainingCurves.perform_learning_curve(mlp, X_train, y_train,
                                                                                              scoring="balanced_accuracy",
                                                                                              cv=cv, n_jobs=4)
    TrainingCurves.plot_learning_curve(train_scores, test_scores, train_sizes, "test")