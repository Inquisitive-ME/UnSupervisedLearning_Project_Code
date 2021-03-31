from data.faces.faces_data import get_faces_dataset, get_faces_dataset_with_all_labels
from data.generated.generated_data import get_noisy_nonlinear_with_non_noisy_labels
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn import random_projection
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree

import pandas as pd


filenames_train, filenames_test, X_train_faces, X_test_faces, y_train_faces, y_test_faces =\
get_faces_dataset_with_all_labels()

X_train_gnnl, X_test_gnnl, y_train_gnnl, y_test_gnnl, y_test_non_noisy_gnnl =\
get_noisy_nonlinear_with_non_noisy_labels()


def get_faces_pca():
    pca_best_num_components = 98
    print("Running PCA for {} components".format(pca_best_num_components))
    pca = PCA(n_components=pca_best_num_components)
    pca.fit(X_train_faces)
    X_train_transformed = pca.transform(X_train_faces)
    X_test_transformed = pca.transform(X_test_faces)
    return X_train_transformed, X_test_transformed


def get_gnnl_pca():
    pca_best_num_components = 6
    print("Running PCA for {} components".format(pca_best_num_components))
    pca = PCA(n_components=pca_best_num_components)
    pca.fit(X_train_gnnl)
    X_train_transformed = pca.transform(X_train_gnnl)
    X_test_transformed = pca.transform(X_test_gnnl)
    return X_train_transformed, X_test_transformed


def get_faces_ica():
    best_ica_components = X_train_faces.shape[1]
    ica = FastICA(random_state=42, max_iter=500)
    print("Running ICA for {} components".format(best_ica_components))
    ica.set_params(n_components=best_ica_components)
    ica.fit(X_train_faces)
    X_train_faces_ica = ica.transform(X_train_faces)
    X_test_faces_ica = ica.transform(X_test_faces)
    X_train_faces_ica_df = pd.DataFrame(X_train_faces_ica)
    ica_kurt = X_train_faces_ica_df.kurt(axis=0)

    return X_train_faces_ica[:, ica_kurt > 6], X_test_faces_ica[:, ica_kurt > 6]


def get_gnnl_ica():
    best_ica_components = X_train_gnnl.shape[1]
    ica = FastICA(random_state=42, max_iter=500)
    print("Running ICA for {} components".format(best_ica_components))
    ica.set_params(n_components=best_ica_components)
    ica.fit(X_train_gnnl)
    X_train_gnnl_ica = ica.transform(X_train_gnnl)
    X_test_gnnl_ica = ica.transform(X_test_gnnl)
    X_train_gnnl_ica_df = pd.DataFrame(X_train_gnnl_ica)
    ica_kurt = X_train_gnnl_ica_df.kurt(axis=0)
    return X_train_gnnl_ica[:, ica_kurt > 200], X_test_gnnl_ica[:, ica_kurt > 200]


def get_faces_random_projection():
    transformer = random_projection.GaussianRandomProjection(n_components=X_train_faces.shape[1]//2, random_state=9)
    transformer.fit(X_train_faces)
    return transformer.transform(X_train_faces), transformer.transform(X_test_faces)


def get_gnnl_random_projection():
    transformer = random_projection.GaussianRandomProjection(n_components=X_train_gnnl.shape[1]//2, random_state=9)
    transformer.fit(X_train_gnnl)
    return transformer.transform(X_train_gnnl), transformer.transform(X_test_gnnl)


def get_faces_boosted_best_features():
    base_estimator = tree.DecisionTreeClassifier(ccp_alpha=0.002, max_depth=1)
    final_params = {'base_estimator': base_estimator, 'learning_rate': 0.22, 'n_estimators': 15}
    clf = AdaBoostClassifier(**final_params)
    clf.fit(X_train_faces, y_train_faces['sex'])
    return X_train_faces[X_train_faces.columns[clf.feature_importances_>0]], X_test_faces[X_train_faces.columns[clf.feature_importances_>0]]


def get_gnnl_boosted_best_features():
    base_estimator = tree.DecisionTreeClassifier(ccp_alpha=0.001, max_depth=3)
    final_params = {'base_estimator': base_estimator, 'learning_rate': 0.1, 'n_estimators': 9}
    clf = AdaBoostClassifier(**final_params)
    clf.fit(X_train_gnnl, y_train_gnnl)
    return X_train_gnnl[:, clf.feature_importances_>0.06], X_test_gnnl[:, clf.feature_importances_>0.06]

