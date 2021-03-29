import os
import pandas as pd
import sklearn
import sklearn.model_selection
from sklearn import preprocessing

try:
    from data.faces.faces_generate_HOG_features import FACES_ZIP_DATA_FILE, FACES_DATA_FILE
except ModuleNotFoundError:
    from faces_generate_HOG_features import FACES_ZIP_DATA_FILE, FACES_DATA_FILE


def import_faces_dataset() -> pd.DataFrame:
    if os.path.exists(FACES_DATA_FILE):
        return pd.read_csv(FACES_DATA_FILE)
    else:
        df = pd.read_csv(FACES_ZIP_DATA_FILE, compression='gzip')
        return df.drop(["Unnamed: 0"], axis=1, errors='ignore')


def random_sample_instances(df, num_instances=3500):
    df = df.sample(frac=1, random_state=1).reset_index(drop=True)
    return df[:num_instances]

def normalize_dataset(train, test, MIN_MAX=True):
    if MIN_MAX:
        scaler = preprocessing.MinMaxScaler()
    else:
        scaler = preprocessing.StandardScaler()

    train_scaled = scaler.fit_transform(train.values)
    train = pd.DataFrame(train_scaled, columns=train.columns)

    test_scaled = scaler.transform(test.values)
    test = pd.DataFrame(test_scaled, columns=test.columns)

    return train, test

def get_faces_dataset_with_filenames(num_instances=3500):
    # For being able to view image
    faces = random_sample_instances(import_faces_dataset(), num_instances=num_instances)
    labels = faces["sex"]
    features = faces.drop(["age", "sex", "race"], axis=1)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(features, labels, test_size=0.2)
    filenames_train = X_train['filename']
    filenames_test = X_test['filename']
    X_train.drop("filename", axis=1, inplace=True)
    X_test.drop("filename", axis=1, inplace=True)

    return filenames_train, filenames_test, X_train, X_test, y_train, y_test

def get_faces_dataset_with_all_labels(num_instances=3500):
    # For being able to view image
    faces = random_sample_instances(import_faces_dataset(), num_instances=num_instances)
    labels = faces[["age", "sex", "race"]]
    features = faces.drop(["age", "sex", "race"], axis=1)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(features, labels, test_size=0.2)
    filenames_train = X_train['filename']
    filenames_test = X_test['filename']
    X_train.drop("filename", axis=1, inplace=True)
    X_test.drop("filename", axis=1, inplace=True)

    return filenames_train, filenames_test, X_train, X_test, y_train, y_test

def get_faces_dataset(num_instances=3500):
    faces = random_sample_instances(import_faces_dataset(), num_instances=num_instances)
    labels = faces["sex"].copy()
    features = faces.drop(["filename", "age", "sex", "race"], axis=1)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(features, labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    faces = import_faces_dataset()
    filenames_train, filenames_test, X_train, X_test, y_train, y_test = get_faces_dataset_with_all_labels()
    print(faces.head())
    print(faces.info())
    print(faces.columns)
