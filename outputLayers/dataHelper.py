import numpy as np
import pandas as pd
from os.path import basename
from os.path import splitext

from sklearn.preprocessing import normalize

features_file = '../imageFeatures/outputFeatures/image_features.csv'
labels_file = '../data/labels/label.csv'


def _load_data(feature_filename, ratings_filename, rot_flag=False):
    XidCol = "Unnamed: 0"
    yidCol = "Unnamed: 0"
    X = pd.read_csv(feature_filename)
    if rot_flag:
        X[XidCol] = X[XidCol].apply(lambda x: splitext(basename(x))[0][0:-2]).astype(int)
    else:
        X[XidCol] = X[XidCol].apply(lambda x: splitext(basename(x))[0]).astype(int)
    y = pd.read_csv(ratings_filename)

    X = X.sort_values(by=XidCol)
    y = y.sort_values(by=yidCol)

    xiny = np.in1d(X[XidCol], y[yidCol])
    X = X.iloc[xiny]

    if np.sum(X[XidCol] == y[yidCol]) != X[XidCol].shape[0]:
        raise Exception("Labels not matched")

    return X, y

def get_data(feature_filename, ratings_filename):
    X, y = _load_data(feature_filename, ratings_filename)

    np.random.seed(616)
    idx = np.random.permutation(len(X))
    X = X.iloc[idx]
    y = y.iloc[idx]

    X = X.drop("Unnamed: 0", axis=1)
    y = y.drop("Unnamed: 0", axis=1)
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)

    X = X.astype(np.float32)
    #y = y.reshape(-1, 1)
    y = y.astype(np.float32)

    return X.values, y.values


def get_data_for_classification(features_train, labels_train, features_test, labels_test):
    x_train = pd.read_csv(features_train, header=0, index_col=0)
    y_train = pd.read_csv(labels_train, header=0, index_col=0)

    x_test = pd.read_csv(features_test, header=0, index_col=0)
    y_test = pd.read_csv(labels_test, header=0, index_col=0)

    return x_train.values, y_train.values, x_test.values, y_test.values


def get_data_rot(global_features, rot_1, rot_2, ratings_filename):
    g, y = _load_data(global_features, ratings_filename)
    r_1, _ = _load_data(rot_1, ratings_filename, rot_flag=True)
    r_2, _ = _load_data(rot_2, ratings_filename, rot_flag=True)

    X = pd.concat([g, r_1, r_2], axis=1)
    # X = pd.concat([r_1, r_2], axis=1)

    np.random.seed(616)
    idx = np.random.permutation(len(X))
    X = X.iloc[idx]
    y = y.iloc[idx]

    X = X.drop("Unnamed: 0", axis=1)
    y = y.drop("Unnamed: 0", axis=1)

    X = X.astype(np.float32)
    y = y.astype(np.float32)

    return X.values, y.values


def generate_sample_weights(labels):
    hist, _ = np.histogram(labels, bins=[1,2,3,4,5,6,7,8,9,10])

    labels_as_bin = np.round(labels).astype("int")
    labels_as_bin = np.reshape(labels_as_bin, (1, len(labels_as_bin)))

    sample_weights = float(np.sum(hist))/hist[labels_as_bin-1]
    # print np.shape(sample_weights)
    # print np.shape(labels)

    sample_weights = np.reshape(sample_weights, np.shape(sample_weights)[1])
    return sample_weights


def validation_split(x, y, ratio):
    # print "shape of x = " + str(np.shape(x)) # (4989, 4096)
    # print "shape of y = " + str(np.shape(y)) # (4989, 1)

    size_val = np.round(np.shape(y)[0] * ratio)
    # print size_val

    np.random.seed()
    ind_v = np.random.choice(range(0, np.shape(y)[0]), replace=False, size=size_val)
    ind_t = np.array([i for i in range(0, np.shape(y)[0]) if i not in ind_v])

    x_v = x[ind_v]
    y_v = y[ind_v]

    x_t = x[ind_t]
    y_t = y[ind_t]

    return x_t, y_t, x_v, y_v


if __name__ == '__main__':
    directory_image_features = "../imageFeatures/outputFeatures/"
    image_features_filename = "image_features.csv"
    rot_1_features_filename = "image_features_rot_1.csv"
    rot_2_features_filename = "image_features_rot_2.csv"

    directory_image_ratings = "../data/labels/"
    ratings_filename = "label.csv"

    train_features_filename = "../imageFeatures/outputFeatures/classification/features_train.csv"
    train_labels_filename = "../data/classification/train/labels.csv"

    test_features_filename = "../imageFeatures/outputFeatures/classification/features_test.csv"
    test_labels_filename = "../data/classification/test/labels.csv"

    xt, yt, x, y = get_data_for_classification(features_train=train_features_filename,
                                               features_test=test_features_filename,
                                               labels_train=train_labels_filename,
                                               labels_test=test_labels_filename)

    # X, y = get_data(
    #     feature_filename=directory_image_features + image_features_filename,
    #     ratings_filename=directory_image_ratings + ratings_filename)

    # X, y = get_data_rot(global_features=directory_image_features + image_features_filename,
    #              rot_1=directory_image_features + rot_1_features_filename,
    #              rot_2=directory_image_features + rot_2_features_filename,
    #              ratings_filename=directory_image_ratings + ratings_filename)

    # print y
    # print np.shape(y)

    # generate_sample_weights(y)

    # validation_split(X, y, 0.1)
    # print np.mean(y)
    # print np.std(y)
