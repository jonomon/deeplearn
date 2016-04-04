import numpy as np
import pandas as pd
from os.path import basename
from os.path import splitext

features_file = '../imageFeatures/outputFeatures/image_features.csv'
labels_file = '../data/labels/label.csv'


def get_data(feature_filename, ratings_filename):
    XidCol = "Unnamed: 0"
    yidCol = "Unnamed: 0"
    X = pd.read_csv(feature_filename)
    y = pd.read_csv(ratings_filename)

    Xid = X[XidCol]
    Xid = Xid.apply(lambda x: splitext(basename(x))[0]).astype(int)
    yid = y[yidCol]

    xiny = np.in1d(Xid, yid)
    X = X.iloc[xiny]
    newXid = X[XidCol].apply(lambda x: splitext(basename(x))[0]).astype(int)
    if np.sum(newXid == yid) != newXid.shape[0]:
        print("Labels are not matched")
        return
    idx = np.random.permutation(len(X))
    X = X.iloc[idx]
    y = y.iloc[idx]

    X = X.drop("Unnamed: 0", axis=1)
    y = y.drop("Unnamed: 0", axis=1)
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)

    X = X.astype(np.float32)
    #y = y.reshape(-1, 1)
    y = y.astype(np.int32)

    return X.values, y.values


if __name__ == '__main__':
    directory_image_features = "../imageFeatures/outputFeatures/"
    image_features_filename = "image_features.csv"

    directory_image_ratings = "../data/labels/"
    ratings_filename = "label.csv"
    X, y = get_data(
        feature_filename=directory_image_features + image_features_filename,
        ratings_filename=directory_image_ratings + ratings_filename)
