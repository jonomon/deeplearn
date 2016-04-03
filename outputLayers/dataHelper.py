import numpy as np

features_file = '../imageFeatures/outputFeatures/image_features.csv'
labels_file = '../data/labels/label.csv'


def load():
    features = np.genfromtxt(features_file, delimiter=',')

    # (n + 1) x (4096 + 1)
    # print np.shape(features)
    # print np.shape(features[1:, 1:])

    # drop headers and ids
    features = features[1:, 1:]

    labels = np.genfromtxt(labels_file)

    return features, labels

if __name__ == '__main__':
    x, y = load()