import numpy as np
import pandas as pd

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import BatchNormLayer
from lasagne.layers import prelu
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import rectify
from nolearn.lasagne import NeuralNet, BatchIterator
from lasagne.updates import nesterov_momentum, adam

from sklearn.preprocessing import StandardScaler

def get_data(feature_filename, ratings_filename):
    X = pd.read_csv(feature_filename)
    y = np.random.rand(2500) * 100
    idx = np.random.permutation(len(X))
    X = X.iloc[idx]
    y = y[idx]

    X = X.drop("Unnamed: 0", axis=1)
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    
    X = X.astype(np.float32)
    y = y.reshape(-1, 1)
    y = y.astype(np.int32)

    return X.values, y
    
def build_net(dims):
    lr = 0.01
    l_i = InputLayer(shape=(None, dims))
    l_h = DenseLayer(l_i, num_units=300, nonlinearity=rectify)
    l_o = DenseLayer(l_h, num_units=1, nonlinearity=None)
    net = NeuralNet(l_o, max_epochs=100,
                    batch_iterator_train = BatchIterator(batch_size=32),
                    verbose=1, regression=True,
                    update_learning_rate=lr)
    return net

if __name__ == "__main__":
    directory_image_features = "../imageFeatures/outputFeatures/"
    image_features_filename = "image_features.csv"

    directory_image_ratings = ""
    ratings_filename = ""
    X, y = get_data(
        feature_filename=directory_image_features + image_features_filename,
        ratings_filename=directory_image_ratings + ratings_filename)

    net = build_net(X.shape[1])
    net.fit(X, y)
    import pdb; pdb.set_trace();    
