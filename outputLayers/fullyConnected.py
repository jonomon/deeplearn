import os
import numpy as np
import pandas as pd

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import BatchNormLayer
from lasagne.layers import prelu
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import rectify
from lasagne.objectives import squared_error

from nolearn.lasagne import NeuralNet, BatchIterator

from sklearn.preprocessing import StandardScaler

def get_data(feature_filename, ratings_filename):
    XidCol = "Unnamed: 0"
    yidCol = "Unnamed: 0"
    X = pd.read_csv(feature_filename)
    X[XidCol] = X[XidCol].apply(lambda x: os.path.splitext(os.path.basename(x))[0]).astype(int)
    y = pd.read_csv(ratings_filename)

    X = X.sort(columns = XidCol)
    y = y.sort(columns = yidCol)

    xiny = np.in1d(X[XidCol], y[yidCol])
    X = X.iloc[xiny]

    if np.sum(X[XidCol] == y[yidCol]) != X[XidCol].shape[0]:
        raise Exception("Labels not matched")
    
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
    
def build_net(dims):
    lr = 0.000001
    l_i = InputLayer(shape=(None, dims))
    #l_h = DenseLayer(l_i, num_units=5, nonlinearity=rectify)
    l_o = DenseLayer(l_i, num_units=1, nonlinearity=None)
    net = NeuralNet(l_o, max_epochs=2500,
                    #batch_iterator_train = BatchIterator(batch_size=32),
                    verbose=1, regression=True,
                    update_learning_rate=lr,
                    objective_loss_function=squared_error)
    return net

if __name__ == "__main__":
    directory_image_features = "../imageFeatures/outputFeatures/"
    image_features_filename = "image_features.csv"

    directory_image_ratings = "../data/labels/"
    ratings_filename = "label.csv"
    X, y = get_data(
        feature_filename=directory_image_features + image_features_filename,
        ratings_filename=directory_image_ratings + ratings_filename)
    net = build_net(X.shape[1])
    net.fit(X, y)
    import pdb; pdb.set_trace();    
    
