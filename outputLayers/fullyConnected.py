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
    
def build_net(dims):
    lr = 0.01
    l_i = InputLayer(shape=(None, dims))
    l_h = DenseLayer(l_i, num_units=5, nonlinearity=rectify)
    l_o = DenseLayer(l_h, num_units=1, nonlinearity=None)
    net = NeuralNet(l_o, max_epochs=100,
                    batch_iterator_train = BatchIterator(batch_size=32),
                    verbose=1, regression=True,
                    update_learning_rate=lr,
                    objective_loss_function=squared_error)
    return net

if __name__ == "__main__":
    import dataHelper as data

    directory_image_features = "../imageFeatures/outputFeatures/"
    image_features_filename = "image_features.csv"

    directory_image_ratings = "../data/labels/"
    ratings_filename = "label.csv"
    X, y = data.get_data(
        feature_filename=directory_image_features + image_features_filename,
        ratings_filename=directory_image_ratings + ratings_filename)
    # net = build_net(X.shape[1])
    # net.fit(X, y)
    
    from sklearn.linear_model import LinearRegression
    from sklearn.cross_validation import cross_val_predict
    clf = LinearRegression()
    predicted = cross_val_predict(clf, X, y, cv=10)
    
    import pdb; pdb.set_trace();    
