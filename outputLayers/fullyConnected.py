import os
import numpy as np
import pandas as pd

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import rectify
from lasagne.objectives import squared_error
from lasagne.updates import adam
from nolearn.lasagne import NeuralNet, BatchIterator

from sklearn.preprocessing import StandardScaler
    
def build_net(dims):
    #lr = 0.000001
    l_i = InputLayer(shape=(None, dims))
    #l_h = DenseLayer(l_i, num_units=200, nonlinearity=None)
    l_do = DropoutLayer(l_i, p=0.5)
    l_o = DenseLayer(l_do, num_units=1, nonlinearity=None)
    net = NeuralNet(l_o, max_epochs=2500,
                    batch_iterator_train = BatchIterator(batch_size=32),
                    verbose=1, regression=True,
                    #update_learning_rate=lr,
                    update=adam,
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
    net = build_net(X.shape[1])
    net.fit(X, y)
    import pdb; pdb.set_trace();    
    
