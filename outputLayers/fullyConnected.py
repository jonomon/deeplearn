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

class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_params_from(self.best_weights)
            raise StopIteration()

def build_net(dims):
    # def precent_squared_error(a, b):
    #     return ((a - b) / a)**2
    #lr = 0.000001
    l_i = InputLayer(shape=(None, dims))
    l_do1 = DropoutLayer(l_i, p=0.8) #0.8
    l_h = DenseLayer(l_do1, num_units=20, nonlinearity=None) #20
    l_do2 = DropoutLayer(l_h, p=0.5)
    l_o = DenseLayer(l_do2, num_units=1, nonlinearity=None)
    net = NeuralNet(l_o, max_epochs=500,
                    batch_iterator_train = BatchIterator(batch_size=32),
                    verbose=1, regression=True,
                    update=adam,
                    objective_loss_function=squared_error,
                    on_epoch_finished=[EarlyStopping(patience=20)])
    return net

#Best 0.467700 valid
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
    y_predicted = net.predict(X)
    for i in range(X.shape[0]):
        print("{}, {}".format(y[i], y_predicted[i]))
    
