import os
import numpy as np
import pandas as pd

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import rectify, softmax
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
    l_i = InputLayer(shape=(None, dims))
    l_do1 = DropoutLayer(l_i, p=0.80) #0.80
    l_h = DenseLayer(l_do1, num_units=20, nonlinearity=None) #20
    l_do2 = DropoutLayer(l_h, p=0.5) #0.5
    l_o = DenseLayer(l_do2, num_units=2, nonlinearity=softmax)
    net = NeuralNet(l_o, max_epochs=100,
                    batch_iterator_train = BatchIterator(batch_size=128), #128
                    verbose=1, regression=False,
                    update=adam,
                    on_epoch_finished=[EarlyStopping(patience=5)]) #5
    return net

#Best val = 0.530831, test accuracy = 0.768
if __name__ == "__main__":
    import dataHelper as data

    train_features_filename = "../imageFeatures/outputFeatures/classification/features_train.csv"
    train_labels_filename = "../data/classification/train/labels.csv"

    test_features_filename = "../imageFeatures/outputFeatures/classification/features_test.csv"
    test_labels_filename = "../data/classification/test/labels.csv"
    np.random.seed(616)
    xt, yt, x, y = data.get_data_for_classification(features_train=train_features_filename,
                                                    features_test=test_features_filename,
                                                    labels_train=train_labels_filename,
                                                    labels_test=test_labels_filename)
    y = y.astype(np.int32).reshape(-1,)
    yt = yt.astype(np.int32).reshape(-1,)
    net = build_net(xt.shape[1])
    net.fit(xt, yt)
    y_predicted = net.predict(x)
    test_accuracy = np.mean(y_predicted == y)
    print("test accuracy {}".format(test_accuracy))
    import pdb; pdb.set_trace();

    
