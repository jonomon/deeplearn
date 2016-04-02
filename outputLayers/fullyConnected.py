import numpy as np
import pandas as pd

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import BatchNormLayer
from lasagne.layers import prelu
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import rectify
from nolearn.lasagne import NeuralNet
from lasagne.updates import nesterov_momentum, adam

from sklearn.preprocessing import StandardScaler

def get_data(filename):
    X = pd.read_csv(filename)
    y = np.random.rand(2500) * 100
    idx = np.random.permutation(len(X))
    X = X.iloc[idx]
    y = y[idx]

    X = X.drop("Unnamed: 0", axis=1)
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    
    X = X.astype(np.float32)
    y = y.astype(np.int32)

    return X, y
    
def build_net(dims):
    lr = 0.01
    l_i = InputLayer(shape=(None, dims))
    l_h1 = DenseLayer(l_i, 300)
    l_do1 = DropoutLayer(l_h1, p=0.05)
    l_o = DenseLayer(l_do1, num_units=1, nonlinearity=rectify)
    net = NeuralNet(l_o, max_epochs=3000,
                    verbose=1, regression=False,
                    update_learning_rate=lr)
    return net

if __name__ == "__main__":
    directory = "../imageFeatures/outputFeatures/"
    filename = "image_features.csv"
    X, y = get_data(directory + filename)
    net = build_net(X.shape[1])
    net.fit(X, y)
