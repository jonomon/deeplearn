from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.regularizers import l2, activity_l2
import numpy as np

import theano.tensor as T


def percent_mse(y_true, y_pred):
    return T.mean(T.sqr(y_pred - y_true), axis=-1)


def buidl_model(x_dim):
    model = Sequential()
    # Dense(64) is a fully-connected layer with 64 hidden units.
    # in the first layer, you must specify the expected input data shape:
    # here, 20-dimensional vectors.

    model.add(Dense(360, input_dim=x_dim, init='uniform', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop')

    return model


def evaluate(x, y, model):
    predictions = model.predict(x, verbose=0)
    predictions = np.round(predictions)

    dif = y - predictions
    n_errors = np.count_nonzero(dif)
    loss = float(n_errors) / len(predictions)

    return 1-loss


if __name__ == '__main__':
    # instructions:
    # GET IMAGE
    # - run data/Acc/data.py to download images
    # - run imageFeatures/image_resize.py to resize images to 244x244

    # GET DATA
    # - run pull_labels.py : prepare_data_for_classification to generate label files and image lists
    # - run image_features.py : prepare_train_test_features_244 to read in the image lists and generate features

    import dataHelper as data

    # eps = [2, 5, 10, 20, 40]
    eps = [10]

    train_features_filename = "../imageFeatures/outputFeatures/classification/features_train.csv"
    train_labels_filename = "../data/classification/train/labels.csv"

    test_features_filename = "../imageFeatures/outputFeatures/classification/features_test.csv"
    test_labels_filename = "../data/classification/test/labels.csv"

    xt, yt, x, y = data.get_data_for_classification(features_train=train_features_filename,
                                                    features_test=test_features_filename,
                                                    labels_train=train_labels_filename,
                                                    labels_test=test_labels_filename)

    for ep in eps:

        iteration = 10

        print "---"
        print "epoch={}".format(ep)
        while iteration > 0:

            x_train, y_train, x_val, y_val = data.validation_split(xt, yt, 0.1)

            model = buidl_model(np.shape(x_train)[1])
            model.fit(x_train, y_train,
                      nb_epoch=ep,
                      batch_size=16,
                      validation_split=0.0,
                      show_accuracy=True,
                      verbose=0)

            hit_train = evaluate(x_train, y_train, model)
            hit_val = evaluate(x_val, y_val, model)
            hit_test = evaluate(x, y, model)

            print "{}, {}, {}".format(hit_train, hit_val ,hit_test)
            iteration -= 1



