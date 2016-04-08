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

    model.add(Dense(480, input_dim=x_dim, init='lecun_uniform'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(20, init='lecun_uniform'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(1, init='lecun_uniform'))
    model.add(Activation('linear'))

    model.compile(loss="mean_squared_error",
                  optimizer='adam')

    return model


def eval(model, x_val, y_val):
    score = model.predict_proba(x_val, verbose=0)

    count = 0;
    count_extreme = 0;
    count_extreme_correct = 0;

    # taken from mean and std of dataset

    cutoff = 1.67814

    upper = 5.5
    lower = 4.5

    for i in range(len(score)):
        # print "{}, {}".format(y_val[i], score[i])
        if y_val[i] < lower or y_val[i] > upper:
            count_extreme += 1
        if abs(y_val[i] - score[i]) < cutoff:
            count += 1
            if y_val[i] < lower or y_val[i] > upper:
                count_extreme_correct += 1

    # print "{} % images within +/-{}".format(100 * float(count)/len(score), cutoff)
    # print "{} % extreme images within +/-{}".format(100 * float(count_extreme_correct)/count_extreme, cutoff)
    print "{}, {}".format(100 * float(count)/len(score), 100 * float(count_extreme_correct)/count_extreme)


if __name__ == '__main__':
    import dataHelper as data

    directory_image_features = "../imageFeatures/outputFeatures/"
    image_features_filename = "image_features.csv"

    directory_image_ratings = "../data/labels/"
    ratings_filename = "label.csv"

    # eps = [10, 20, 40, 60, 80]
    eps = [20]

    for ep in eps:

        iteration = 10

        print "epoch, {}".format(ep)
        while iteration > 0:
            x, y = data.get_data(
                feature_filename=directory_image_features + image_features_filename,
                ratings_filename=directory_image_ratings + ratings_filename)

            x_train, y_train, x_val, y_val = data.validation_split(x, y, 0.1)

            model = buidl_model(np.shape(x_train)[1])
            model.fit(x_train, y_train,
                      nb_epoch=ep,
                      batch_size=32,
                      validation_split=0.0,
                      show_accuracy=True,
                      verbose=0)
            # score = model.evaluate(x, y, batch_size=16)

            # print model.predict_proba(x, verbose=1)
            eval(model, x_val, y_val)
            eval(model, x, y)
            print "\n"
            iteration -= 1

