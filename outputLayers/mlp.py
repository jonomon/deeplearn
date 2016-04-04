from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.regularizers import l2, activity_l2
import numpy as np

def buidl_model(x_dim):
    model = Sequential()
    # Dense(64) is a fully-connected layer with 64 hidden units.
    # in the first layer, you must specify the expected input data shape:
    # here, 20-dimensional vectors.
    model.add(Dense(output_dim=128, input_dim=x_dim, init='uniform',
                    W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
    model.add(Activation('linear'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=64, input_dim=128, init='uniform',
                    W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
    model.add(Activation('linear'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=1, input_dim=64, init='uniform',
                    W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
    model.add(Activation("linear"))

    # model.add(Dense(output_dim=1, input_dim=x_dim, init='uniform'))
    # model.add(Activation("linear"))

    # model.add(Activation('softmax'))

    # sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error',
                  class_mode='binary',
                  optimizer='rmsprop')

    return model


if __name__ == '__main__':
    import dataHelper as data

    directory_image_features = "../imageFeatures/outputFeatures/"
    image_features_filename = "image_features.csv"

    directory_image_ratings = "../data/labels/"
    ratings_filename = "label.csv"
    x, y = data.get_data(
        feature_filename=directory_image_features + image_features_filename,
        ratings_filename=directory_image_ratings + ratings_filename)
    model = buidl_model(np.shape(x)[1])

    model.fit(x, y,
              nb_epoch=100,
              batch_size=8,
              validation_split=0.1,
              show_accuracy=True,
              verbose=1)
    # score = model.evaluate(x, y, batch_size=16)

    # print model.predict_proba(x, verbose=1)

    score = model.predict_proba(x, verbose=1)
    print y
    print score


