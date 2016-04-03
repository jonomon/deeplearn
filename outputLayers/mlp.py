from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np

def buidl_model(x_dim):
    model = Sequential()
    # Dense(64) is a fully-connected layer with 64 hidden units.
    # in the first layer, you must specify the expected input data shape:
    # here, 20-dimensional vectors.
    model.add(Dense(64, input_dim=x_dim, init='uniform'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(64, init='uniform'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=1, init='uniform'))
    # model.add(Activation('softmax'))

    # sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_absolute_error',
                  optimizer='rmsprop')

    return model


if __name__ == '__main__':

    import dataHelper as data

    x, y = data.load()
    model = buidl_model(np.shape(x)[1])

    model.fit(x, y,
              nb_epoch=20,
              batch_size=16,
              show_accuracy=True)
    # score = model.evaluate(x, y, batch_size=16)

    print model.predict_proba(x, verbose=1)


