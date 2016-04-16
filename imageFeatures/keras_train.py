## change directory variable to the directory of the images
## output features will be in the outputFeatures folder named image_features.csv

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import glob
import os
import h5py

batch_size = 32
nb_classes = 9
nb_epoch = 200
img_channels = 3
img_rows = 224
img_cols = 224

def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,img_rows,img_cols)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(1000, activation='softmax'))

    assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
    f = h5py.File(weights_path)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers): # ignore the last layer
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()

    return model

def getOutput(model, filename):
    from scipy.misc import imread, imresize, imsave
    img = imresize(imread(filename), (img_rows, img_cols))
    img = img.transpose((2, 0, 1)).astype('float64')
    img[:, :, 0] -= 103.939
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 123.68
    img = np.expand_dims(img, axis=0)
    out = model.predict(img)
    return out

def getScore(filename, ava):
    int_filename = int(filename)
    if int_filename not in ava.index:
        return None

    line = ava.loc[int_filename]
    scores = line[0:10].values
    # weighted_score = 0.0
    # for i, score in enumerate(scores):
    #     i = float(i)
    #     weighted_score += score * (i + 1)
    # average_weighted_score = weighted_score / np.sum(scores)

    largest_n = 2;
    indices = np.argpartition(scores, -largest_n)[-largest_n:]
    vals = scores[indices]
    bins = indices + 1

    average_weighted_score = float(np.sum(vals * bins))/np.sum(vals)
    return average_weighted_score

def parseFilename(filename):
    # remove ROT identifiers
    basename = os.path.basename(filename).split('.')[0]
    if "_" in basename:
        return basename.split('_')[0]
    else:
        return basename

def saveOutput(features, metadata, filecount):
    data = []
    # make output file
    for k in features:
        if k not in metadata:
            raise Exception('No metadata found for {}'.format(k))
        feature = np.hstack(features[k]).astype('float32').reshape(-1)
        metadata_i = metadata[k]
        metadata_array = np.array(metadata_i)
        data_i = np.append(metadata_array, feature)
        data.append(data_i)

    data = np.vstack(data)
    f = pd.DataFrame(data)
    f.columns = np.append(np.array(["ID", "weighted_score"]), np.arange(0, data.shape[1] - 2))
    f.to_csv("outputFeatures/image_features{}.csv".format(filecount))

if __name__ == "__main__":
    ava_file = "../dataAcc/AVA_dataset/AVA.txt"
    ava = pd.read_csv(ava_file, sep=" ", header=None).set_index(1).drop(0, axis=1)

    # model = VGG_16('weights/vgg16_weights.h5')
    # sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(optimizer=sgd, loss='categorical_crossentropy')
    
    #directory = "../data/resized_224/" #input directory
    #directory = "../data/original/"
    directory = "../data/rot/"
    metadata = {}
    features = {}
    input_images = glob.glob(directory+'/*ppm')
    split_file_every = 20000 # first number must be divisible by 4 (or how many ROT created)
    file_count = 0
    for i, filename in enumerate(input_images):
        print("{0}/{1} Analysing {2}".format(i + 1, len(input_images), filename))
        #output = getOutput(model, filename)
        output = np.zeros((1, 4096))
        parsedFilename = parseFilename(filename)
        score = getScore(parsedFilename, ava)
        if score == None:
            print("No score found for {}".format(filename))
            continue

        metadata_i = (parsedFilename, score)
        metadata[parsedFilename] = metadata_i
        
        if parsedFilename not in features:
            features[parsedFilename] = []
        features[parsedFilename].append(output)
        if (i + 1) % split_file_every == 0:
            file_count += 1
            saveOutput(features, metadata, file_count)
            features = {}
            metadata = {}
    file_count += 1
    saveOutput(features, metadata, file_count)

