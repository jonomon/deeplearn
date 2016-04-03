import vgg
from keras.optimizers import SGD
import glob
import numpy as np
import pandas as pd

weights_file = 'weights/vgg16_weights.h5'
img_rows = 224
img_cols = 224

def build_model():
    model = vgg.VGG_16(weights_file)
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    return model


def getOutput(model, filename):
    from scipy.misc import imread, imresize
    img = imresize(imread(filename), (img_rows, img_cols))
    img = img.transpose((2, 0, 1)).astype('float64')

    # mean pixel values
    img[:, :, 0] -= 103.939
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 123.68
    img = np.expand_dims(img, axis=0)
    out = model.predict(img)

    return out


def get_features_244():
    directory = "../data/resized_224/"

    model = build_model()

    features = []
    filenames = []
    input_images = glob.glob(directory+'/*ppm')

    for i, filename in enumerate(input_images):
        print("{0}/{1} Analysing {2}".format(i, len(input_images), filename))
        output = getOutput(model, filename)
        filenames.append(filename)
        features.append(output)
    f = pd.DataFrame(np.concatenate(features), index=filenames)
    f.to_csv("outputFeatures/image_features.csv")

if __name__ == '__main__':
    get_features_244()