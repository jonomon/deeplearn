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


def prepare_train_test_features_244(in_dir, train_list, test_list, train_feature_path, test_feature_path):
    model = build_model()

    get_features_244_classification(model, in_dir, train_list, train_feature_path)
    get_features_244_classification(model, in_dir, test_list, test_feature_path)


def get_features_244_classification(model, in_dir, img_list_file, save_path):
    features = []

    df = pd.read_csv(img_list_file, header=0, index_col=0)
    im_ids = []
    i = 1
    for img_id in df.values:
        filename = in_dir + str(img_id[0]) + ".ppm"
        print("{0}/{1} Analysing {2}".format(i, len(df.values), filename))
        output = getOutput(model, filename)
        features.append(output)

        im_ids.append(img_id)
        i += 1
    f = pd.DataFrame(np.concatenate(features))
    f.to_csv(save_path, index=im_ids)


def get_features_rot():
    directory = "../data/rot/"

    model = build_model()

    features = []
    filenames = []
    input_images = glob.glob(directory+'/*_1.ppm')

    for i, filename in enumerate(input_images):
        print("{0}/{1} Analysing {2}".format(i, len(input_images), filename))
        output = getOutput(model, filename)
        filenames.append(filename)
        features.append(output)

    f = pd.DataFrame(np.concatenate(features), index=filenames)
    f.to_csv("outputFeatures/image_features_rot_1.csv")

    features = []
    filenames = []
    input_images = glob.glob(directory+'/*_2.ppm')

    for i, filename in enumerate(input_images):
        print("{0}/{1} Analysing {2}".format(i, len(input_images), filename))
        output = getOutput(model, filename)
        filenames.append(filename)
        features.append(output)
    f = pd.DataFrame(np.concatenate(features), index=filenames)
    f.to_csv("outputFeatures/image_features_rot_2.csv")


if __name__ == '__main__':
    # get_features_244()
    # get_features_rot()

    img_dir = "../data/resized_224/"
    train_img_list_file = "../data/classification/train/imgIds.csv"
    test_img_list_file = "../data/classification/test/imgIds.csv"
    save_path_train = "outputFeatures/classification/features_train.csv"
    save_path_test = "outputFeatures/classification/features_test.csv"

    prepare_train_test_features_244(in_dir=img_dir,
                                    train_list=train_img_list_file,
                                    test_list=test_img_list_file,
                                    train_feature_path=save_path_train,
                                    test_feature_path=save_path_test)