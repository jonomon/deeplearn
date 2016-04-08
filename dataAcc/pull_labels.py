import numpy as np
import pandas as pd
from os.path import basename
from os.path import splitext
import glob

ava_file = "AVA_dataset/AVA.txt"

image_features_file = "../imageFeatures/outputFeatures/image_features.csv"

train_test_image_dir = "AVA_dataset/aesthetics_image_lists/"
output_path = "../data/labels/label.csv"


def error(str):
    raise StandardError(str);
    exit(-1)


# taking average as score
def get_score(line):
    scores = np.array([int(i) for i in line[2:12]])
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


def get_label(line, sigma):
    score = get_score(line)

    label = -1
    if score > 5 + sigma:
        label = 1
    elif score < 5 - sigma:
        label = 0

    return label


def get_labels(avadict, img_ids, delta):
    labels = []
    ids = []
    for img_id in img_ids:
        line = avadict[str(img_id)]
        label = get_label(line, delta)

        # drop invalid labels
        if label == -1:
            continue
        else:
            # build both score list and image id list
            labels.append(label)  # int
            ids.append(img_id)  # int
    return labels, ids


# get test and train image ids for food pictures
def get_ids():
    #filenames = glob.glob(train_test_image_dir+'*.jpgl')
    filenames = glob.glob(train_test_image_dir + '*fooddrink*.jpgl')
    ids = []

    for filename in filenames:
        with open(filename) as infile:
            for line in infile:
                ids.append(line.strip())
    return ids


# build the ava.txt file into a dictionary, indexed by image id string
def build_ava_dictionary():
    ava_dict = {}

    with open(ava_file) as infile:
        for line in infile:
            line = line.split()

            img_id = line[1]
            if img_id in ava_dict:
                error("Abort: duplicate image found! possible data corruption!")
            else:
                ava_dict[img_id] = line
    return ava_dict


def build_image_list_and_lables(in_dir, out_dir, sigma):
    # build ava dict
    ava_dict = build_ava_dictionary()

    # take local images that are already downloaded and get score
    filenames = glob.glob(in_dir + '*.ppm')

    labels = []
    img_ids = []
    for name in filenames:
        img_id = splitext(basename(name))[0]
        line = ava_dict[img_id]
        label = get_label(line, sigma)

        # drop invalid labels
        if label == -1:
            continue
        else:
            # build both score list and image id list
            labels.append(label)  # int
            img_ids.append(int(img_id))  # int

    # write to file
    df = pd.DataFrame(np.asarray(img_ids))
    df.to_csv(out_dir + "imgIds.csv")
    df = pd.DataFrame(np.asarray(labels))
    df.to_csv(out_dir + "labels.csv")


# save both training and test images labels for ALL images
def get_all_labels():
    test_ids, train_ids = get_ids()

    with open(ava_file) as infile:
        output = open(output_path, "w")

        results = []

        for line in infile:
            line = line.split()

            img_id = int(line[1])

            if img_id in test_ids or img_id in train_ids:
                score = get_score(line)
                results.append([img_id, score])

        np.savetxt(output, np.asarray(results), fmt="%d, %f")


# look at the \imageFeatures\outputFeatures\image_features.csv and only creates label files for images
# present in the image_features.csv file
def get_labels_for_images_with_features():
    # sanity check
    import os.path
    if not os.path.isfile(image_features_file):
        error("Abort: image features have not been processed! Make sure \imageFeatures\outputFeatures\image_features.csv \
         is present! ")

    print "reading \imageFeatures\outputFeatures\image_features.csv..."

    img_ids = []

    with open(image_features_file) as infile:
        for line in infile:
            image_name = line.split(",")[0]

            if image_name is not "":
                img_ids.append(splitext(basename(image_name))[0])

    ava_dict = build_ava_dictionary()

    # train & test images combined!
    labels = []
    ids = []

    for img_id in img_ids:
        if img_id in ava_dict:
            score = get_score(ava_dict[img_id])

            labels.append(score)
            ids.append(img_id)
        else:
            print("Abort: image {} not found in AVA.txt, possible data corruption!".format(
                img_id))

    from scipy.interpolate import interp1d

    min_label = np.min(labels)
    max_label = np.max(labels)

    print min_label
    print max_label
    m = interp1d([min_label, max_label], [1, 9])

    # outDF = pd.DataFrame(m(labels), index=ids)
    outDF = pd.DataFrame(labels, index=ids)

    outDF.to_csv(output_path)

    hist, _ = np.histogram(labels, bins=[1,2,3,4,5,6,7,8,9,10])
    print hist


def prepare_data_for_classification(in_dir, out_dir_train, out_dir_test, delta_train, delta_test, test_size):
    # build ava dict
    ava_dict = build_ava_dictionary()

    # take local images that are already downloaded and get the image list
    filenames = glob.glob(in_dir + '*.ppm')

    img_ids = []
    for name in filenames:
        img_id = splitext(basename(name))[0]
        img_ids.append(int(img_id))

    if test_size > len(img_ids):
        raise Exception("test set larger than image set")

    # randomly split test and train
    img_ids = np.asarray(img_ids)
    np.random.shuffle(img_ids)
    train_ids, test_ids = img_ids[test_size:], img_ids[:test_size]

    # get labels for both train and test
    train_labels, train_ids = get_labels(ava_dict, train_ids, delta_train)
    test_labels, test_ids = get_labels(ava_dict, test_ids, delta_test)

    # write to file
    df = pd.DataFrame(train_ids)
    df.to_csv(out_dir_train + "imgIds.csv")
    df = pd.DataFrame(np.asarray(train_labels))
    df.to_csv(out_dir_train + "labels.csv")

    df = pd.DataFrame(test_ids)
    df.to_csv(out_dir_test + "imgIds.csv")
    df = pd.DataFrame(np.asarray(test_labels))
    df.to_csv(out_dir_test + "labels.csv")


if __name__ == '__main__':
    # get_labels_for_images_with_features()
    in_dir = "../data/resized_224/"
    out_dir_train = "../data/classification/train/"
    out_dir_test = "../data/classification/test/"

    prepare_data_for_classification(in_dir=in_dir,
                                    out_dir_train=out_dir_train,
                                    out_dir_test=out_dir_test,
                                    delta_train=0,
                                    delta_test=0,
                                    test_size=500)
