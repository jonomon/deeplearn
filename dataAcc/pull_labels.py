import numpy as np
from os.path import basename
from os.path import splitext

ava_file = "AVA_dataset/AVA.txt"

image_features_file = "../imageFeatures/outputFeatures/image_features.csv"

food_list_test_file = "AVA_dataset/aesthetics_image_lists/fooddrink_test.jpgl"
food_list_train_file = "AVA_dataset/aesthetics_image_lists/fooddrink_train.jpgl"

output_path = "../data/labels/label.csv"

def error(str):
    raise StandardError(str);
    exit(-1)

# taking average as score
def get_score(line):
    scores = np.array([int(i) for i in line[2:12]])
    score = np.mean(scores)
    return score

# get test and train image ids for food pictures
def get_ids():
    test_ids = []
    train_ids = []
    with open(food_list_test_file) as infile:
        for line in infile:
            test_ids.append(line.strip())

    with open(food_list_train_file) as infile:
        for line in infile:
            train_ids.append(line.strip())

    return test_ids, train_ids


# build the ava.txt file into a dictionary, indexed by image id string
def build_ava_dictionary():
    ava_dict = {}

    with open(ava_file) as infile:
        for line in infile:
            line = line.split()

            img_id = line[1]
            if img_id in ava_dict:
                error( "Abort: duplicate image found! possible data corruption!")
            else:
                ava_dict[img_id] = line

    return ava_dict


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

    test_ids, train_ids = get_ids()

    ava_dict = build_ava_dictionary()

    # train & test images combined!
    labels = []

    for img_id in img_ids:
        if img_id in ava_dict:
            score = get_score(ava_dict[img_id])
            if img_id in test_ids or img_id in train_ids:
                labels.append(score)
            else:
                error("Abort: image not found as train or test, possible data corruption!")
        else:
            error("Abort: image not found in AVA.txt, possible data corruption!")

    np.savetxt(output_path, np.asarray(labels), fmt="%f")

if __name__ == '__main__':
    get_labels_for_images_with_features();