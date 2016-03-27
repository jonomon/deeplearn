import numpy as np
ava_file = "AVA_dataset/AVA.txt"
output_path = "../data/labels/label.csv"

food_list_test_file = "AVA_dataset/aesthetics_image_lists/fooddrink_test.jpgl"
food_list_train_file = "AVA_dataset/aesthetics_image_lists/fooddrink_train.jpgl"

food_img_ids = []

with open(food_list_test_file) as infile:
    for line in infile:
        food_img_ids.append(int(line))

with open(food_list_train_file) as infile:
    for line in infile:
        food_img_ids.append(int(line))


with open(ava_file) as infile:

    output = open(output_path, "w")

    results = []

    for line in infile:
        line = line.split()

        id = int( line[1])

        if (id in food_img_ids):
            scores = np.array([int(i) for i in line[2:12]])
            score = np.mean(scores)
            results.append([id, score])

    np.savetxt(output, np.asarray(results), fmt="%d, %f")


