import glob
import pandas as pd
import pull_labels as pl
import numpy as np
import json
lower = 4.53178
upper = 5.88806


def get_food_images(filename):
    input_file = pd.read_csv(filename, header=None)
    ret = []
    for idx, row in input_file.iterrows():
        # pull_files(row[0], output_folder)
        ret.append(str(row[0]))
    return ret


def get_food_ratings(food_ids, ava_dictionary, save_file):
    ids = []
    ratings = []
    for im_id in food_ids:
        if im_id in ava_dictionary.keys():
            ids.append(im_id)
            line = ava_dictionary[im_id]
            ratings.append(pl.get_score(line))
        else:
            print "ID: {} not in ava_txt".format(im_id)

    df = pd.DataFrame(ratings, index=ids)
    df.to_csv(save_file)


def get_food_bins(avadict, food_images_ids):
    bins = {"1":[], "2":[], "3":[], "4":[], "5":[],
            "6":[], "7":[], "8":[], "9":[]}

    for k in food_images_ids:
        if k in avadict.keys():
            line = avadict[k]
            score = pl.get_score(line)

            if score > 4 and score <= 5:
                bins["4"].append(k)
            elif score > 5 and score <= 6:
                bins["5"].append(k)
            elif score > 3 and score <= 4:
                bins["3"].append(k)
            elif score > 6 and score <= 7:
                bins["6"].append(k)
            elif score > 2 and score <= 3:
                bins["2"].append(k)
            elif score > 7 and score <= 8:
                bins["7"].append(k)
            elif score > 1 and score <= 2:
                bins["1"].append(k)
            elif score > 8 and score <= 9:
                bins["8"].append(k)
            elif score > 9 and score <= 10:
                bins["9"].append(k)
            else:
                print "error"
                print score
    return bins


def get_nonfood_bins(avadict, food_images_ids):

    bins = {"1":[], "2":[], "3":[], "4":[], "5":[],
            "6":[], "7":[], "8":[], "9":[]}
    # all_scores = []
    for k in avadict.keys():
        if k not in food_images_ids:
            # pull_files(row[0], output_folder)
            score = pl.get_score(avadict[k])
            # all_scores.append(score)
            if score > 4 and score <= 5:
                bins["4"].append(k)
            elif score > 5 and score <= 6:
                bins["5"].append(k)
            elif score > 3 and score <= 4:
                bins["3"].append(k)
            elif score > 6 and score <= 7:
                bins["6"].append(k)
            elif score > 2 and score <= 3:
                bins["2"].append(k)
            elif score > 7 and score <= 8:
                bins["7"].append(k)
            elif score > 1 and score <= 2:
                bins["1"].append(k)
            elif score > 8 and score <= 9:
                bins["8"].append(k)
            elif score > 9 and score <= 10:
                bins["9"].append(k)
            else:
                print "error"
                print score

    # hist, _ = np.histogram(all_scores, bins=range(1, 11))
    # print hist
    return bins



if __name__ == "__main__":
    ava_file = "AVA_dataset/AVA.txt"
    in_dir = "AVA_dataset/aesthetics_image_lists/"
    save_file = "food_drink_ratings.csv"

    filenames = glob.glob(in_dir + '*fooddrink*.jpgl')

    # get food/drinks images list
    food_images_ids = []
    for filename in filenames:
        food_images_ids += get_food_images(filename)
    print food_images_ids
    # get ratings of food/drinks images
    ava_dict = pl.build_ava_dictionary()

    # get_food_ratings(food_images_ids, ava_dict, save_file)

    # food_ratings = pd.read_csv(save_file)
    # labels = np.asarray(food_ratings.iloc[:, 1].values)

    # hist1, _ = np.histogram(labels, bins=range(1,11))
    # print hist1
    # [   2    4   96 1826 2515  518   25    2    1]
    # all ava bins
    # [   422    525   5112  77230 141319  28786   1886    170     80]
    # none food bins
    # [   420    521   5016  75404 138804  28268   1861    168     79]
    # hist2, _ = np.histogram(labels, bins=[1, lower, upper, 10])
    # print hist2


    # get bins for food and non-food images (ids)
    non_food_bins = get_nonfood_bins(ava_dict, food_images_ids)

    food_bins = get_food_bins(ava_dict, food_images_ids)

    final_food_ids = []
    final_non_food_ids = []
    # try to balance
    bin_size = 500
    for k in food_bins.keys():
        if len(food_bins[k]) < bin_size:
            # take photos from nonfood category
            take = bin_size - len(food_bins[k])

            if take >= len(non_food_bins[k]):
                # take all
                final_non_food_ids += non_food_bins[k]
            else:
                # take some
                final_non_food_ids += non_food_bins[k][0:take]
            final_food_ids += food_bins[k]

            print "take key = {}, {}".format(k, len(food_bins[k]) + len(non_food_bins[k][0:take]))
        else:
            # drop photos from food category
            final_food_ids += food_bins[k][0:bin_size]
            print "drop key = {}, {}".format(k, len(food_bins[k][0:bin_size]))

    food_ids_file = "balanced_food_ids.csv"
    non_food_ids_file = "balanced_non_food_ids.csv"

    df = pd.DataFrame(final_food_ids)
    df.to_csv(food_ids_file)

    df = pd.DataFrame(final_non_food_ids)
    df.to_csv(non_food_ids_file)




