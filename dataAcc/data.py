import pandas as pd
from urllib2 import urlopen, URLError, HTTPError
import bs4
import glob
import os.path as ospath
from joblib import Parallel, delayed
import os
import numpy as np

def file_image_url(image_id, html):
    soup = bs4.BeautifulSoup(html, "lxml")
    imgs = soup.findAll(
        lambda tag:
        'alt' in tag.attrs and
        'src' in tag.attrs and
        tag.attrs['src'].startswith('http://images.dpchallenge.com/')
        and 'style' in tag.attrs and
        tag.attrs['src'].find('thumb') < 0
    )
    if len(imgs) < 1:
        if soup.find(text='Invalid IMAGE_ID provided.') is not None:
            return None
        raise Exception('No image found at url {}'.format(image_id))
    elif len(imgs) > 1:
        raise Exception('More than one image found at url {}'.format(image_id))

    img_url = imgs[0]['src']
    return img_url


def pull_files(image_id, output_folder, i):
    url_base = "http://www.dpchallenge.com/image.php?IMAGE_ID="
    url = url_base + str(image_id)
    output_filename = output_folder + str(image_id) + ".jpg"
    try:
        f = urlopen(url)
        print str(i) + " downloading " + url
        image_url = file_image_url(image_id, f.read())
        img = urlopen(image_url)
        with open(output_filename, "wb") as local_file:
            local_file.write(img.read())
    except HTTPError, e:
        print "HTTP Error:", e.code, url
    except URLError, e:
        print "URL Error:", e.reason, url
    except Exception as e:
        print e.message

def download_balanced(in_dir):
    output_folder_balanced = "../data/balanced/"
    filenames = glob.glob("balanced_*.csv")

    for filename in filenames:
        input_file = pd.read_csv(filename, header=0, index_col=0)

        for idx, row in input_file.iterrows():
            if not ospath.isfile(output_folder_balanced + str(row[0]) + ".jpg"):
                pull_files(row[0], output_folder_balanced)

def download_ava():
    ava_file = "AVA_dataset/AVA.txt"
    out_dir = "../data/data_backup/"

    ava_file = "../dataAcc/AVA_dataset/AVA.txt"
    ava = pd.read_csv(ava_file, sep=" ", header=None).set_index(1).drop(0, axis=1)

    paths = glob.glob(out_dir + "*jpg")
    paths2 = []
    for path in paths:
        basename = os.path.basename(path)
        paths2.append(basename.split('.')[0])
    not_in_path_idx = np.in1d(ava.index, paths2)
    toDownload = ava.index[np.logical_not(not_in_path_idx)]
    Parallel(n_jobs=32, verbose=10)(
        delayed(pull_files)(f, out_dir, i) for i, f in enumerate(toDownload))
    
if __name__ == "__main__":
    in_dir = "AVA_dataset/aesthetics_image_lists/"
    #filenames = glob.glob(in_dir+'*.jpgl')
    # filenames = glob.glob(in_dir + '*fooddrink*.jpgl')

    # output_folder = "../data/original/"
    # for filename in filenames:
    #     input_file = pd.read_csv(filename, header=None)
    #     for idx, row in input_file.iterrows():
    #         pull_files(row[0], output_folder)

    download_ava()
