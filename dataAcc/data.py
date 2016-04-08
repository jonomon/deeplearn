import pandas as pd
from urllib2 import urlopen, URLError, HTTPError
import bs4
import glob
import os.path as ospath

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


def pull_files(image_id, output_folder):
    url_base = "http://www.dpchallenge.com/image.php?IMAGE_ID="
    url = url_base + str(image_id)
    output_filename = output_folder + str(image_id) + ".jpg"
    try:
        f = urlopen(url)
        print "downloading " + url
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

    with open(ava_file, 'r') as fh:
        for line in fh:
            line = line.split()
            im_id = line[1]
            if not ospath.isfile(out_dir + im_id + ".jpg"):
                pull_files(im_id, out_dir)

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
