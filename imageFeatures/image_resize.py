"""
Fast image resize script: DRD@Kaggle

__author__ : Abhishek Thakur
"""

import os
import glob
from joblib import Parallel, delayed

in_dir = '../data/original/'
out_dir = '../data/resized_224/'
IMAGE_SIZE = 224

from PIL import Image, ImageChops
JPEG_FILES = glob.glob(in_dir+'*.jpg')


def convert(img_file):
    im = Image.open(img_file)
    im = im.convert('RGB')
    base = os.path.splitext(os.path.basename(img_file))[0]
    im.resize((IMAGE_SIZE, IMAGE_SIZE)).save(out_dir + base + ".ppm", 'PPM')

Parallel(n_jobs=32, verbose=10)(delayed(convert)(f) for f in JPEG_FILES)