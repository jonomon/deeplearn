from PIL import Image
import glob
from os.path import basename
from os.path import splitext

from joblib import Parallel, delayed

IMAGE_SIZE = 224

in_dir = '../data/original/'
out_dir = '../data/rot/'


def rot_crop_all():
    JPEG_FILES = glob.glob(in_dir+'*.jpg')
    Parallel(n_jobs=32, verbose=10)(delayed(rot_crop)(f) for f in JPEG_FILES)


def rot_crop(img_file):
    im = Image.open(img_file)
    (x, y) = im.size

    if x > y:
        # landscape image
        l = x/6
        r = l + x/3
        t = y/6
        b = t + 2*y/3

        box_1 = (l, t, r, b)

        box_2 = (l + x/3, t, r + x/3, b)
    else:
        # portrait image
        l = x/6
        r = l + x/3 * 2
        t = y/6
        b = t + y/3

        box_1 = (l, t, r, b)

        box_2 = (l, t + y/3, r, b + y/3)

    base = splitext(basename(img_file))[0]
    im.crop(box_1).resize((IMAGE_SIZE, IMAGE_SIZE)).save(out_dir + base + "_1" + ".ppm", 'PPM')
    im.crop(box_2).resize((IMAGE_SIZE, IMAGE_SIZE)).save(out_dir + base + "_2" + ".ppm", 'PPM')

if __name__ == '__main__':
    rot_crop_all()