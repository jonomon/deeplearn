from PIL import Image
import glob
from os.path import basename
from os.path import splitext

from joblib import Parallel, delayed

IMAGE_SIZE = 224

in_dir = '../data/data_backup/'
out_dir = '../data/rot/'


def rot_crop_all():
    JPEG_FILES = glob.glob(in_dir+'*.jpg')
    Parallel(n_jobs=32, verbose=10)(delayed(rot_crop4)(f) for f in JPEG_FILES)

def rot_crop4_1(img_file):
    print(img_file)
    # divide the image into 4 images based on the rule of thirds
    try:
        im = Image.open(img_file)
    except IOError:
        print("Error opening")
    im = im.convert('RGB')
    (size_x, size_y) = im.size
    x1 = size_x/3
    x2 = size_x * 2/3
    y1 = size_y/3
    y2 = size_y * 2/3

    cropped_third_image_size = (size_x * 2/3, size_y * 2/3)
    min_axis = min(cropped_third_image_size)
    
    # ____________
    # |_1_|_1_|___|
    # |_1_|_1_|___|
    # |___|___|___| Example of box1 therefore the third will be in the centre of box1
    box1 = (x1 - min_axis/2, y1 - min_axis/2,
            x1 + min_axis/2, y1 + min_axis/2)
    box2 = (x1 - min_axis/2, y2 - min_axis/2,
            x1 + min_axis/2, y2 + min_axis/2)
    box3 = (x2 - min_axis/2, y1 - min_axis/2,
            x2 + min_axis/2, y1 + min_axis/2)
    box4 = (x2 - min_axis/2, y2 - min_axis/2,
            x2 + min_axis/2, y2 + min_axis/2)
    base = splitext(basename(img_file))[0]
    im1 = im.crop(box1)
    im2 = im.crop(box2)
    im3 = im.crop(box3)
    im4 = im.crop(box4)
    
    
def rot_crop4(img_file):
    # divide the image into 4 images based on the rule of thirds
    im = Image.open(img_file)
    im = im.convert('RGB')
    (size_x, size_y) = im.size
    x1 = size_x/3
    x2 = size_x * 2/3
    y1 = size_y/3
    y2 = size_y * 2/3

    cropped_third_image_size = (size_x * 2/3, size_y * 2/3)
    min_axis = min(cropped_third_image_size)
    
    # ____________
    # |_1_|_1_|___|
    # |_1_|_1_|___|
    # |___|___|___| Example of box1 therefore the third will be in the centre of box1
    box1 = (x1 - min_axis/2, y1 - min_axis/2,
            x1 + min_axis/2, y1 + min_axis/2)
    box2 = (x1 - min_axis/2, y2 - min_axis/2,
            x1 + min_axis/2, y2 + min_axis/2)
    box3 = (x2 - min_axis/2, y1 - min_axis/2,
            x2 + min_axis/2, y1 + min_axis/2)
    box4 = (x2 - min_axis/2, y2 - min_axis/2,
            x2 + min_axis/2, y2 + min_axis/2)
    base = splitext(basename(img_file))[0]
    im.crop(box1).resize((IMAGE_SIZE, IMAGE_SIZE)).save(out_dir + base + "_1" + ".ppm", 'PPM')
    im.crop(box2).resize((IMAGE_SIZE, IMAGE_SIZE)).save(out_dir + base + "_2" + ".ppm", 'PPM')
    im.crop(box3).resize((IMAGE_SIZE, IMAGE_SIZE)).save(out_dir + base + "_3" + ".ppm", 'PPM')
    im.crop(box4).resize((IMAGE_SIZE, IMAGE_SIZE)).save(out_dir + base + "_4" + ".ppm", 'PPM')
    
def rot_crop2(img_file):
    im = Image.open(img_file)

    im = im.convert('RGB')
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
