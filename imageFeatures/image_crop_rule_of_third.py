from PIL import Image

IMAGE_SIZE = 224

img_file = '../data/42462.jpg'


im = Image.open(img_file)

im.show()

raw_input()