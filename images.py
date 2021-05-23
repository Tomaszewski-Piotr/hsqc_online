from PIL import Image
import numpy as np
import pathlib

value_names =  ['glc', 'man', 'gal', 'ara', 'xyl']
pred_names = ['glc_pred', 'man_pred', 'gal_pred', 'ara_pred', 'xyl_pred']

target_width = 224
target_height = 224
channels = 3

def get_data_files():
    files = []
    for subdir in pathlib.Path('trainingset').glob('*'):
        for file in pathlib.Path(subdir).glob('*.png'):
            files.append(file)
    return files

def extract_values(filename):
    splitted = str(filename.name).split('_')
    values=[]
    for i in range(0, 5):
        values.append(float(splitted[i]))
    return values


def preprocess_image(filename):
    img = Image.open(filename)
    # setting crop values that "seem to work" to remove axis from the picture
    #remove 5% from each side to get rid of the black frame
    width = img.size[0]
    heigth = img.size[1]

    cropped_img = img.crop((int(width/20), int(heigth/20), int(width*0.95), int(heigth*0.95)))
    resized_img = cropped_img.resize(size=(target_height, target_width))
    as_array = np.array(np.asarray(resized_img))
    as_array = as_array[:,:,:channels]
    img.close()
    resized_img.close()
    return as_array

def FindCrop(filename):
    # Read image

    img = Image.open(filename)
    #img.show()
    print(img.size)
    width = img.size[0]
    height = img.size[1]
    #area = (emptyX, 0, 7200, 4800)
    #cropped_img = img.crop(area)
    #cropped_img.show()
    #exit(1)
    # Output Images
    #img.show()
    xstart=0
    ystart=0
    xstop=width
    ystop = height
    a = np.asarray(img)
    black = False
    for x in range(0, width):
        point = a[emptyY, x]
        if black and point[0] == 255:
            xstart = x
            break
        if not black and point[0] != 255:
            black = True
    black = False
    for x in range(0, width):
        point = a[emptyY, width-x-1]
        if black and point[0] == 255:
            xstop=width-x-1
            break
        if not black and point[0] != 255:
            black = True
    black = False
    for y in range(0, height):
        point = a[y, emptyX]
        if black and point[0] == 255:
            ystart = y
            break
        if not black and point[0] != 255:
            black = True

    black = False
    for y in range(0, height):
        point = a[height-y-1, emptyX]
        if black and point[0] == 255:
            ystop=height-y-1
            break
        if not black and point[0] != 255:
            black = True

    print(xstart, ystart, xstop, ystop)
    return xstart+10, ystart+10, xstop-30, ystop-30