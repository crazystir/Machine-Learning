import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import numpy as np
import random
import sys
import shiqiang as sq

#ASSET_DIR = '../asset'
ASSET_DIR = '/Users/tianzhang/Documents/NU/EECS349/project/asset'
ANNOTATION_FILE = ASSET_DIR + '/annotation.txt'

new_max = 127
new_min = -128

annotation = open(ANNOTATION_FILE, "r")
NUMBER = 1000
PIC_SIZE = 64
map_characters = {'abraham_grampa_simpson\n':0, 'apu_nahasapeemapetilon\n':1, 'bart_simpson\n':2,
        'charles_montgomery_burns\n':3, 'chief_wiggum\n':4, 'comic_book_guy\n':5, 'edna_krabappel\n':6,
        'homer_simpson\n':7, 'kent_brockman\n':8, 'krusty_the_clown\n':9, 'lisa_simpson\n':10,
        'marge_simpson\n':11, 'milhouse_van_houten\n':12, 'moe_szyslak\n':13,
        'ned_flanders\n':14, 'nelson_muntz\n':15, 'principal_skinner\n':16, 'sideshow_bob\n':17}

def _parse_function (filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)
    # image_resized = tf.image.resize_images(image_decoded, [100, 100], method=1)
    return image_decoded

#return a list-of-pic_dir
#return tag: transfer tag from characters to int
def pic_dir_list():
    pic_dir = []
    _x1 = []
    _y1 = []
    _x2 = []
    _y2 = []
    _tag_temp = []
    _tag = []
    first = True
    for line in annotation:
        if(first):
            first = False
            continue
        [image_dir, x1, y1, x2, y2, tag] = line.split(',')
        image_dir = ASSET_DIR + image_dir[1:]
        pic_dir.append(image_dir)
        _x1.append(x1)
        _y1.append(y1)
        _x2.append(x2)
        _y2.append(y2)
        _tag_temp.append(tag)

    for i in _tag_temp:
        _tag.append(map_characters[i])
    return pic_dir, _x1, _y1, _x2, _y2, _tag

#return 500 randomly choosen pictures from pic_list and relavant parameters
def random_choose_pics(pic_list, x1, y1, x2, y2, tag):
    pic_choosen = []
    _x1 = []
    _y1 = []
    _x2 = []
    _y2 = []
    _tag = []
    num_choosen = random.sample(range(1, len(pic_list)+1), NUMBER)
    for i in num_choosen:
        pic_choosen.append(pic_list[i])
        _x1.append(x1[i])
        _y1.append(y1[i])
        _x2.append(x2[i])
        _y2.append(y2[i])
        _tag.append(tag[i])
    return pic_choosen, _x1, _y1, _x2, _y2, _tag

#return array of resized pictures
#return array of tag of pictures
def loadImage():

    pic_list = []
    pic_choosen = []
    x1 =[]
    y1 = []
    x2 = []
    y2 = []
    tag_ = []
    tag_train = []
    tag_test = []
    pic_res = []
    test = []
    pic_list, x1, y1, x2, y2, tag_ = pic_dir_list()

    #randomly choose 500 pictures from pic_list
    pic_choosen, x1, y1, x2, y2, tag_  = random_choose_pics(pic_list, x1, y1, x2, y2, tag_)

    # check error
    if len(pic_choosen) != NUMBER or len(x1) != NUMBER or len(y1) != NUMBER or len(x2) != NUMBER or len(y2) != NUMBER or len(tag_) != NUMBER:
        print >> sys.stderr, 'wrong index!'

    print "len of pic_choosen is:", len(pic_choosen)
    #here for use if statement for debug use:
    for i in range(0, len(pic_choosen)):
        with tf.Session() as sess:
            _image = []
            image = _parse_function(pic_choosen[i])
            if int(y2[i]) - int(y1[i]) <= 0 or int(x2[i]) - int(x1[i]) <= 0:
                continue
            image = tf.image.crop_to_bounding_box(image, int(y1[i]), int(x1[i]), int(y2[i]) - int(y1[i]), int(x2[i]) - int(x1[i]))
            image_resized = tf.image.resize_images(image, [PIC_SIZE, PIC_SIZE], method=1)

            _image = np.array(image_resized.eval(session=sess))

            if i % 10 == 0 or i % 10 == 5 or i % 10 == 8:
                test.append(_image)
                tag_test.append(tag_[i])
            else:
                pic_res.append(_image)
                tag_train.append(tag_[i])
            #plt.imshow(image.eval(session=sess))
            #plt.show()
    tag_num = len(tag_train)
    tag_train = np.array(tag_train)
    tag_train.shape = (1, tag_num)

    tag_num = len(tag_test)
    tag_test = np.array(tag_test)
    tag_test.shape = (1, tag_num)

    pic_res = np.array(pic_res)
    test = np.array(test)

    return pic_res, test, tag_train.T, tag_test.T

#input
def Normalization(a):
    #rememeber the outer dimension of array
    o_dimension = a.shape[0]
    for i in range(o_dimension):
        i_dimension = a[i].shape
        temp1 = a[i].flatten()
        temp2 = sorted(a[i].flatten())
        min = temp2[0]
        max = temp2[-1]
        idx = 0
        while(idx < len(temp1)):
            temp1[idx] = (temp1[idx]-min)*(new_max - new_min) / (max - min) + new_min
            idx += 1
        temp1.shape = i_dimension
        a[i] = temp1
    return a

if __name__ == '__main__':
    picsdata = []
    testdata = []
    tag_train = []
    tag_test = []

    picsdata, testdata, tag_train, tag_test = loadImage()
    picsdata = Normalization(picsdata)
    testdata = Normalization(testdata)
    tag_train = Normalization(tag_train)
    tag_test = Normalization(tag_test)

    print("the shape of pic data is: ")
    print(picsdata.shape)
    print("the shape of test data is: ")
    print(testdata.shape)
    print("the shape of tag_train data is: ")
    print(tag_train.shape)
    print("the shape of tag_test data is: ")
    print(tag_test.shape)

    sq.TrainAndTest(picsdata, testdata, tag_train.flatten(), tag_test.flatten())
