import tensorflow as tf
import random
import sys
import numpy as np
from numpy import linalg as LA

# ASSET_DIR = '../asset'
#ASSET_DIR = './the-simpsons-characters-dataset'
ASSET_DIR = './asset'
ANNOTATION_FILE = ASSET_DIR + '/annotation.txt'

NORM_FACTOR = 128
OFFSET = 0.5

annotation = open(ANNOTATION_FILE, "r")
NUMBER = 2000
PIC_SIZE = 64
map_characters = {'abraham_grampa_simpson\n': 0, 'apu_nahasapeemapetilon\n': 1, 'bart_simpson\n': 2,
                  'charles_montgomery_burns\n': 3, 'chief_wiggum\n': 4, 'comic_book_guy\n': 5, 'edna_krabappel\n': 6,
                  'homer_simpson\n': 7, 'kent_brockman\n': 8, 'krusty_the_clown\n': 9, 'lisa_simpson\n': 10,
                  'marge_simpson\n': 11, 'milhouse_van_houten\n': 12, 'moe_szyslak\n': 13,
                  'ned_flanders\n': 14, 'nelson_muntz\n': 15, 'principal_skinner\n': 16, 'sideshow_bob\n': 17}

def _parse_function(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)
    # image_resized = tf.image.resize_images(image_decoded, [100, 100], method=1)
    return image_decoded


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
        if (first):
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


# return 500 randomly choosen pictures from pic_list and relavant parameters
def random_choose_pics(pic_list, x1, y1, x2, y2, tag):
    pic_choosen = []
    _x1 = []
    _y1 = []
    _x2 = []
    _y2 = []
    _tag = []
    num_choosen = random.sample(range(1, len(pic_list)), NUMBER)
    for i in num_choosen:
        pic_choosen.append(pic_list[i])
        _x1.append(x1[i])
        _y1.append(y1[i])
        _x2.append(x2[i])
        _y2.append(y2[i])
        _tag.append(tag[i])
    return pic_choosen, _x1, _y1, _x2, _y2, _tag

def norm(v):
    if LA.norm(v) == 0:
       return 1
    norm = LA.norm(v)
    return norm

if __name__ == "__main__":
    examples = []
    pic_list, x1, y1, x2, y2, tag_ = pic_dir_list()
    # randomly choose 500 pictures from pic_list
    pic_choosen, x1, y1, x2, y2, tag_ = random_choose_pics(pic_list, x1, y1, x2, y2, tag_)

    # check error
    if len(pic_choosen) != NUMBER or len(x1) != NUMBER or len(y1) != NUMBER or len(x2) != NUMBER or len(
            y2) != NUMBER or len(tag_) != NUMBER:
        print >> sys.stderr, 'wrong index!'

    print "len of pic_choosen is:", len(pic_choosen)

    train_writer = tf.python_io.TFRecordWriter("train.tfrecord")
    test_writer = tf.python_io.TFRecordWriter("test.tfrecord")
    sess = tf.Session()
    for i in range(0, len(pic_choosen)):
        _image = []
        image = _parse_function(pic_choosen[i])
        if int(y2[i]) - int(y1[i]) <= 0 or int(x2[i]) - int(x1[i]) <= 0:
            continue

        image = tf.image.crop_to_bounding_box(image, int(y1[i]), int(x1[i]), int(y2[i]) - int(y1[i]),
                                              int(x2[i]) - int(x1[i]))

        image_resized = tf.image.resize_images(image, [PIC_SIZE, PIC_SIZE], method=1)

        # #do the normalizatin for each image:
        # each_image = np.array(image_resized.eval(session=sess), dtype=np.int32)
        # #R
        # each_image[:,:,0] = ((each_image[:,:,0] / norm(each_image[:,:,0])) - OFFSET) * NORM_FACTOR
        # #G
        # each_image[:,:,1] = ((each_image[:,:,1] / norm(each_image[:,:,1])) - OFFSET) * NORM_FACTOR
        # #B
        # each_image[:,:,2] = ((each_image[:,:,2] / norm(each_image[:,:,2])) - OFFSET) * NORM_FACTOR
        #each_image = each_image.flatten()

        each_image = np.array(image_resized.eval(session=sess), dtype=np.int32).flatten()

        feature = each_image.tostring()
        label = tag_[i]
        example = tf.train.Example(
            features=tf.train.Features(
                feature={'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                         'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[feature]))
                         }))
        examples.append(example)

    random.shuffle(examples)

    for i in range(0, len(examples) * 3 / 10):
        test_writer.write(examples[i].SerializeToString())
    for i in range(len(examples) * 3 / 10, len(examples)):
        train_writer.write(examples[i].SerializeToString())

    train_writer.close()
    test_writer.close()