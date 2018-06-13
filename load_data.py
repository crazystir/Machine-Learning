import tensorflow as tf
import random
import sys
import numpy as np
from numpy import linalg as LA

# ASSET_DIR = '../asset'
#ASSET_DIR = './the-simpsons-characters-dataset'
ASSET_DIR = '/Users/tianzhang/Documents/NU/EECS349/project/asset'
ANNOTATION_FILE = ASSET_DIR + '/annotation.txt'

LABEL1 = 0
LABEL2 = 1
LABEL3 = 2

annotation = open(ANNOTATION_FILE, "r")
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
def two_people_pics(pic_list, x1, y1, x2, y2, tag, LABEL1, LABEL2, LABEL3):
    pic_choosen = []
    _x1 = []
    _y1 = []
    _x2 = []
    _y2 = []
    _tag = []
    _idx = 0

    for i in tag:
        if i == LABEL1 or i == LABEL2 or i == LABEL3:
            pic_choosen.append(pic_list[_idx])
            _x1.append(x1[_idx])
            _y1.append(y1[_idx])
            _x2.append(x2[_idx])
            _y2.append(y2[_idx])
            _tag.append(tag[_idx])
        _idx += 1

    return pic_choosen, _x1, _y1, _x2, _y2, _tag



if __name__ == "__main__":
    examples = []
    img_dir = []
    pic_list, x1, y1, x2, y2, tag_ = pic_dir_list()
    # randomly choose 500 pictures from pic_list
    pic_choosen, x1, y1, x2, y2, tag_ = two_people_pics(pic_list, x1, y1, x2, y2, tag_, LABEL1, LABEL2, LABEL3)

    for i in tag_:
        print "calss is: ", i

    # check dir correct or not, _tag correct correct or not
    # print len(pic_choosen)
    #
    # idx_print = 0
    # while(idx_print < len(pic_choosen)):
    #     print "idex is: ", idx_print
    #     print "pic dir is: ", pic_choosen[idx_print]
    #     idx_print += 1


    train_writer = tf.python_io.TFRecordWriter("train.tfrecord")
    test_writer = tf.python_io.TFRecordWriter("test.tfrecord")
    img_writer1 = tf.python_io.TFRecordWriter("img1.tfrecord")
    img_writer2 = tf.python_io.TFRecordWriter("img2.tfrecord")
    img_writer3 = tf.python_io.TFRecordWriter("img3.tfrecord")
    img_writer4 = tf.python_io.TFRecordWriter("img4.tfrecord")
    img_writer5 = tf.python_io.TFRecordWriter("img5.tfrecord")
    img_writer6 = tf.python_io.TFRecordWriter("img6.tfrecord")
    sess = tf.Session()
    num = 0
    for i in range(0, len(pic_choosen)):
        _image = []
        image = _parse_function(pic_choosen[i])
        if int(y2[i]) - int(y1[i]) <= 0 or int(x2[i]) - int(x1[i]) <= 0:
            continue
        image = tf.image.crop_to_bounding_box(image, int(y1[i]), int(x1[i]), int(y2[i]) - int(y1[i]),
                                              int(x2[i]) - int(x1[i]))
        image_resized = tf.image.resize_images(image, [PIC_SIZE, PIC_SIZE], method=1)


        each_image = np.array(image_resized.eval(session=sess), dtype=np.int32).flatten()

        feature = each_image.tostring()
        label = tag_[i]
        example = tf.train.Example(
            features=tf.train.Features(
                feature={'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                         'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[feature]))
                         }))
        examples.append(example)

        if i == 97:
             img_writer1.write(example.SerializeToString())
        if i == 98:
             img_writer2.write(example.SerializeToString())

        if i == 658:
             img_writer3.write(example.SerializeToString())
        if i == 659:
             img_writer4.write(example.SerializeToString())

        if i == 735:
             img_writer5.write(example.SerializeToString())
        if i == 736:
             img_writer6.write(example.SerializeToString())

    random.shuffle(examples)

    for i in range(0, len(examples) / 3):
        test_writer.write(examples[i].SerializeToString())
    for i in range(len(examples) / 3, len(examples)):
        train_writer.write(examples[i].SerializeToString())

    train_writer.close()
    test_writer.close()
    img_writer1.close()
    img_writer2.close()
    img_writer3.close()
    img_writer4.close()
    img_writer5.close()
    img_writer6.close()

    print "load_data finish!"