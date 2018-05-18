import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import numpy as np

ASSET_DIR = '../asset'
ANNOTATION_FILE = ASSET_DIR + '/annotation.txt'

annotation = open(ANNOTATION_FILE, "r")

def _parse_function (filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)
    # image_resized = tf.image.resize_images(image_decoded, [100, 100], method=1)
    return image_decoded, label



def loadImage():
    first = True
    for line in annotation:
        if (first):
            first = False
            continue
        [image_dir, x1, y1, x2, y2, tag] = line.split(',')
        image_dir = ASSET_DIR + image_dir[1:]
        # image = mpimg.imread(image_dir)
        # plt.imshow(image)
        # plt.show()
        with tf.Session() as sess:
            image, tag = _parse_function(image_dir, tag)
            # image = tf.expand_dims(tf.image.convert_image_dtype(image, dtype=tf.float32), 0)
            # boxes = tf.constant([[[0.1, 0.1, 0.8, 0.8]]], dtype=tf.float32, shape=[1,1,4])
            # tf.image.draw_bounding_boxes(image, boxes)
            # plt.figure(1)
            print (image.shape)
            if int(y2) - int(y1) <= 0 or int(x2) - int(x1) <= 0:
                continue
            image = tf.image.crop_to_bounding_box(image, int(y1), int(x1), int(y2) - int(y1), int(x2) - int(x1))
            plt.imshow(image.eval(session=sess))
            plt.show()



if __name__ == '__main__':
    loadImage()