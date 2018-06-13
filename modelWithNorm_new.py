import tensorflow as tf
import numpy as np
import random
import sys
import matplotlib.pyplot as plt

tf.logging.set_verbosity(tf.logging.INFO)


def load_data_source(filename, batch_size):
    classes = 3
    depth = 3
    input_size = 64
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([1], tf.int64),
                                           'image': tf.FixedLenFeature([], tf.string),
                                       })
    images = tf.decode_raw(features['image'], tf.int32)
    images = tf.reshape(images, [depth, input_size, input_size])
    images = tf.transpose(images, [1, 2, 0])
    label = tf.cast(features['label'], tf.int32)

    x_batch, y_batch = tf.train.shuffle_batch([images, label], batch_size=batch_size,
                                              capacity=1000, min_after_dequeue=200, num_threads=4)
    index = tf.expand_dims(tf.range(0, batch_size), 1)
    concated = tf.concat([index, y_batch], axis=1)
    y_batch = tf.cast(tf.sparse_to_dense(concated, [batch_size, classes], 1.0, 0.0), dtype=tf.float32)

    return x_batch, y_batch


def cnn_model_fn():

    # tag of train of test
    tag = tf.placeholder(tf.bool, name = 'tag')
    # Input Layer
    x = tf.placeholder("float", [None , 64, 64, 3], name="x")

    x_norm = tf.layers.batch_normalization(x, training = tag)
    # Convolutional Layer #1
    #32
    conv1_1 = tf.layers.conv2d(
        inputs=x_norm,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        name="first",
        activation=tf.nn.relu)

    # conv1_1_norm = tf.layers.batch_normalization(conv1_1, training = tag)
    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1_1, pool_size=[2, 2], strides=2)

    pool1_norm = tf.layers.batch_normalization(pool1, training = tag)
    # Convolutional Layer #2
    #64
    conv2_1 = tf.layers.conv2d(
        inputs=pool1_norm,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        name="second",
        activation=tf.nn.relu
    )

    # conv2_1_norm = tf.layers.batch_normalization(conv2_1, training=tag)
    # Pooling Layer #2

    pool2 = tf.layers.max_pooling2d(inputs=conv2_1, pool_size=[2, 2], strides=2)

    pool2_norm = tf.layers.batch_normalization(pool2, training=tag)
    # Convolutional Layer #3
    #128
    conv3_1 = tf.layers.conv2d(
        inputs=pool2_norm,
        filters=128,
        kernel_size=[5, 5],
        padding="same",
        name="third",
        activation=tf.nn.relu
    )

    # conv3_1_norm = tf.layers.batch_normalization(conv3_1, training=tag)
    # Pooling Layer #3
    pool3 = tf.layers.average_pooling2d(inputs=conv3_1, pool_size=[2, 2], strides=2)

    pool3_norm = tf.layers.batch_normalization(pool3, training=tag)

    pool_flat = tf.reshape(pool3_norm, [-1, 8 * 8 * 128])

    dense = tf.layers.dense(inputs=pool_flat, units=1024, activation=tf.nn.relu)

    dropout = tf.layers.dropout(
        inputs=dense, rate=0.5)

    logits = tf.layers.dense(inputs=dropout, units=3)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    y = tf.placeholder("float", [None, 3], name="y")
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=logits)
    tf.summary.scalar('loss', loss)

    lr = tf.train.exponential_decay(0.01, 5000, 200, 0.9)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

    correct_prediction = tf.equal(tf.argmax(logits, axis=1), tf.argmax(y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar('accuracy', accuracy)
    merged = tf.summary.merge_all()

    return x, y, tag, train_op, accuracy, loss, merged, predictions

def save(sess):
    saver = tf.train.Saver()
    save_path = saver.save(sess, "./model/model.ckpt")
    print("Model saved in file: %s" % save_path)

def TrainAndTest():
    # Load training and eval data
    # mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    batch_size = 64
    test_step = 10
    x_train, y_train = load_data_source("train.tfrecord", batch_size)
    x_test, y_test = load_data_source("test.tfrecord", batch_size)
    img_x_test, img_y_test = load_data_source("img.tfrecord", batch_size)

    training_iters = 300
    x, y, tag, train_op, accuracy, loss, merged, predictions = cnn_model_fn()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        sess.run(init)
        print("===== Start Training! =====")
        train_writer = tf.summary.FileWriter('./train_v1', sess.graph)
        generation_error = []
        step = 1
        time = []
        train_acc_list = []
        test_acc_list = []
        while step <= training_iters:
            batch_xs, batch_ys = sess.run([x_train, y_train])
            sess.run(train_op, feed_dict={x: batch_xs, y: batch_ys, tag: True})
            if step % test_step == 0:
                summary, train_loss, train_acc = sess.run([merged, loss, accuracy],
                                              feed_dict={x: batch_xs, y: batch_ys, tag: True})
                train_writer.add_summary(summary, step)
                print("Iter " + str(step) + ":\nTrain Loss = " + "{:.6f}".format(
                    train_loss) + ", Train Accuracy = " + "{:.5f}".format(train_acc))
                test_data, test_label = sess.run([x_test, y_test])
                test_loss, test_acc = sess.run([loss, accuracy], feed_dict={x: test_data, y: test_label, tag: False})
                print("Test Loss = " + "{:.6f}".format(test_loss) + ", Test Accuracy = " + "{:.5f}".format(test_acc))
                print("Gen Error = " + "{:.6f}".format(test_loss - train_loss))
                generation_error.append(test_loss - loss)
                time.append(step)
                train_acc_list.append(train_acc)
                test_acc_list.append(test_acc)
            step += 1

        print("plotting")
        plt.plot(time, train_acc_list, label='train', linewidth=2)
        plt.plot(time, test_acc_list, label='test', linewidth=2)
        plt.title('Train accuracy VS time and Test accuracy VS time')
        plt.xlabel('time')
        plt.ylabel('accuracy')
        plt.legend(('train', 'test'), shadow = True, loc = 'lower right')
        plt.show()

        print("Optimization Finished!")
        test_data, test_label = sess.run([img_x_test, img_y_test])
        prediction = sess.run([predictions], feed_dict={x: test_data, y: test_label, tag: False})
        print("prediction: ", prediction)
        np.save('generation_v1.npy', np.array(generation_error))
        train_writer.close()
        coord.request_stop()
        coord.join(threads)
        save(sess)


if __name__=="__main__":
    TrainAndTest()
