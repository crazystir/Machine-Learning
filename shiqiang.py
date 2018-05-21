import tensorflow as tf
import numpy as np
import random
import sys

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 64, 64, 3])

    # Convolutional Layer #1
    conv1_1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        name="first",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1_1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2
    conv2_1 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        name="second",
        activation=tf.nn.relu
    )

    # Pooling Layer #2
    pool2 = tf.layers.max_pooling2d(inputs=conv2_1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #3

    conv3_1 = tf.layers.conv2d(
        inputs=pool2,
        filters=128,
        kernel_size=[5, 5],
        padding="same",
        name="third",
        activation=tf.nn.relu
    )

    # Pooling Layer #3
    pool3 = tf.layers.max_pooling2d(inputs=conv3_1, pool_size=[2, 2], strides=2)

    pool_flat = tf.reshape(pool3, [-1, 8 * 8 * 128])

    dense = tf.layers.dense(inputs=pool_flat, units=1024, activation=tf.nn.relu)

    dropout = tf.layers.dropout(
        inputs=dense, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout, units=17)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=17)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def TrainAndTest (train_data, test_data, train_labels, test_labels):
    # Load training and eval data
    #mnist = tf.contrib.learn.datasets.load_dataset("mnist")

    # Train data and Train Labels
    #train_data = mnist.train.images  # Returns np.array
    train_data = np.asarray(train_data, dtype=np.float32)
    test_data = np.asarray(test_data, dtype=np.float32)
    train_labels = np.asarray(train_labels, dtype=np.int32)
    test_labels =  np.asarray(test_labels, dtype=np.int32)

    # Test data and Test labels
    #eval_data = mnist.test.images  # Returns np.array
    eval_data = test_data
    eval_labels = np.asarray(test_labels, dtype=np.int32)

    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="/Users/tianzhang/Documents/NU/EECS349/project/data")

    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=10,
        num_epochs=None,
        shuffle=True)
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=4000,
        hooks=[logging_hook])

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


