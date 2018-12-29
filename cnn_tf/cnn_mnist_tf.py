#!/bin/python3.6

from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

# https://www.tensorflow.org/tutorials/estimators/cnn

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
    """Model function for CNN"""

    # Initialize input layers and convert feature_map to mnist-dataset shape
    input_layers = tf.reshape(features["x"], [-1, 28, 28, 1])

    # Convolutional layer 1 with 32 5x5 filters and activation-function `ReLU`
    conv1 = tf.layers.conv2d(
        inputs=input_layers,
        filters=32,
        kernel_size=5,
        padding="same",
        activation=tf.nn.relu
    )

    # Pooling layer 1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional layer 2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=5,
        # Same means adding 0 values to the edges of the input tensor
        padding="same",
        activation=tf.nn.relu
    )

    # Pooling layer 2
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2, strides=2)

    # Dense layer
    # Flatten so that our tensor has only two dimensions
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        # Dropout will only be added if in TRAIN mode
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN
    )

    # Logits layer which  will return the raw values for our predictions
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        # Our predicted class is the element in the corresponding row of the logits tensor with the highest raw value
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the `logging_hook`
        # Can derive probabilities from our logits layer by applying softmax activation
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    # We compile our predictions in a dict, and return an EstimatorSpec object
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure our model to optimize this loss value during training
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"]
        )
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops
    )


def main(unused_argv):
    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    mnist_classifier = tf.estimator.Estimator(
        # Model_fn specifies the model function
        model_fn=cnn_model_fn, model_dir="/tmp/mnist_covnet_model"
    )

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50
    )

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        # Model will train on minibatches of size 100
        batch_size=100,
        # Model will train until number of specified steps is reached
        num_epochs=None,
        shuffle=True
    )

    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=20000,
        hooks=[logging_hook]
    )

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        # We obviously want to evaluate over one epoch
        num_epochs=1,
        shuffle=False
    )
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
    tf.app.run()