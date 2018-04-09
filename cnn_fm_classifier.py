# first attempt at classifier

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# Our application logic will be added here

def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer

  input_layer = tf.reshape(features["x"], [-1, 1, 2048, 1])

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[1, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[1, 4], strides=4)

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[1, 5],
      padding="same",
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[1, 4], strides=4)



  # Dense Layer
  pool2_flat = tf.reshape(pool2, [-1, 2048//4//4 * 64])



  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=2)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


# Set up logging for predictions
  # tensors_to_log = {"probabilities": "softmax_tensor"}
  # logging_hook = tf.train.LoggingTensorHook(
  #     tensors=tensors_to_log, every_n_iter=50)


def main(unused_argv):
  # Load training and eval data
  valid_fm, invalid_fm = np.load("data/fm_data.npy")

  valid_fm_train = valid_fm[:30]
  invalid_fm_train = invalid_fm[:30]

  valid_fm_test = valid_fm[30:]
  invalid_fm_test = invalid_fm[30:]

  training_data = np.vstack((valid_fm_train, invalid_fm_train))
  testing_data = np.vstack((valid_fm_test, invalid_fm_test))

  training_labels = np.vstack((np.ones((valid_fm_train.shape[0], 1)), np.zeros((invalid_fm_train.shape[0], 1))))
  testing_labels = np.vstack((np.ones((valid_fm_test.shape[0], 1)), np.zeros((invalid_fm_test.shape[0], 1))))



  train_data = training_data.astype(np.float32)
  train_labels = training_labels.astype(np.int32)
  eval_data = testing_data.astype(np.float32)
  eval_labels = testing_labels.astype(np.int32)

  # tensors_to_log = {"probabilities": "softmax_tensor"}
  # logging_hook = tf.train.LoggingTensorHook(
  #     tensors=tensors_to_log, every_n_iter=50)


  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=60,
      num_epochs=None,
      shuffle=True)
  mnist_classifier.train(
      input_fn=train_input_fn,
      steps=10000,
      # hooks=[logging_hook]
      )

  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)



  # Sample Code on how to get predictions for all data
  
  # all_data = np.vstack((valid_fm, invalid_fm)).astype(np.float32)
  # pred_input_fn = tf.estimator.inputs.numpy_input_fn(
  #     x={"x": all_data},
  #     # y=eval_labels,
  #     num_epochs=1,
  #     shuffle=False)
  # pred_results = mnist_classifier.predict(input_fn=pred_input_fn)
  
  # for i, result in enumerate(pred_results):
  #   print(i, result)


# Create the Estimator, save the model and the training
mnist_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir="/tmp/fm_convnet_model1")


if __name__ == "__main__":
  tf.app.run()






