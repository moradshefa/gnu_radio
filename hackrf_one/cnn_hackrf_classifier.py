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

  interval = 2048
  input_layer = tf.reshape(features["x"], [-1, 1, interval, 1])

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[1, 10],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  p1 = 4
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[1, p1], strides=4)

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[1, 5],
      padding="same",
      activation=tf.nn.relu)

  p2 = 4
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[1, p2], strides=4)



  # Dense Layer
  pool2_flat = tf.reshape(pool2, [-1, 2048//p1//p2 * 64])

  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  num_classes = 3
  logits = tf.layers.dense(inputs=dropout, units=num_classes)

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

  ave_pspect_signals, center_freqs = np.load("ave_pspect_signals.npy")
  # ave_pspect_signals = ave_pspect_signals

  # 0.7 percent of data will be used to train, rest to validate
  train_percentage = 0.8


  train_data = []
  test_data = []

  train_labels = []
  test_labels = []


  print()
  signals = []
  interval = 0
  for i, key in enumerate(ave_pspect_signals.keys()):
    num_samples, interval = ave_pspect_signals[key].shape

    num_training_samples = int(num_samples * train_percentage)
    train = ave_pspect_signals[key][:num_training_samples]
    test = ave_pspect_signals[key][num_training_samples:]

    signals.append(key)
    print(i, key, "Number of samples: ", num_samples)
    print("     training on ", train.shape[0], " samples")
    print("     validating on ", test.shape[0], " samples")

    train_data.extend(train)
    train_labels.extend([i]*train.shape[0])
    test_data.extend(test)
    test_labels.extend([i]*test.shape[0])
    interval = test.shape[1]



  train_data = np.array(train_data).astype(np.float32)
  test_data = np.array(test_data).astype(np.float32)

  train_labels = np.array(train_labels).astype(np.int32)
  test_labels = np.array(test_labels).astype(np.int32)

  print("\n------------------------------------------------------------")
  print("interval ", interval)

  print("train_data.shape", train_data.shape)
  print("test_data.shape", test_data.shape)

  print("train_labels.shape", train_labels.shape)
  print("test_labels.shape", test_labels.shape)
  print("------------------------------------------------------------")


  # tensors_to_log = {"probabilities": "softmax_tensor"}
  # logging_hook = tf.train.LoggingTensorHook(
  #     tensors=tensors_to_log, every_n_iter=50)


  # Train the model
  training_steps = 11000
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=60,
      num_epochs=None,
      shuffle=True)
  mnist_classifier.train(
      input_fn=train_input_fn,
      steps=training_steps,
      # hooks=[logging_hook]
      )

  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": test_data},
      y=test_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  
  print("\n------------------------------------------------------------")
  print("Results on Test Data")
  print(eval_results)
  print("------------------------------------------------------------")



  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print("\n------------------------------------------------------------")
  print("Results on Training Data")
  print(eval_results)
  print("------------------------------------------------------------")






  # Sample Code on how to get predictions for all data
  
  # find misclassified data and print its information
  true_vals = []
  all_data = []
  for i, name in enumerate(signals):
    signal = ave_pspect_signals[name]
    true_vals.extend([i]* len(signal))
    all_data.extend(signal)

  all_data = np.array(all_data).astype(np.float32)
  pred_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": all_data},
      # y=eval_labels,
      num_epochs=1,
      shuffle=False)
  pred_results = mnist_classifier.predict(input_fn=pred_input_fn)
  


  print("\n------------------------------------------------------------")
  print("Misclassified points")
  for i, result in enumerate(pred_results):
    pred_class = result["classes"]
    true_class = true_vals[i]
    if true_class != pred_class:
      print(i,"True: ",signals[true_class]," Pred: ",  signals[pred_class], " Probab: ", result["probabilities"][pred_class], " Freq: ", center_freqs[i][1])
  print("------------------------------------------------------------")


# Create the Estimator, save the model and the training
mnist_classifier = tf.estimator.Estimator(
   model_fn=cnn_model_fn, model_dir="/tmp/hackrf2")
    # model_fn=cnn_model_fn)


if __name__ == "__main__":
  tf.app.run()






