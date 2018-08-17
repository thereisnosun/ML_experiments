import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def dnn_ready():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
    X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    X_valid, X_train = X_train[:5000], X_train[5000:]
    y_valid, y_train = y_train[:5000], y_train[5000:]

    feature_cols = [tf.feature_column.numeric_column("X", shape=[28 * 28])]
    dnn_clf = tf.estimator.DNNClassifier(hidden_units=[300,100], n_classes=10,
                                         feature_columns=feature_cols)

    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"X": X_train}, y=y_train, num_epochs=40, batch_size=50, shuffle=True)
    dnn_clf.train(input_fn=input_fn)


def neuron_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs + n_neurons)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name="kernel")
        b = tf.Variable(tf.zeros([n_neurons]), name="bias")
        Z = tf.matmul(X, W) + b
        if activation is not None:
            return activation(Z)
        else:
            return Z

dnn_ready()


n_inputs = 28*28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

###Construction phase###
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1",
        activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2",
        activation=tf.nn.relu)
    logits = tf.layers.dense(hidden2, n_outputs, name="outputs")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
        logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

learning_rate = 0.01
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

###Execution phase###
mnist = input_data.read_data_sets("/tmp/data/")

n_epochs = 40
batch_size = 50
#
# with tf.Session() as sess:
#     init.run()
#     for epoch in range(n_epochs):
#         for iteration in range(mnist.train.num_examples // batch_size):
#             X_batch, y_batch = mnist.train.next_batch(batch_size)
#             sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
#         acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
#         acc_val = accuracy.eval(feed_dict={X: mnist.validation.images,
#                                     y: mnist.validation.labels})
#         print(epoch, "Train accuracy:", acc_train, "Val accuracy:", acc_val)
#     save_path = saver.save(sess, "./my_model_final.ckpt")

with tf.Session() as sess:
    saver.restore(sess, "./my_model_final.ckpt")
    X_new_scaled = mnist.train.images # some new images (scaled from 0 to 1)
    Z = logits.eval(feed_dict={X: X_new_scaled})
    y_pred = np.argmax(Z, axis=1)