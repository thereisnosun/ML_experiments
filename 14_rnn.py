import tensorflow as tf
import numpy as np

n_inputs = 3
n_neurons = 5

def manual_rnn():
    x0 = tf.placeholder(tf.float32, [None, n_inputs])
    x1 = tf.placeholder(tf.float32, [None, n_inputs])

    Wx = tf.Variable(tf.random_normal(shape=[n_inputs, n_neurons], dtype=tf.float32))
    Wy = tf.Variable(tf.random_normal(shape=[n_neurons, n_neurons], dtype=tf.float32))

    b = tf.Variable(tf.zeros([1, n_neurons], dtype=tf.float32))

    Y0 = tf.tanh(tf.matmul(x0, Wx) + b)
    Y1 = tf.tanh(tf.matmul(Y0, Wy) + tf.matmul(x1, Wx) + b)

    init = tf.global_variables_initializer()

    X0_batch = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 0, 1]])
    X1_batch = np.array([[9, 8, 7], [0, 0, 0], [6, 5, 4], [3, 2, 1]])

    with tf.Session() as sess:
        init.run()
        Y0_val, Y1_val = sess.run([Y0, Y1], feed_dict={x0: X0_batch, x1: X1_batch})
        print(Y0_val)
        print(Y1_val)


n_steps = 2


def tf_stat_rnn():
    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    X_seqs = tf.unstack(tf.transpose(X, perm=[1, 0, 2]))

    basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
    output_seq, states = tf.contrib.rnn.static_rnn(basic_cell, X_seqs, dtype=tf.float32)

    outputs = tf.transpose(tf.stack(output_seq), perm=[1, 0, 2])

    init = tf.global_variables_initializer()
    X_batch = np.array([
        # t = 0      t = 1
        [[0, 1, 2], [9, 8, 7]],  # instance 1
        [[3, 4, 5], [0, 0, 0]],  # instance 2
        [[6, 7, 8], [6, 5, 4]],  # instance 3
        [[9, 0, 1], [3, 2, 1]],  # instance 4
    ])

    with tf.Session() as sess:
        init.run()
        outputs_eval = outputs.eval(feed_dict={X: X_batch})
        print(outputs_eval)


def tf_dynamic_rnn():
    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])

    basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
    seq_length = tf.placeholder(tf.int32, [None])
    output, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32, sequence_length=seq_length)

    init = tf.global_variables_initializer()

    X_batch = np.array([
        # step 0     step 1
        [[0, 1, 2], [9, 8, 7]],  # instance 1
        [[3, 4, 5], [0, 0, 0]],  # instance 2 (padded with zero vectors)
        [[6, 7, 8], [6, 5, 4]],  # instance 3
        [[9, 0, 1], [3, 2, 1]],  # instance 4
    ])
    seq_length_batch = np.array([2, 1, 2, 2])

    with tf.Session() as sess:
        init.run()
        output_val, state_val = sess.run([output, states], feed_dict={X: X_batch, seq_length: seq_length_batch})
        print(output_val)
        print(state_val)

def rnn_mnist():
    n_steps = 28
    n_inputs = 28
    n_neurons = 150
    n_outputs = 10
    learning_rate = 0.001
    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    y = tf.placeholder(tf.int32, [None])
    basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
    outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)
    logits = tf.layers.dense(states, n_outputs)
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                              logits=logits)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    init = tf.global_variables_initializer()

    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("/tmp/data/")
    X_test = mnist.test.images.reshape((-1, n_steps, n_inputs))
    y_test = mnist.test.labels

    n_epochs = 100
    batch_size = 150
    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for iteration in range(mnist.train.num_examples // batch_size):
                X_batch, y_batch = mnist.train.next_batch(batch_size)
                X_batch = X_batch.reshape((-1, n_steps, n_inputs))
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
            print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)


#tf_dynamic_rnn()
rnn_mnist()