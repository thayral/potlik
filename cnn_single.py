import os
import numpy as np
import tensorflow as tf
import scipy.io
from matplotlib import pyplot as plt


# this is going to be some CNN stuff
#svhn digit cinvenet
#reverse engineering depuis tensorpack




def get_data_grey(title_of_set):
    #should be train, test or extra
    image_size = 32

    filename = os.path.join(data_dir, title_of_set + '_32x32.mat')
    data = scipy.io.loadmat(filename)
    X_rgb = data['X'].transpose(3, 0, 1, 2)

    X = np.dot(X_rgb[...,:3], [0.299, 0.587, 0.114])
    X=X.reshape((-1, image_size, image_size, 1)).astype(np.float32)
    Y = data['y'].reshape((-1))
    Y[Y == 10] = 0
    return X, Y

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == labels)
          / predictions.shape[0])


def LecunLCN(X, image_shape, threshold=1e-4, radius=7, use_divisor=True):
    """Local Contrast Normalization"""
    """[http://yann.lecun.com/exdb/publis/pdf/jarrett-iccv-09.pdf]"""

    # Get Gaussian filter
    filter_shape = (radius, radius, image_shape[3], 1)

    #self.filters = theano.shared(self.gaussian_filter(filter_shape), borrow=True)
    filters = gaussian_filter(filter_shape)
    X = tf.convert_to_tensor(X, dtype=tf.float32)
    # Compute the Guassian weighted average by means of convolution
    convout = tf.nn.conv2d(X, filters, [1,1,1,1], 'SAME')

    # Subtractive step
    mid = int(np.floor(filter_shape[1] / 2.))

    # Make filter dimension broadcastable and subtract
    centered_X = tf.subtract(X, convout)

    # Boolean marks whether or not to perform divisive step
    if use_divisor:
        # Note that the local variances can be computed by using the centered_X
        # tensor. If we convolve this with the mean filter, that should give us
        # the variance at each point. We simply take the square root to get our
        # denominator

        # Compute variances
        sum_sqr_XX = tf.nn.conv2d(tf.square(centered_X), filters, [1,1,1,1], 'SAME')

        # Take square root to get local standard deviation
        denom = tf.sqrt(sum_sqr_XX)

        per_img_mean = tf.reduce_mean(denom)
        divisor = tf.maximum(per_img_mean, denom)
        # Divisise step
        new_X = tf.truediv(centered_X, tf.maximum(divisor, threshold))
    else:
        new_X = centered_X

    return new_X


def gaussian_filter(kernel_shape):
    x = np.zeros(kernel_shape, dtype = float)
    mid = np.floor(kernel_shape[0] / 2.)

    for kernel_idx in range(0, kernel_shape[2]):
        for i in range(0, kernel_shape[0]):
            for j in range(0, kernel_shape[1]):
                x[i, j, kernel_idx, 0] = gauss(i - mid, j - mid)

    return tf.convert_to_tensor(x / np.sum(x), dtype=tf.float32)

def gauss(x, y, sigma=3.0):
    Z = 2 * np.pi * sigma ** 2
    return  1. / Z * np.exp(-(x ** 2 + y ** 2) / (2. * sigma ** 2))


if __name__ == '__main__': # When we call the script directly ...

    # dataset should be in /data
    #SVHN_URL = "http://ufldl.stanford.edu/housenumbers/"

    current_dir = os.path.dirname(__file__)
    data_dir = os.path.join(current_dir,"data")

    dataset_X, dataset_Y = get_data_grey("train")
    train_dataset = dataset_X[:50000]
    train_labels = dataset_Y[:50000]
    valid_dataset = dataset_X[50000:]
    valid_labels = dataset_Y[50000:]

    test_dataset, test_labels = get_data_grey("test")

    print(valid_dataset.shape)
    print(valid_labels.shape)
    print(test_dataset.shape)
    print(test_labels.shape)
    print(train_dataset.shape)
    print(train_labels.shape)



    #print(test_labels[111])
#    plt.imshow(test_dataset[111], cmap='gray', interpolation='nearest')
#    plt.show()

    #exit(0)


    image_size = 32
    num_labels = 10
    num_channels = 1
    # grayscale

    batch_size = 64
    patch_size = 5
    depth1 = 16
    depth2 = 32
    depth3 = 64
    num_hidden1 = 64
    num_hidden2 = 16
    shape=[batch_size, image_size, image_size, num_channels]

    # Construct a 7-layer CNN.
    # C1: convolutional layer, batch_size x 28 x 28 x 16, convolution size: 5 x 5 x 1 x 16
    # S2: sub-sampling layer, batch_size x 14 x 14 x 16
    # C3: convolutional layer, batch_size x 10 x 10 x 32, convolution size: 5 x 5 x 16 x 32
    # S4: sub-sampling layer, batch_size x 5 x 5 x 32
    # C5: convolutional layer, batch_size x 1 x 1 x 64, convolution size: 5 x 5 x 32 x 64
    # Dropout
    # F6: fully-connected layer, weight size: 64 x 16
    # Output layer, weight size: 16 x 10

    graph = tf.Graph()

with graph.as_default():

  # Input data.
  tf_train_dataset = tf.placeholder(
    tf.float32, shape=(batch_size, image_size, image_size, num_channels))
  tf_train_labels = tf.placeholder(tf.int64, shape=(batch_size))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)

  # Variables.
  layer1_weights = tf.get_variable("W1", shape=[patch_size, patch_size, num_channels, depth1],\
           initializer=tf.contrib.layers.xavier_initializer_conv2d())
  layer1_biases = tf.Variable(tf.constant(1.0, shape=[depth1]), name='B1')
  layer2_weights = tf.get_variable("W2", shape=[patch_size, patch_size, depth1, depth2],\
           initializer=tf.contrib.layers.xavier_initializer_conv2d())
  layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth2]), name='B2')
  layer3_weights = tf.get_variable("W3", shape=[patch_size, patch_size, depth2, num_hidden1],\
           initializer=tf.contrib.layers.xavier_initializer_conv2d())
  layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden1]), name='B3')
  layer4_weights = tf.get_variable("W4", shape=[num_hidden1, num_hidden2],\
           initializer=tf.contrib.layers.xavier_initializer())
  layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden2]), name='B4')
  layer5_weights = tf.get_variable("W5", shape=[num_hidden2, num_labels],\
           initializer=tf.contrib.layers.xavier_initializer())
  layer5_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]), name='B5')

  # Model.
  def model(data, keep_prob, shape):
    LCN = LecunLCN(data, shape)
    conv = tf.nn.conv2d(LCN, layer1_weights, [1,1,1,1], 'VALID', name='C1')
    hidden = tf.nn.relu(conv + layer1_biases)
    lrn = tf.nn.local_response_normalization(hidden)
    sub = tf.nn.max_pool(lrn, [1,2,2,1], [1,2,2,1], 'SAME', name='S2')
    conv = tf.nn.conv2d(sub, layer2_weights, [1,1,1,1], padding='VALID', name='C3')
    hidden = tf.nn.relu(conv + layer2_biases)
    lrn = tf.nn.local_response_normalization(hidden)
    sub = tf.nn.max_pool(lrn, [1,2,2,1], [1,2,2,1], 'SAME', name='S4')
    conv = tf.nn.conv2d(sub, layer3_weights, [1,1,1,1], padding='VALID', name='C5')
    hidden = tf.nn.relu(conv + layer3_biases)
    hidden = tf.nn.dropout(hidden, keep_prob)
    shape = hidden.get_shape().as_list()
    reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
    hidden = tf.nn.relu(tf.matmul(reshape, layer4_weights) + layer4_biases)
    return tf.matmul(hidden, layer5_weights) + layer5_biases

  # Training computation.
  logits = model(tf_train_dataset, 0.9375, shape)
  loss = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits( labels=tf_train_labels,logits=logits))

  # Optimizer.
  #optimizer = tf.train.AdagradOptimizer(0.01).minimize(loss)
  global_step = tf.Variable(0)
  learning_rate = tf.train.exponential_decay(0.05, global_step, 10000, 0.95)
  optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss, global_step=global_step)

  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(model(tf_train_dataset, 1.0, shape))
  valid_prediction = tf.nn.softmax(model(tf_valid_dataset, 1.0, shape))
  test_prediction = tf.nn.softmax(model(tf_test_dataset, 1.0, shape))

  saver = tf.train.Saver()

num_steps = 10001

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print('Initialized')
  for step in range(num_steps):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    batch_labels = train_labels[offset:(offset + batch_size)]
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)

    if (step % 500 == 0):
      print('Minibatch loss at step %d: %f' % (step, l))
      print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
      print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
  save_path = saver.save(session, os.path.join(os.getcwd(),"CNN_1.ckpt"))
  print("Model saved in file: %s" % save_path)
