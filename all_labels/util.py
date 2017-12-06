import numpy as np
import scipy.io
import gzip
import cPickle
import matplotlib as mlp

mlp.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
import matplotlib as mlp
mlp.use('Agg')


def isolate_class(l):
    class_a_idx = []
    for i in range(len(l)):
        if (l[i] == 0 or l[i] == 1 or l[i] == 2 or l[i] == 3 or l[i] == 4 or l[i] == 5 or l[i] == 6 or l[i] == 7 or l[
            i] == 8 or l[i] == 9):
            class_a_idx.append(i)
    return class_a_idx


def vectorized_result(j,n=10):
    e = np.zeros(n)
    e[j] = 1
    return e

def vectorized_result_svhn(j, n=10):
    e = np.zeros(n)
    e[j-1] = 1
    return e

''' no need to isolate data any more
tr_label_idx = isolate_class(training_labels)
tr_data = [training_data[tr_label_idx[d]] for d in range(len(tr_label_idx))]
tr_labels = [training_labels[tr_label_idx[d]] for d in range(len(tr_label_idx))]
tr_labels = [vectorized_result(d) for d in tr_labels]
'''


'''
val_label_idx = isolate_class(validation_labels)
val_data = [validation_data[val_label_idx[d]] for d in range(len(val_label_idx))]
val_labels = [validation_labels[val_label_idx[d]] for d in range(len(val_label_idx))]
val_labels = [vectorized_result(d) for d in val_labels]
'''


'''
tst_label_idx = isolate_class(test_labels)
tst_data = [test_data[tst_label_idx[d]] for d in range(len(tst_label_idx))]
tst_labels = [test_labels[tst_label_idx[d]] for d in range(len(tst_label_idx))]
tst_labels = [vectorized_result(d) for d in tst_labels]
'''
# print 'shape of training data : ', np.array(training_data).shape
# print 'shape of training labels : ', len(training_labels)

def pickle_mnist():
    f = gzip.open('MNIST/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)



def rap_mnist():
    tr_d, va_d, te_d = pickle_mnist()
    training_data = [np.reshape(x, (28, 28, 1)) for x in tr_d[0]]
    training_labels = [y for y in tr_d[1]]
    validation_data = [np.reshape(x, (28, 28, 1)) for x in va_d[0]]
    test_data = [np.reshape(x, (28, 28, 1)) for x in te_d[0]]
    return training_data, training_labels, validation_data, va_d[1], test_data, te_d[1]


def plot(samples):
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(10, 10)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def load_SVHN(file):

    return scipy.io.loadmat(file)


def lrelu(x, alpha=0.2):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)


def huber_loss(labels, predictions, delta=1.0):
    # this function is direct implementation from https://github.com/gitlimlab/SSGAN-Tensorflow/blob/master/ops.py


    residual = tf.abs(predictions - labels)

    condition = tf.less(residual, delta)
    small_res = 0.5 * tf.square(residual)
    large_res = delta * residual - 0.5 * tf.square(delta)
    return tf.where(condition, small_res, large_res)


def group_labels(data, num=100):
    # this function is to limit the number of labels that are used in ssl
    # it returns the indexes according the labels
    # data is an array of labels
    # num is the number of labels needed per class

    labels = np.unique(data)
    co_l = []

    for l in labels:
        el_l = [np.where(data == l)]
        co_l.append(np.array(el_l).flatten()[:num])
    return co_l

