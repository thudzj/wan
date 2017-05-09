import tensorflow as tf
import tensorflow.contrib.layers as tcl


from layers import *


class Discriminator(object):
    def __init__(self):
        self.x_dim = 32*32*3
        self.y_dim = 10
        self.name = 'svhn/dcgan/d_net'

    def __call__(self, x, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()

            conv1 = tcl.convolution2d(
                x, 64, [4, 4], [2, 2],
                activation_fn=tf.identity)
            conv1 = leaky_relu(conv1)

            conv2 = tcl.convolution2d(
                conv1, 128, [4, 4], [2, 2],
                activation_fn=tf.identity)
            conv2 = leaky_relu(tcl.batch_norm(conv2))

            conv2 = tcl.convolution2d(
                conv2, 256, [4, 4], [2, 2],
                activation_fn=tf.identity)
            conv2 = leaky_relu(tcl.batch_norm(conv2))
            
            conv3 = tcl.convolution2d(
                conv2, 512, [4, 4], [2, 2],
                activation_fn=tf.identity)
            conv3 = leaky_relu(tcl.batch_norm(conv3))
            conv3 = tcl.flatten(conv3)

            fc1 = tcl.fully_connected(conv3, self.y_dim, activation_fn=tf.identity)
            return fc1
    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class Generator(object):
    def __init__(self):
        self.z_dim = 100
        self.x_dim = 1024*3
        self.name = 'svhn/dcgan/g_net'

    def __call__(self, y, z):
        with tf.variable_scope(self.name) as vs:
            bs = tf.shape(z)[0]
            fc = tcl.fully_connected(tf.concat([z,y], 1), 4 * 4 * 512, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm)
            conv1 = tf.reshape(fc, [bs, 4, 4, 512])
            conv2 = tcl.conv2d_transpose(
                conv1, 256, [5, 5], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm
            )
            conv3 = tcl.conv2d_transpose(
                conv2, 128, [5, 5], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm
            )
            # conv4 = tcl.conv2d_transpose(
            #     conv3, 64, [4, 4], [2, 2],
            #     weights_initializer=tf.random_normal_initializer(stddev=0.02),
            #     activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm
            # )
            conv5 = tcl.conv2d_transpose(
                conv3, 3, [5, 5], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.tanh
            )
            return conv5

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class Generator_s(object):
    def __init__(self):
        self.z_dim = 100
        self.x_dim = 1024*3
        self.name = 'svhn/dcgan/g_net'

    def __call__(self, y, z, reuse = False):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            bs = tf.shape(z)[0]
            # y_fc1 = tcl.fully_connected(
            #     y, 1024,
            #     weights_initializer=tf.random_normal_initializer(stddev=0.02),
            #     activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm)
            y_fc2 = tcl.fully_connected(
                y, 8 * 8 * 64,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm)
            y_fc2 = tf.reshape(y_fc2, tf.stack([bs, 8, 8, 64])) 

            # z_fc1 = tcl.fully_connected(
            #     z, 1024,
            #     weights_initializer=tf.random_normal_initializer(stddev=0.02),
            #     activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm)
            z_fc2 = tcl.fully_connected(
                z, 8 * 8 * 64,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm)
            z_fc2 = tf.reshape(z_fc2, tf.stack([bs, 8, 8, 64])) 

            conv1 = tf.concat([y_fc2, z_fc2], 3)
            conv1 = tcl.convolution2d_transpose(
                conv1, 64, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm
            )

            conv1 = tcl.convolution2d_transpose(
                conv1, 32, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm
            )

            conv2 = tcl.convolution2d_transpose(
                conv1, 3, [4, 4], [1, 1],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.tanh)
            #conv2 = tf.reshape(conv2, tf.stack([bs, 784]))
            return conv2
    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]