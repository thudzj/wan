import numpy as np
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('../data/mnist', one_hot = True, validation_size = 0)


# class DataSampler(object):
#     def __init__(self):
#         self.shape = [28, 28, 1]
#         self.num_test_examples = mnist.test.num_examples
#         print mnist.test.num_examples, mnist.train.num_examples

#     def __call__(self, batch_size):
#         return mnist.train.next_batch(batch_size)

#     def data2img(self, data):
#         return np.reshape(data, [data.shape[0]] + self.shape)

#     def test_batch(self, batch_size):
#         return mnist.test.next_batch(batch_size)


class NoiseSampler(object):
    def __call__(self, batch_size, z_dim):
        return np.random.uniform(-1.0, 1.0, [batch_size, z_dim])

class LabelSampler(object):
    def __call__(self, batch_size, y_dim = 10):
        return np.random.multinomial(1, [1/float(y_dim)]*int(y_dim), size=(batch_size))