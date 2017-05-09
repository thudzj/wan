import numpy as np

class NoiseSampler(object):
    def __call__(self, batch_size, z_dim):
        return np.random.uniform(-1.0, 1.0, [batch_size, z_dim])

class LabelSampler(object):
    def __call__(self, batch_size, y_dim = 10):
        return np.random.multinomial(1, [1/float(y_dim)]*int(y_dim), size=(batch_size))