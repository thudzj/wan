import os
import time
import argparse
import importlib
import tensorflow as tf
import tensorflow.contrib as tc

from visualize import *
from tensorflow.examples.tutorials.mnist import input_data

rng = np.random.RandomState(1)

def gf_gan(v):
    return -tf.log(1 + tf.exp(-v))

def f_star_gan(t):
    return -tf.log(1 - tf.exp(t) + 1e-8)

def gf_kl(v):
    return v

def f_star_kl(t):
    return tf.exp(t - 1)

def gf_rkl(v):
    return -tf.exp(v)

def f_star_rkl(t):
    return -1 - tf.log(-t)

def gf_gs(v):
    return np.log(2.0) - tf.log(1 + tf.exp(-v))

def f_star_gs(t):
    return -tf.log(2.0 - tf.exp(t))

def gf_pearson(v):
    return v

def f_star_pearson(t):
    return 0.25 * tf.square(t) + t

def gf_wgan(v):
    return v

def f_star_wgan(t):
    return t

gfs = {'gan': gf_gan, 'kl': gf_kl, 'rkl': gf_rkl, 'gs': gf_gs, 'pearson': gf_pearson, 'wgan': gf_wgan}
f_stars = {'gan': f_star_gan, 'kl': f_star_kl, 'rkl': f_star_rkl, 'gs': f_star_gs, 'pearson': f_star_pearson, 'wgan': f_star_wgan}

class WassersteinGAN(object):
    def __init__(self, g_net, d_net, z_sampler, y_sampler, data, model, gf, f_star, gpus, d_iters, g_iters, freg, ldir):
        self.model = model
        self.data = data
        self.gf = gf
        self.freg = freg
        self.logdir = ldir
        self.f_star = f_star
        self.gpus = gpus
        self.g_net = g_net
        self.d_net = d_net
        self.d_iters = d_iters
        self.g_iters = g_iters
        self.z_sampler = z_sampler
        self.y_sampler = y_sampler
        self.x_dim = self.d_net.x_dim
        self.z_dim = self.g_net.z_dim
        self.y_dim = self.d_net.y_dim
        self.shape = [28, 28, 1]
        self.x = tf.placeholder(tf.float32, [None, self.x_dim], name='x')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.fy = tf.placeholder(tf.float32, [None, self.y_dim], name='fy')
        self.ty = tf.placeholder(tf.float32, [None, self.y_dim], name='ty')
        self.xu = tf.placeholder(tf.float32, [None, self.x_dim], name='xu')

        self.x_ = self.g_net(tf.concat(1, [self.z, self.fy]))

        self.d = self.d_net(self.x, reuse=False)
        self.d_ = self.d_net(self.x_)
        self.du = self.d_net(self.xu)

        self.g_loss = -tf.reduce_mean(tf.reduce_sum(self.d_[:, :-1] * (self.fy*1.1-0.1), 1))
        self.d_loss_unlabeled = - tf.reduce_mean(tf.reduce_max(self.du[:, :-1], 1)) + tf.reduce_mean(self.du[:, -1]) - tf.reduce_mean(tf.reduce_sum(tf.nn.softmax(self.du[:, :-1]) * tf.nn.log_softmax(self.du[:, :-1]), 1))
        # self.d_loss_unlabeled = - tf.reduce_mean(tf.nn.softmax(self.du[:, :-1]) * tf.nn.log_softmax(self.du[:, :-1]))
        self.d_loss_labeled = -tf.reduce_mean(self.d_[:, -1]) - tf.reduce_mean(tf.reduce_sum(self.d[:, :-1] * self.ty, 1))
        # self.d_loss_clf = - tf.reduce_mean(tf.reduce_sum(self.d * self.ty + self.d_ * self.fy, 1))
        self.d_loss = self.d_loss_labeled + self.d_loss_unlabeled * 1.0
        
        self.d_loss += tf.reduce_mean(self.d[:, -1])+ tf.reduce_mean(tf.reduce_max(self.d_[:, :-1], 1)) - tf.reduce_mean(tf.reduce_sum(tf.nn.softmax(self.d[:, :-1]) * tf.nn.log_softmax(self.d[:, :-1]), 1)) 
        # self.g_loss += -tf.reduce_mean(tf.reduce_sum(tf.nn.softmax(self.d_[:, :-1]) * tf.nn.log_softmax( self.d_[:, :-1]), 1))
        if self.freg:
            print("Add reg...")
            self.g_loss += tf.reduce_mean(self.d_[:, -1])
            self.d_loss += tf.reduce_mean(self.d[:, -1] + self.du[:, -1]) # + tf.reduce_max(self.d_[:, :-1], 1))

        self.reg = tc.layers.apply_regularization(
            tc.layers.l1_regularizer(2.5e-5),
            weights_list=[var for var in tf.global_variables() if 'weights' in var.name]
        )
        self.g_loss_reg = self.g_loss + self.reg
        self.d_loss_reg = self.d_loss + self.reg

        # RMSPropOptimizer, AdamOptimizer
        optimizer = tf.train.RMSPropOptimizer
        self.d_step = optimizer(learning_rate=5e-5)\
            .minimize(self.d_loss_reg, var_list=self.d_net.vars)
        self.g_step = optimizer(learning_rate=5e-5)\
            .minimize(self.g_loss_reg, var_list=self.g_net.vars)
        # self.d_step_clf = optimizer(learning_rate=5e-5)\
        #    .minimize(self.d_loss_clf, var_list=self.d_net.vars)

        self.d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in self.d_net.vars]
        gpu_options = tf.GPUOptions()#allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
       
    def dense_to_one_hot(self, labels_dense, num_classes=10):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

    def train(self, batch_size=100, num_batches=300):
        mnist = input_data.read_data_sets('../data/mnist', one_hot = True, validation_size = 0)
        trainx = mnist.train._images
        trainx_unl = trainx.copy()
        trainy = mnist.train._labels.astype(np.int32)
        testx = mnist.test._images
        testy = mnist.test._labels.astype(np.int32)

        # select labeled data
        inds = rng.permutation(trainx.shape[0])
        trainx = trainx[inds]
        trainy = trainy[inds]
        trainy_one_hot = np.argmax(trainy, 1)
        txs = []
        tys = []
        for j in range(10):
            txs.append(trainx[trainy_one_hot==j][:args.count])
            tys.append(trainy[trainy_one_hot==j][:args.count])
        txs = np.concatenate(txs, axis=0)
        tys = np.concatenate(tys, axis=0)
        nr_batches_train = int(trainx.shape[0]/batch_size)
        nr_batches_test = int(testx.shape[0]/batch_size)
        plt.ion()
        self.sess.run(tf.global_variables_initializer())
        start_time = time.time()
        for t in range(0, num_batches):

            trainx = []
            trainy = []
            for tt in range(trainx_unl.shape[0]/txs.shape[0]):
                inds = rng.permutation(txs.shape[0])
                trainx.append(txs[inds])
                trainy.append(tys[inds])
            trainx = np.concatenate(trainx, axis=0)
            trainy = np.concatenate(trainy, axis=0)
            trainx_unl = trainx_unl[rng.permutation(trainx_unl.shape[0])]

            dl, gl = 0.0, 0.0
            for tt in range(nr_batches_train): 
                d_iters = self.d_iters
                g_iters = self.g_iters
                if tt == 0:# or (t == 0 and tt < 20):
                     d_iters = 100
                for _ in range(0, d_iters):
                    bz = self.z_sampler(batch_size, self.z_dim)
                    bfy = self.y_sampler(batch_size, self.y_dim)
                    _, d_loss_ = self.sess.run([self.d_step, self.d_loss], feed_dict={self.x: trainx[tt*batch_size:(tt+1)*batch_size], \
                        self.ty: trainy[tt*batch_size:(tt+1)*batch_size], self.z: bz, self.fy: bfy, \
                        self.xu: trainx_unl[tt*batch_size:(tt+1)*batch_size]})
                    self.sess.run(self.d_clip)
                    dl += d_loss_ / d_iters

                for _ in range(0, g_iters):
                    bz = self.z_sampler(batch_size, self.z_dim)
                    bfy = self.y_sampler(batch_size, self.y_dim)
                    _, g_loss_ = self.sess.run([self.g_step, self.g_loss], feed_dict={self.z: bz, self.fy: bfy})
                    gl += g_loss_ / g_iters

            bz = self.z_sampler(batch_size, self.z_dim)
            # bfy = self.y_sampler(batch_size, self.y_dim)
            bfy = []
            for i in range(self.y_dim):
                bfy += [i] * int(batch_size / self.y_dim)
            bfy = self.dense_to_one_hot(np.array(bfy))
            bx = self.sess.run(self.x_, feed_dict={self.z: bz, self.fy: bfy})
            bx = np.reshape(bx, [batch_size] + self.shape)
            fig = plt.figure(self.data + '.' + self.model)
            grid_show(fig, bx, self.shape)
            if not os.path.exists('./logs/{}/{}'.format(self.data, self.logdir)):
                os.makedirs('./logs/{}/{}'.format(self.data, self.logdir))
            fig.savefig('./logs/{}/{}/{}.pdf'.format(self.data, self.logdir, t))

            # for _ in range(0, 1000):
            #     bx, bty = self.x_sampler(batch_size)
            #     bz = self.z_sampler(batch_size, self.z_dim)
            #     bfy = self.y_sampler(batch_size, self.y_dim)
            #     #self.sess.run(self.d_clip)
            #     self.sess.run(self.d_step_clf, feed_dict={self.x: bx, self.z: bz, self.fy: bfy, self.ty: bty})

            preds = np.zeros([nr_batches_test * batch_size])
            labels = np.zeros([nr_batches_test * batch_size])
            for ite in range(nr_batches_test):
                bl_ = self.sess.run(self.d, feed_dict={self.x: testx[ite * batch_size: (ite + 1) * batch_size]})
                labels[ite * batch_size: (ite + 1) * batch_size] = np.argmax(testy[ite * batch_size: (ite + 1) * batch_size], 1)
                preds[ite * batch_size: (ite + 1) * batch_size] = np.argmax(bl_[:, :-1], 1)
            print('Iter [%8d] Time [%5.4f] d_loss [%.4f] g_loss [%.4f] test_acc [%.6f]' %
                    (t + 1, time.time() - start_time, dl / nr_batches_train, gl / nr_batches_train, np.sum(preds == labels) / float(preds.shape[0])))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--data', type=str, default='mnist')
    parser.add_argument('--model', type=str, default='dcgan')
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--func', type=str, default='wgan')
    parser.add_argument('--logdir', type=str, default='')
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--d', type=int, default=1)
    parser.add_argument('--g', type=int, default=1)
    parser.add_argument('--reg', type=int, default=0)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    data = importlib.import_module(args.data)
    model = importlib.import_module(args.data + '.' + args.model)
    zs = data.NoiseSampler()
    ys = data.LabelSampler()
    d_net = model.Discriminator()
    g_net = model.Generator()
    gf = gfs[args.func]
    f_star = f_stars[args.func]

    wgan = WassersteinGAN(g_net, d_net, zs, ys, args.data, args.model, gf, f_star, args.gpus, args.d, args.g, args.reg == 1, args.logdir)
    wgan.train()
