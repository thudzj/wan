import os
import time
import argparse
import importlib
import tensorflow as tf
import tensorflow.contrib as tc

from visualize import *
import svhn_data

rng = np.random.RandomState(1)
def rescale(mat):
    return np.transpose(np.cast[np.float32]((-127.5 + mat)/127.5),(3,0,1,2))

class WassersteinGAN(object):
    def __init__(self, g_net, d_net, z_sampler, y_sampler, data, model, gpus, d_iters, g_iters, ldir, iw):
        self.model = model
        self.data = data
        self.logdir = ldir
        self.w = iw
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
        self.shape = [32, 32, 3]
        self.x = tf.placeholder(tf.float32, [None, 32, 32, 3], name='x')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.fy = tf.placeholder(tf.float32, [None, self.y_dim], name='fy')
        self.ty = tf.placeholder(tf.float32, [None, self.y_dim], name='ty')
        self.xu = tf.placeholder(tf.float32, [None, 32, 32, 3], name='xu')

        self.x_ = self.g_net(self.fy, self.z)

        self.d = tf.reduce_sum(self.d_net(self.x, reuse=False) * (self.ty*(1+self.w)-self.w), 1)
        self.pred = tf.argmax(self.d_net(self.x), 1)
        self.d_f = tf.reduce_sum(self.d_net(self.x_) * self.fy, 1)
        self.d_t = tf.reduce_sum(self.d_net(self.x_) * (self.fy*(1+self.w)-self.w), 1)

        #self.du = self.d_net(self.xu)
        # weight = self.w
        # weightg = 0.3
        #  - weight*self.d_[:, -1]
        # self.g_loss = -tf.reduce_mean(tf.reduce_sum(self.d_[:, :-1] * (self.fy*(1+weightg)-weightg), 1))
        # self.g_loss = -tf.reduce_mean(tf.reduce_sum(self.d_[:, :-1] * self.fy, 1))
        #self.d_loss_unlabeled = - tf.reduce_mean(tf.reduce_max(self.du[:, :-1], 1)) + tf.reduce_mean(self.du[:, -1])
        # self.d_loss_labeled = - tf.reduce_mean(self.d_[:, -1] - weight*tf.reduce_sum(self.d_[:, :-1], 1)) \
                              # - tf.reduce_mean(tf.reduce_sum(self.d[:, :-1] * (self.ty*(1+weight)-weight), 1) - weight*self.d[:, -1])
        # self.d_loss = self.d_loss_labeled# + self.d_loss_unlabeled * 1.0
        
        self.g_loss = tf.reduce_mean(-self.d_t)
        self.d_loss = tf.reduce_mean(-self.d) + tf.reduce_mean(self.d_f)# - tf.reduce_mean(self.df)

        self.reg = tc.layers.apply_regularization(
            tc.layers.l1_regularizer(2.5e-5),
            weights_list=[var for var in tf.global_variables() if 'weights' in var.name]
        )
        self.g_loss_reg = self.g_loss + self.reg
        self.d_loss_reg = self.d_loss + self.reg

        # RMSPropOptimizer, AdamOptimizer
        optimizer = tf.train.RMSPropOptimizer
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_step = optimizer(learning_rate=5e-5)\
                .minimize(self.d_loss_reg, var_list=self.d_net.vars)
            self.g_step = optimizer(learning_rate=5e-5)\
                .minimize(self.g_loss_reg, var_list=self.g_net.vars)

        with tf.control_dependencies([self.d_step]):
            self.d_step = tf.tuple([tf.assign(var, tf.clip_by_value(var, -0.01, 0.01)) for var in self.d_net.vars])

        gpu_options = tf.GPUOptions()#allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
       
    def dense_to_one_hot(self, labels_dense, num_classes=10):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

    def train(self, batch_size=100, num_batches=500):
        # load SVHN data

        trainx, trainy = svhn_data.load('../data/svhn','train')
        testx, testy = svhn_data.load('../data/svhn','test')
        trainy = self.dense_to_one_hot(trainy)
        testy = self.dense_to_one_hot(testy)
        trainx = rescale(trainx)
        testx = rescale(testx)
        trainx_unl = trainx.copy()
        nr_batches_train = int(trainx.shape[0]/batch_size)
        nr_batches_test = int(np.ceil(float(testx.shape[0])/batch_size))

        # select labeled data
        # inds = rng.permutation(trainx.shape[0])
        # trainx = trainx[inds]
        # trainy = trainy[inds]
        # trainy_one_hot = np.argmax(trainy, 1)
        # txs = []
        # tys = []
        # for j in range(10):
        #     txs.append(trainx[trainy_one_hot==j][:args.count])
        #     tys.append(trainy[trainy_one_hot==j][:args.count])
        # txs = np.concatenate(txs, axis=0)
        # tys = np.concatenate(tys, axis=0)
        # txs = trainx.copy()
        # tys = trainy.copy()
        plt.ion()
        self.sess.run(tf.global_variables_initializer())
        start_time = time.time()
        for t in range(0, num_batches):
            inds = rng.permutation(trainx.shape[0])
            trainx = trainx[inds]
            trainy = trainy[inds]
            # trainx = []
            # trainy = []
            # for tt in range(int(np.ceil(trainx_unl.shape[0]/float(txs.shape[0])))):
                # inds = rng.permutation(txs.shape[0])
                # trainx.append(txs[inds])
                # trainy.append(tys[inds])
            # trainx = np.concatenate(trainx, axis=0)
            # trainy = np.concatenate(trainy, axis=0)
            # trainx_unl = trainx_unl[rng.permutation(trainx_unl.shape[0])]

            dl, gl = 0.0, 0.0
            for tt in range(nr_batches_train): 
                d_iters = self.d_iters
                g_iters = self.g_iters
                if tt == 0 or (t == 0 and tt < 20):
                     d_iters = 100
                for _ in range(0, d_iters):
                    bz = self.z_sampler(batch_size, self.z_dim)
                    bfy = self.y_sampler(batch_size, self.y_dim)

                    _, d_loss_ = self.sess.run([self.d_step, self.d_loss], \
                        feed_dict={ \
                            self.x: trainx[tt*batch_size:(tt+1)*batch_size], \
                            self.ty: trainy[tt*batch_size:(tt+1)*batch_size], \
                            #self.xu: trainx_unl[tt*batch_size:(tt+1)*batch_size], \
                            self.z: bz, self.fy: bfy
                        })
                    dl += d_loss_ / d_iters

                for _ in range(0, g_iters):
                    bz = self.z_sampler(batch_size, self.z_dim)
                    bfy = self.y_sampler(batch_size, self.y_dim)
                    _, g_loss_ = self.sess.run([self.g_step, self.g_loss], feed_dict={self.z: bz, self.fy: bfy, self.x: trainx[tt*batch_size:(tt+1)*batch_size]})
                    gl += g_loss_ / g_iters

                if tt % 100 == 99:
                    print('Epoch[%8d] Iter [%8d] Time [%5.4f] d_loss [%.4f] g_loss [%.4f]' %
                        (t + 1, tt, time.time() - start_time, dl / 100.0, gl / 100.0))
                    dl = 0.0
                    gl = 0.0

            tmpz = self.z_sampler(10, self.z_dim)
            # bfy = self.y_sampler(batch_size, self.y_dim)
            bfy = []
            bz = []
            for i in range(10):
                bfy += [i] * int(10)
                bz.append(tmpz)
            bfy = self.dense_to_one_hot(np.array(bfy))
            bz = np.concatenate(bz, 0)
            bx = self.sess.run(self.x_, feed_dict={self.z: bz, self.fy: bfy})
            bx = np.reshape(bx, [100] + self.shape)
            fig = plt.figure(self.data + '.' + self.model)
            grid_show(fig, (bx + 1) / 2, self.shape)
            if not os.path.exists('./logs/{}/{}'.format(self.data, self.logdir)):
                os.makedirs('./logs/{}/{}'.format(self.data, self.logdir))
            fig.savefig('./logs/{}/{}/{}.png'.format(self.data, self.logdir, t))

            # for _ in range(0, 1000):
            #     bx, bty = self.x_sampler(batch_size)
            #     bz = self.z_sampler(batch_size, self.z_dim)
            #     bfy = self.y_sampler(batch_size, self.y_dim)
            #     #self.sess.run(self.d_clip)
            #     self.sess.run(self.d_step_clf, feed_dict={self.x: bx, self.z: bz, self.fy: bfy, self.ty: bty})

            preds = np.zeros([len(testy)])
            labels = np.zeros([len(testy)])
            for ite in range(nr_batches_test):
                last_ind = np.minimum((ite+1)*batch_size, len(testy))
                first_ind = last_ind - batch_size
                bl_ = self.sess.run(self.pred, feed_dict={self.x: testx[first_ind: last_ind]})
                labels[first_ind: last_ind] = np.argmax(testy[first_ind: last_ind], 1)
                preds[first_ind: last_ind] = bl_
            print('Epoch[%8d] Time [%5.4f] acc [%.4f]' %
                        (t + 1, time.time() - start_time, np.sum(labels==preds)/float(len(testy))))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--data', type=str, default='svhn')
    parser.add_argument('--model', type=str, default='dcgan')
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--logdir', type=str, default='')
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--d', type=int, default=5)
    parser.add_argument('--g', type=int, default=1)
    parser.add_argument('--w', type=float, default=0.0)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    data = importlib.import_module(args.data)
    model = importlib.import_module(args.data + '.' + args.model)
    zs = data.NoiseSampler()
    ys = data.LabelSampler()
    d_net = model.Discriminator()
    g_net = model.Generator()

    wgan = WassersteinGAN(g_net, d_net, zs, ys, args.data, args.model, args.gpus, args.d, args.g, args.logdir, args.w)
    wgan.train()
