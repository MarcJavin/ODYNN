"""
.. module:: Optimizer
    :synopsis: Module containing classes for optimization with Tensorflow

.. moduleauthor:: Marc Javin
"""

import pickle
import time
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf

from data import FILE_LV, SAVE_PATH, get_data_dump
from utils import OUT_SETTINGS, set_dir, OUT_PARAMS, plot_loss_rate


class Optimized(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def build_graph(self):
        pass

    @abstractmethod
    def settings(self):
        pass

    @staticmethod
    def plot_vars(var_dic, suffix, show, save):
        pass

    def apply_constraints(self, session):
        pass

    def get_params(self):
        return []


class Optimizer(ABC):
    min_loss = 1.

    def __init__(self, optimized):
        self.start_time = time.time()
        self.optimized = optimized
        self.parallel = self.optimized.num

    def _build_train(self):
        """learning rate and optimization"""
        self.global_step = tf.Variable(0, trainable=False)
        # progressive learning rate
        self.learning_rate = tf.train.exponential_decay(
            self.start_rate,  # Base learning rate.
            self.global_step,  # Current index to the dataset.
            self.decay_step,  # Decay step.
            self.decay_rate,  # Decay rate.
            staircase=True)
        # self.learning_rate = 0.1
        tf.summary.scalar('learning rate', self.learning_rate)
        opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        gvs = opt.compute_gradients(self.loss)
        grads, vars = zip(*gvs)
        print(grads)

        if self.parallel > 1:
            grads_normed = []
            for i in range(self.parallel):
                # clip by norm for each parallel model (neuron or circuit)
                gi = [g[..., i] for g in grads]
                # if isinstance(self.optimized, Circuit.Circuit):
                #     #[synapse, model]
                #     gi = [g[:,i] for g in grads]
                # else:
                #     gi = [g[i] for g in grads]
                gi_norm, _ = tf.clip_by_global_norm(gi, 5.)
                grads_normed.append(gi_norm)
            grads_normed = tf.stack(grads_normed)
            # resize to tf format
            try:  # for circuits
                grads_normed = tf.transpose(grads_normed, perm=[1, 2, 0])
                grads_normed = tf.unstack(grads_normed, axis=0)
            except:
                grads_normed = tf.unstack(grads_normed, axis=1)
        else:
            grads_normed, _ = tf.clip_by_global_norm(grads, 5.)
        self.train_op = opt.apply_gradients(zip(grads_normed, vars), global_step=self.global_step)

        self.saver = tf.train.Saver()

    def _init(self, subdir, suffix, file, l_rate, w, yshape):
        """initialize objects to be optimized and write setting in the directory"""
        self.suffix = suffix
        self.dir = set_dir(subdir + '/')
        tf.reset_default_graph()
        self.start_rate, self.decay_step, self.decay_rate = l_rate

        self._T, self._X, self._V, self._Ca = get_data_dump(file)
        assert (self.optimized.dt == self._T[1] - self._T[0])

        self.n_batch = self._X.shape[1]
        self.write_settings(w)

        if self.parallel > 1:
            # add dimension for neurons trained in parallel
            # [time, n_batch, neuron]
            self._X = np.stack([self._X for _ in range(self.parallel)], axis=self._X.ndim)
            self._V = np.stack([self._V for _ in range(self.parallel)], axis=self._V.ndim)
            self._Ca = np.stack([self._Ca for _ in range(self.parallel)], axis=self._Ca.ndim)
            yshape.append(self.parallel)

        self.xs_, self.res = self.optimized.build_graph(batch=self.n_batch)
        self.ys_ = tf.placeholder(shape=yshape, dtype=tf.float32, name='voltage_Ca')

        print('i expected : ', self.xs_.shape)
        print('i : ', self._X.shape, 'V : ', self._V.shape)

    def write_settings(self, w):
        """Write the settings of the optimization in a file"""
        with open(self.dir + OUT_SETTINGS, 'w') as f:
            f.write('Weights (out, cac) : {}'.format(w) + '\n' +
                    'Start rate : {}, decay_step : {}, decay_rate : {}'.format(self.start_rate, self.decay_step,
                                                                               self.decay_rate) + '\n' +
                    'Number of batches : {}'.format(self.n_batch) + '\n' +
                    'Number of time steps : {}'.format(self._T.shape) + 'Input current shape : {}'.format(
                self._X.shape) +
                    'Output voltage shape : {}'.format(self._V.shape) + '\n' +
                    self.optimized.settings())

    def _train_and_gather(self, sess, i, losses, rates, vars):
        """Train the model and collect loss, learn_rate and variables"""
        summ, results, _, train_loss = sess.run([self.summary, self.res, self.train_op, self.loss], feed_dict={
            self.xs_: self._X,
            self.ys_: np.array([self._V, self._Ca])
        })

        self.tdb.add_summary(summ, i)

        self.optimized.apply_constraints(sess)

        with open('{}{}_{}.txt'.format(self.dir, OUT_PARAMS, self.suffix), 'w') as f:
            for name, v in self.optimized.get_params():
                v_ = sess.run(v)
                f.write('{} : {}\n'.format(name, v_))
                vars[name][i + 1] = v_

        rates[i] = sess.run(self.learning_rate)
        losses[i] = train_loss
        if self.parallel > 1:
            train_loss = np.nanmean(train_loss)
        print('[{}] loss : {}'.format(i, train_loss))
        return results

    def _plots_dump(self, sess, losses, rates, vars, i):
        """Plot the variables evolution, the loss and saves it in a file"""
        with (open(self.dir + FILE_LV, 'wb')) as f:
            pickle.dump([losses, rates, vars], f)
        plot_loss_rate(losses[:i + 1], rates[:i + 1], suffix=self.suffix, show=False, save=True)
        self.saver.save(sess, '{}{}'.format(self.dir, SAVE_PATH))
        self.optimized.plot_vars(dict([(name, val[:i + 2]) for name, val in vars.items()]),
                                 suffix=self.suffix + 'evolution', show=False,
                                 save=True)
