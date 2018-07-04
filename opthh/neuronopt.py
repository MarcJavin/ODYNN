"""
.. module:: neuronopt
    :synopsis: Module for optimizing neurons

.. moduleauthor:: Marc Javin
"""
import pickle
import time

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from . import hhmodel
from .model import Ca_pos, V_pos
from .neuron import NeuronTf
from .optimize import Optimizer, SAVE_PATH, FILE_LV
from .utils import plots_output_double
from .config import RES_DIR


class NeuronOpt(Optimizer):
    """
    Class for optimization of a neuron
    """

    dim_batch = 1

    def __init__(self, neuron=None, epochs=500, init_p=hhmodel.give_rand(), fixed=[], constraints=hhmodel.CONSTRAINTS,
                 dt=0.1):
        if neuron is not None:
            self.neuron = neuron
        else:
            self.neuron = NeuronTf(init_p, dt=dt, fixed=fixed, constraints=constraints)
        Optimizer.__init__(self, self.neuron, epochs)

    def _build_loss(self, w):
        """Define how the loss is computed"""
        with tf.variable_scope('Loss'):
            cac = self.res[:, Ca_pos]
            out = self.res[:, V_pos]
            losses_v = w[0] * tf.square(tf.subtract(out, self.ys_[0]))
            losses_ca = w[1] * tf.square(tf.subtract(cac, self.ys_[-1]))
            losses = losses_v + losses_ca
            self.loss = tf.reduce_mean(losses, axis=[0,1])
        # print(self.loss)
        # self.loss = self.loss[tf.random_uniform([1], 0, self.n_batch, dtype=tf.int32)[0]]  # tf.reduce_mean(losses, axis=[0, 1])

    def optimize(self, subdir, train=None, test=None, w=[1, 0], l_rate=[0.1, 9, 0.92], suffix='', step=None,
                 reload=False, reload_dir=None):
        """Optimize the neuron parameters"""
        print(suffix, step)

        T, X, V, Ca = train
        if test is not None:
            T_test, X_test, V_test, Ca_test = test

        yshape = [2, None, None]

        if reload_dir is None:
            reload_dir = subdir
        self._init(subdir, suffix, train, test, l_rate, w, yshape)


        if self._V is None:
            self._V = np.full(self._Ca.shape, -50.)
            w[0] = 0

        self._build_loss(w)
        self._build_train()
        self.summary = tf.summary.merge_all()

        with tf.Session() as sess:

            self.tdb = tf.summary.FileWriter(self.dir + '/tensorboard',
                                             sess.graph)

            sess.run(tf.global_variables_initializer())
            losses = np.zeros((self._epochs, self.parallel))
            rates = np.zeros(self._epochs)

            if reload:
                """Get variables and measurements from previous steps"""
                self.saver.restore(sess, '%s/%s' % (RES_DIR + reload_dir, SAVE_PATH))
                with open(RES_DIR + reload_dir + '/' + FILE_LV, 'rb') as f:
                    l, self._test_losses, r, vars = pickle.load(f)
                losses = np.concatenate((l, losses))
                rates = np.concatenate((r, rates))
                sess.run(tf.assign(self.global_step, 200))
                len_prev = len(l)
            else:
                vars = {var : [val] for var, val in self.optimized.init_p.items()}
                len_prev = 0

            vars = dict([(var, np.vstack((val, np.zeros((self._epochs, self.parallel))))) for var, val in vars.items()])

            for i in tqdm(range(self._epochs)):
                results = self._train_and_gather(sess, len_prev + i, losses, rates, vars)

                # if losses[len_prev+i]<self.min_loss:
                #     self.plots_dump(sess, losses, rates, vars, len_prev + i)
                #     return i+len_prev

                for b in range(self.n_batch):
                    plots_output_double(self._T, X[:, b], results[:, V_pos, b], V[:, b], results[:, Ca_pos, b],
                                        Ca[:, b], suffix='%s_train%s_%s_%s' % (suffix, b, step, i + 1), show=False,
                                        save=True, l=0.7, lt=2)

                if i % self._frequency == 0 or i == self._epochs - 1:
                    res_test = self._plots_dump(sess, losses, rates, vars, len_prev + i)
                    if res_test is not None:
                        for b in range(self.n_batch):
                            plots_output_double(self._T, X_test[:, b], res_test[:, V_pos, b], V_test[:, b], res_test[:, Ca_pos, b],
                                                Ca_test[:, b], suffix='%s_test%s_%s_%s' % (suffix, b, step, i + 1),
                                                show=False,
                                                save=True, l=0.7, lt=2)

            with open(self.dir + 'time', 'w') as f:
                f.write(str(time.time() - self.start_time))

        return -1
