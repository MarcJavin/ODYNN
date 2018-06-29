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

import hhmodel
from datas import SAVE_PATH, FILE_LV
from model import Ca_pos, V_pos
from neuron import NeuronTf
from optimize import Optimizer
from utils import plots_output_double


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
        cac = self.res[:, Ca_pos]
        out = self.res[:, V_pos]
        losses_v = w[0] * tf.square(tf.subtract(out, self.ys_[0]))
        losses_ca = w[1] * tf.square(tf.subtract(cac, self.ys_[-1]))
        losses = losses_v + losses_ca
        self.loss = tf.reduce_mean(losses, axis=[0, 1])

    def optimize(self, subdir, train=None, test=None, w=[1, 0], l_rate=[0.1, 9, 0.92], suffix='', step=None,
                 reload=False):
        """Optimize the neuron parameters"""
        print(suffix, step)

        T, X, V, Ca = train

        yshape = [2, None, None]

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
                self.saver.restore(sess, '%s%s' % (self.dir, SAVE_PATH))
                with open(self.dir + FILE_LV, 'rb') as f:
                    l, r, vars = pickle.load(f)
                losses = np.concatenate((l, losses))
                rates = np.concatenate((r, rates))
                len_prev = len(l)
            else:
                vars = dict([(var, [val]) for var, val in self.optimized.init_p.items()])
                len_prev = 0

            vars = dict([(var, np.vstack((val, np.zeros((self._epochs, self.parallel))))) for var, val in vars.items()])

            for i in tqdm(range(self._epochs)):
                results = self._train_and_gather(sess, len_prev + i, losses, rates, vars)

                # if losses[len_prev+i]<self.min_loss:
                #     self.plots_dump(sess, losses, rates, vars, len_prev + i)
                #     return i+len_prev

                for b in range(self.n_batch):
                    plots_output_double(self._T, X[:, b], results[:, V_pos, b], V[:, b], results[:, Ca_pos, b],
                                        self._Ca[:, b], suffix='%s_trace%s_%s_%s' % (suffix, b, step, i + 1), show=False,
                                        save=True, l=0.7, lt=2)

                if i % self._frequency == 0 or i == self._epochs - 1:
                    self._plots_dump(sess, losses, rates, vars, len_prev + i)

            with open(self.dir + 'time', 'w') as f:
                f.write(str(time.time() - self.start_time))

        return -1