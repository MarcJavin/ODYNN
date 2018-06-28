"""
.. module:: neuronopt
    :synopsis: Module for optimizing neurons

.. moduleauthor:: Marc Javin
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from opthh.neuron import NeuronTf, V_pos, Ca_pos
from opthh.optimize import Optimizer
import tensorflow as tf
import numpy as np
from opthh.utils import plots_output_double
from data import get_data_dump, FILE_LV, DUMP_FILE, SAVE_PATH
import pickle
from opthh import neuron_params
from tqdm import tqdm
import time


class NeuronOpt(Optimizer):
    """
    Optimization of a neuron
    """

    dim_batch = 1

    def __init__(self, neuron=None, init_p=neuron_params.give_rand(), fixed=[], constraints=neuron_params.CONSTRAINTS,
                 dt=0.1):
        if neuron is not None:
            self.neuron = neuron
        else:
            self.neuron = NeuronTf(init_p, dt=dt, fixed=fixed, constraints=constraints)
        Optimizer.__init__(self, self.neuron)

    def _build_loss(self, w):
        """Define how the loss is computed"""
        cac = self.res[:, Ca_pos]
        out = self.res[:, V_pos]
        losses_v = w[0] * tf.square(tf.subtract(out, self.ys_[0]))
        losses_ca = w[1] * tf.square(tf.subtract(cac, self.ys_[-1]))
        losses = losses_v + losses_ca
        self.loss = tf.reduce_mean(losses, axis=[0, 1])

    def optimize(self, subdir, w=[1, 0], epochs=500, l_rate=[0.1, 9, 0.92], suffix='', step=None, file=DUMP_FILE,
                 reload=False):
        print(suffix, step)
        T, X, V, Ca = get_data_dump(file)

        yshape = [2, None, None]

        self._init(subdir, suffix, file, l_rate, w, yshape)

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
            losses = np.zeros((epochs, self.parallel))
            rates = np.zeros(epochs)

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

            vars = dict([(var, np.vstack((val, np.zeros((epochs, self.parallel))))) for var, val in vars.items()])

            for i in tqdm(range(epochs)):
                results = self._train_and_gather(sess, len_prev + i, losses, rates, vars)

                # if losses[len_prev+i]<self.min_loss:
                #     self.plots_dump(sess, losses, rates, vars, len_prev + i)
                #     return i+len_prev

                for b in range(self.n_batch):
                    plots_output_double(self._T, X[:, b], results[:, V_pos, b], V[:, b], results[:, Ca_pos, b],
                                        self._Ca[:, b], suffix='%s_trace%s_%s_%s' % (suffix, b, step, i + 1), show=False,
                                        save=True, l=0.7, lt=2)

                if i % 10 == 0 or i == epochs - 1:
                    self._plots_dump(sess, losses, rates, vars, len_prev + i)

            with open(self.dir + 'time', 'w') as f:
                f.write(str(time.time() - self.start_time))

        return -1


if __name__ == '__main__':
    pass
