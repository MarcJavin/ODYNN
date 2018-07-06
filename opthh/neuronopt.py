"""
.. module:: neuronopt
    :synopsis: Module for optimizing neurons

.. moduleauthor:: Marc Javin
"""

import tensorflow as tf

from . import hhmodel
from .neuron import NeuronTf
from .optimize import Optimizer


class NeuronOpt(Optimizer):
    """
    Class for optimization of a neuron
    """

    dim_batch = 1
    yshape = [2, None, None]

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
            cac = self.res[:, self.neuron.Ca_pos]
            out = self.res[:, self.neuron.V_pos]
            losses_v = w[0] * tf.square(tf.subtract(out, self.ys_[0]))
            losses_ca = w[1] * tf.square(tf.subtract(cac, self.ys_[-1]))
            losses = losses_v + losses_ca
            self.loss = tf.reduce_mean(losses, axis=[0,1])
        # print(self.loss)
        # self.loss = self.loss[tf.random_uniform([1], 0, self.n_batch, dtype=tf.int32)[0]]  # tf.reduce_mean(losses, axis=[0, 1])

    def plot_out(self, X, results, res_targ, suffix, step, name, i):
        for b in range(self.n_batch):
            res_t = [res_targ[i][:, b] if res_targ[i] is not None else None for i in range(len(res_targ))]
            self.neuron.plot_output(self._T, X[:, b], results[:, :, b], res_t,
                                    suffix='%s_%s%s_%s_%s' % (suffix, name, b, step, i + 1), show=False,
                                    save=True, l=0.7, lt=2)



    def optimize(self, subdir, train=None, test=None, w=[1, 0], epochs=700, l_rate=[0.1, 9, 0.92], suffix='', step=None,
                 reload=False, reload_dir=None):
        """Optimize the neuron parameters"""
        shapevars = (epochs, self.parallel)
        Optimizer.optimize(self, subdir, train, test, w, epochs, l_rate, suffix, step, reload, reload_dir, yshape=self.yshape, shapevars=shapevars)

