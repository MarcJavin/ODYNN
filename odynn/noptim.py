"""
.. module:: neuropt
    :synopsis: Module for optimizing neurons

.. moduleauthor:: Marc Javin
"""

import tensorflow as tf
import numpy as np

from .neuron import BioNeuronTf, NeuronTf
from .optim import Optimizer


class NeuronOpt(Optimizer):
    """Class for optimization of a neuron"""

    def __init__(self, neuron):
        """
        Initializer, takes a NeuronTf object as argument

        Args:
            neuron(:obj:`NeuronTf`): Neuron to be optimized
        """
        self.neuron = neuron
        if not isinstance(neuron, NeuronTf):
            raise TypeError('The neuron attribute should be a NeuronTf instance')
        Optimizer.__init__(self, self.neuron)

    def _build_loss(self, results, ys_, w):
        """Define self._loss

        Args:
          w(list): weights for the voltage and the ions concentrations

        """
        if self._parallel > 1:
            # [time, state, batch, model]
            res = tf.transpose(results, perm=[3, 0, 1, 2])
        else:
            res = results
        with tf.variable_scope('Loss'):
            out = res[..., self.neuron.V_pos, :]
            losses_v = w[0] * tf.square(tf.subtract(out, ys_[0]))
            losses = losses_v
            for ion, pos in self.neuron.ions.items():
                ionc = res[..., pos, :]
                losses += w[pos] * tf.square(tf.subtract(ionc, ys_[pos]))
            # losses = tf.nn.moments(losses, axes=[-1])[1] + tf.reduce_mean(losses, axis=[-1])
        self._loss = tf.reduce_mean(losses, axis=[-2, -1])
        # print(self.loss)
        # self.loss = self.loss[tf.random_uniform([1], 0, self.n_batch, dtype=tf.int32)[0]]  # tf.reduce_mean(losses, axis=[0, 1])

    def plot_out(self, X, results, res_targ, suffix, step, name, i):
        for b in range(self.n_batch):
            res_t = [res_targ[i][:, b] if res_targ[i] is not None else None for i in range(len(res_targ))]
            self.neuron.plot_output(self.neuron.dt*np.arange(len(X)), X[:, b], results[:, :, b], res_t,
                                    suffix='%s_%s%s_%s_%s' % (suffix, name, b, step, i + 1), show=False,
                                    save=True, l=0.7, lt=2)



    def optimize(self, dir, train, test=None, w=(1, 0), epochs=700, l_rate=(0.1, 9, 0.92), suffix='', step=None,
                 reload=False, reload_dir=None, evol_var=True, plot=True):
        """Optimize the neuron parameters

        Args:
          dir(str): path to the directory for the saved files
          train(list of ndarray): list containing [time, input, voltage, ion_concentration] that will be used fitted
            dimensions : - time : [time]
                        - input, voltage and concentration : [time, batch]
          test(list of ndarray): same as train for the dimensions
            These arrays will be used fo testing the model (Default value = None)
          w(list): list of weights for the loss, the first value is for the voltage and the following ones for the ion concentrations
            defined in the model. (Default value = [1, 0]:
          epochs(int): Number of training steps (Default value = 700)
          l_rate(tuple): Parameters for an exponential decreasing learning rate : (start, number of constant steps, exponent)
            (Default value = [0.1, 9, 0.92]:
          suffix(str): suffix for the saved files (Default value = '')
          step:  (Default value = None)
          reload(bool): If True, will reload the graph saved in reload_dir (Default value = False)
          reload_dir(str): The path to the directory of the experience to reload (Default value = None)

        Returns:
            :obj:`NeuronTf`: neuron attribute after optimization

        """
        if self.optimized.groups != None and self.optimized.num > 1:
            return
        yshape = [None, None]
        Optimizer.optimize(self, dir, train, test, w, epochs, l_rate, suffix, step, reload, reload_dir, yshape=yshape,
                           evol_var=evol_var, plot=plot)

