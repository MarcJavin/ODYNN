"""
.. module:: circopt
    :synopsis: Module for optimizing neural circuits

.. moduleauthor:: Marc Javin
"""

import tensorflow as tf
import numpy as np

from .optim import Optimizer
from .noptim import NeuronOpt


class CircuitOpt(Optimizer):
    """Class for optimization of a neuronal circuit"""

    def __init__(self, circuit):
        """

        Args:
            circuit (:obj:`CircuitTf`): Circuit to be optimized
        """
        self.circuit = circuit
        self.w_n = None
        Optimizer.__init__(self, self.circuit)

    def settings(self, w, train):
        return 'Neural weights : %s'%self.w_n + '\n' + Optimizer.settings(self,w,train)

    def _build_loss(self, results, ys_, w):
        """Define self._loss

        Args:
          w(list): weights for the voltage and the ions concentrations

        """
        # [time, state, batch, neuron, model]
        res = []
        for n in self.n_out:
            res.append(results[:,:,:,n])
        res = tf.stack(res, axis=0)
        if self._parallel > 1:
            res = tf.transpose(res, perm=[4, 1, 2, 3, 0])
        else:
            res = tf.transpose(res, perm=[1, 2, 3, 0])
        out = res[..., self.circuit.neurons.V_pos, :, :]
        losses_v = w[0] * tf.square(tf.subtract(out, ys_[self.circuit.neurons.V_pos]))
        losses = losses_v
        for ion, pos in self.optimized._neurons.ions.items():
            ionc = res[..., pos, :, :]
            losses += w[pos] * tf.square(tf.subtract(ionc, ys_[pos]))
        # losses = tf.nn.moments(losses, axes=[-1])[1] + tf.reduce_mean(losses, axis=[-1])
        if self.w_n is not None:
            losses = losses * self.w_n
        self._loss = tf.reduce_mean(losses, axis=[-1, -2, -3])

    def plot_out(self, X, results, res_targ, suffix, step, name, i):
        for b in range(self.n_batch):
            res_t = [res_targ[i][:, b] if res_targ[i] is not None else None for i in range(len(res_targ))]
            self.circuit.plot_output(self.circuit.dt*np.arange(len(X)), X[:, b, 0], results[:, :, b, self.n_out], res_t,
                                    suffix='trace%s%s_%s' % (name, b, i), show=False, save=True, l=0.8, lt=1.5)

    def optimize(self, subdir, train=None, test=None, w=(1, 0), w_n=None, epochs=700, l_rate=(0.9, 9, 0.95), suffix='',
                 n_out=[1], evol_var=True, plot=True):
        """Optimize the neuron parameters

        Args:
          dir(str): path to the directory for the saved files
          train(list of ndarray): list containing [time, input, voltage, ion_concentration] that will be used fitted
            dimensions : - time : [time]
                        - input, voltage and concentration : [time, batch, neuron]
          test(list of ndarray): same as train for the dimensions
            These arrays will be used fo testing the model (Default value = None)
          w(list): list of weights for the loss, the first value is for the voltage and the following ones for the ion concentrations
            defined in the model. (Default value = [1, 0]:
          epochs(int): Number of training steps (Default value = 700)
          l_rate(tuple): Parameters for an exponential decreasing learning rate : (start, number of constant steps, exponent)
            (Default value = [0.1, 9, 0.92]:
          suffix(str): suffix for the saved files (Default value = '')
          n_out(list of int): list of neurons corresponding to the data in train and test

        Returns:
            NeuronTf: neuron attribute after optimization

        """
        self.circuit.plot(show=False, save=True)
        self.w_n = w_n
        self.n_out = n_out
        yshape = [None, None, len(n_out)]
        print('yshape', yshape)
        Optimizer.optimize(self, subdir, train, test, w, epochs, l_rate, suffix, yshape=yshape, evol_var=evol_var, plot=plot)
