"""
.. module:: neuropt
    :synopsis: Module for optimizing neurons

.. moduleauthor:: Marc Javin
"""

import tensorflow as tf

from .neuron import BioNeuronTf, NeuronTf
from .optimize import Optimizer


class NeuronOpt(Optimizer):
    """Class for optimization of a neuron"""

    yshape = [2, None, None]

    def __init__(self, init_p='random', fixed=(), constraints=None,
                 dt=0.1, neuron=None):
        """
        Initializer, takes a NeuronTf object as argument, or alternatively the parameters to create a BioNeuronTf instance

        Args:
            init_p: parameters for the neuron (Default value = 'random')
            fixed: constant parameters (Default value = ())
            constraints: constraints to be applied (Default value = None)
            dt: time step (Default value = 0.1)
            neuron(:obj:`NeuronTf`): Neuron to be optimized, if not none, other parameters are ignored (Default value = None)
        """
        if neuron is not None:
            self.neuron = neuron
            if not isinstance(neuron, NeuronTf):
                raise TypeError('The neuron attribute should be a NeuronTf instance')
        else:
            self.neuron = BioNeuronTf(init_p, dt=dt, fixed=fixed, constraints=constraints)
        Optimizer.__init__(self, self.neuron)

    def _build_loss(self, w):
        """Define self._loss

        Args:
          w(list): weights for the voltage and the ions concentrations

        """
        with tf.variable_scope('Loss'):
            out = self.res[:, self.neuron.V_pos]
            losses_v = w[0] * tf.square(tf.subtract(out, self.ys_[0]))
            losses = losses_v
            for ion, pos in self.neuron.ions.items():
                ionc = self.res[:, pos]
                losses += w[1] * tf.square(tf.subtract(ionc, self.ys_[-1]))
            self._loss = tf.reduce_mean(losses, axis=[0, 1])
        # print(self.loss)
        # self.loss = self.loss[tf.random_uniform([1], 0, self.n_batch, dtype=tf.int32)[0]]  # tf.reduce_mean(losses, axis=[0, 1])

    def plot_out(self, X, results, res_targ, suffix, step, name, i):
        for b in range(self.n_batch):
            res_t = [res_targ[i][:, b] if res_targ[i] is not None else None for i in range(len(res_targ))]
            self.neuron.plot_output(self._T, X[:, b], results[:, :, b], res_t,
                                    suffix='%s_%s%s_%s_%s' % (suffix, name, b, step, i + 1), show=False,
                                    save=True, l=0.7, lt=2)



    def optimize(self, dir, train, test=None, w=(1, 0), epochs=700, l_rate=(0.1, 9, 0.92), suffix='', step=None,
                 reload=False, reload_dir=None):
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
        shapevars = (epochs, self._parallel)
        Optimizer.optimize(self, dir, train, test, w, epochs, l_rate, suffix, step, reload, reload_dir, yshape=self.yshape, shapevars=shapevars)

