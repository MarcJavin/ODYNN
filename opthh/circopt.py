"""
.. module:: circopt
    :synopsis: Module for optimizing neural circuits

.. moduleauthor:: Marc Javin
"""

import tensorflow as tf

from . import hhmodel
from .circuit import CircuitTf
from .optimize import Optimizer
from .neuropt import NeuronOpt


class CircuitOpt(Optimizer):
    """Class for optimization of a neuron circuit"""
    dim_batch = 1

    def __init__(self, inits_p=None, conns=None, epochs=500, fixed=hhmodel.ALL, dt=0.1, circuit=None):
        """

        Args:
            inits_p:
            conns:
            epochs:
            fixed:
            dt:
            circuit: If not None, all other arguments are ignored
        """
        if circuit is None:
            circuit = CircuitTf(inits_p, conns=conns, fixed=fixed, dt=dt)
        self.circuit = circuit
        Optimizer.__init__(self, self.circuit, epochs)

    def _build_loss(self, w):
        """Define self._loss

        Args:
          w(list): weights for the voltage and the ions concentrations

        """
        out, ca = [], []
        for n in self.n_out:
            out.append(self.res[:, self.circuit.neurons.V_pos, :, n])
            ca.append(self.res[:, self.circuit.neurons.Ca_pos, :, n])
        out = tf.stack(out, axis=2)
        ca = tf.stack(ca, axis=2)
        losses_v = w[0] * tf.square(tf.subtract(out, self.ys_[self.circuit.neurons.V_pos]))
        losses_ca = w[1] * tf.square(tf.subtract(ca, self.ys_[self.circuit.neurons.Ca_pos]))
        losses = losses_v + losses_ca
        self._loss = tf.reduce_mean(losses, axis=[0, 1, 2])

    @staticmethod
    def train_neuron(dir, opt, num, file):
        """train 1 neuron

        Args:
          dir: 
          opt: 
          num: 
          file: 

        Returns:

        """
        wv = 0.2
        wca = 0.8
        suffix = 'neuron%s' % num
        file = '%s%s' % (file, num)
        opt.optimize(dir, (wv, wca), epochs=20, suffix=suffix, step=0, file=file)
        for i in range(10):
            wv = 1 - wv
            wca = 1 - wca
            opt.optimize(dir, (wv, wca), reload=True, epochs=20, suffix=suffix, step=i + 1, file=file)

    def opt_neurons(self, file):
        """optimize only neurons 1 by 1

        Args:
          file: 

        Returns:

        """
        for i in range(self.circuit.neurons.num):
            self.train_neuron('Circuit_0', NeuronOpt(dt=self.circuit.neurons.dt), i, file)

    def plot_out(self, X, results, res_targ, suffix, step, name, i):
        for b in range(self.n_batch):
            res_t = [res_targ[i][:, b] if res_targ[i] is not None else None for i in range(len(res_targ))]
            self.circuit.neurons.plot_output(self._T, X[:, b, 0], results[:, :, b, self.n_out], res_t,
                                              suffix='trace%s%s_%s' % (name, b, i), show=False, save=True)

    def opt_circuits(self, subdir, train=None, test=None, w=(1, 0), epochs=700, l_rate=(0.9, 9, 0.95), suffix='', n_out=[1],):
        """optimize circuit parameters

        Args:
          subdir: 
          train:  (Default value = None)
          test:  (Default value = None)
          w:  (Default value = [1)
          0]: 
          epochs:  (Default value = 700)
          l_rate:  (Default value = [0.9)
          9: 
          0.95]: 
          suffix:  (Default value = '')
          n_out:  (Default value = [1])

        Returns:

        """
        self.n_out = n_out
        yshape = [2, None, None, len(n_out)]
        shapevars = [epochs, self.circuit.n_synapse]
        if self._parallel > 1:
            shapevars.append(self._parallel)
        Optimizer.optimize(self, subdir, train, test, w, epochs, l_rate, suffix, yshape=yshape, shapevars=shapevars)

                    #
                    # for b in range(self.n_batch):
                    #     plots_output_mult(self._T, X[:, b, 0], results[:, self.circuit.neurons.V_pos, b, :], results[:, self.circuit.neurons.Ca_pos, b, :],
                    #                       suffix='circuit_trace%s_%s' % (b, i), show=False, save=True)

