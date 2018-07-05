"""
.. module:: circuitopt
    :synopsis: Module for optimizing neural circuits

.. moduleauthor:: Marc Javin
"""
import time

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from . import hhmodel
from .circuit import CircuitTf
from .optimize import Optimizer
from .neuronopt import NeuronOpt
from .utils import plots_output_double, plots_output_mult


class CircuitOpt(Optimizer):
    """
    Class for ptimization of a neuron circuit
    """
    dim_batch = 1

    def __init__(self, inits_p, conns, epochs=500, fixed=hhmodel.ALL, dt=0.1):
        self.circuit = CircuitTf(inits_p, conns=conns, fixed=fixed, dt=dt)
        Optimizer.__init__(self, self.circuit, epochs)

    def _build_loss(self, w, n_out):
        """Define how to compute the loss"""
        out, ca = [], []
        for n in n_out:
            out.append(self.res[:, self.circuit._neurons.V_pos, :, n])
            ca.append(self.res[:, self.circuit._neurons.Ca_pos, :, n])
        out = tf.stack(out, axis=2)
        ca = tf.stack(ca, axis=2)
        losses_v = w[0] * tf.square(tf.subtract(out, self.ys_[self.circuit._neurons.V_pos]))
        losses_ca = w[1] * tf.square(tf.subtract(ca, self.ys_[self.circuit._neurons.Ca_pos]))
        losses = losses_v + losses_ca
        self.loss = tf.reduce_mean(losses, axis=[0, 1, 2])

    @staticmethod
    def train_neuron(dir, opt, num, file):
        """train 1 neuron"""
        wv = 0.2
        wca = 0.8
        suffix = 'neuron%s' % num
        file = '%s%s' % (file, num)
        opt.optimize(dir, [wv, wca], epochs=20, suffix=suffix, step=0, file=file)
        for i in range(10):
            wv = 1 - wv
            wca = 1 - wca
            opt.optimize(dir, [wv, wca], reload=True, epochs=20, suffix=suffix, step=i + 1, file=file)

    def opt_neurons(self, file):
        """optimize only neurons 1 by 1"""
        for i in range(self.circuit._neurons.num):
            self.train_neuron('Circuit_0', NeuronOpt(dt=self.circuit._neurons.dt), i, file)

    def opt_circuits(self, subdir, train=None, test=None, suffix='', n_out=[1], w=[1, 0], l_rate=[0.9, 9, 0.95]):
        """optimize circuit parameters"""
        print(suffix)
        T, X, V, Ca = train
        res_targ = np.stack([V, Ca], axis=1)

        yshape = [2, None, None, len(n_out)]

        self._init(subdir, suffix, train, test, l_rate, w, yshape)

        if self._V is None:
            self._V = np.full(self._Ca.shape, -50.)
            w[0] = 0

        self._build_loss(w, n_out)
        self._build_train()
        self.summary = tf.summary.merge_all()

        with tf.Session() as sess:

            self.tdb = tf.summary.FileWriter(self.dir + '/tensorboard',
                                             sess.graph)

            sess.run(tf.global_variables_initializer())
            losses = np.zeros((self._epochs, self.parallel))
            rates = np.zeros(self._epochs)
            vars = dict([(var, [val]) for var, val in self.optimized.init_p.items()])

            shapevars = [self._epochs, self.circuit.n_synapse]
            if self.parallel > 1:
                shapevars.append(self.parallel)
            vars = dict([(var, np.vstack((val, np.zeros(shapevars)))) for var, val in vars.items()])

            for i in tqdm(range(self._epochs)):
                results = self._train_and_gather(sess, i, losses, rates, vars)

                for b in range(self.n_batch):
                    self.circuit._neurons.plot_output(self._T, X[:, b, 0], results[:, :, b, n_out], res_targ[:, :, b, 0],
                                                      suffix='trace%s_%s' % (b, i), show=False, save=True)
                    # plots_output_mult(self._T, self._X[:,n_b], results[:,0,b,:], results[:,-1,b,:], suffix='circuit_%s_trace%s'%(i,n_b), show=False, save=True)

                if i % self._frequency == 0 or i == self._epochs - 1:
                    for b in range(self.n_batch):
                        plots_output_mult(self._T, X[:, b, 0], results[:, self.circuit._neurons.V_pos, b, :], results[:, self.circuit._neurons.Ca_pos, b, :],
                                          suffix='circuit_trace%s_%s' % (b, i), show=False, save=True)

                    self._plots_dump(sess, losses, rates, vars, i)

            with open(self.dir + 'time', 'w') as f:
                f.write(str(time.time() - self.start_time))

        return self.optimized