import numpy as np
from Neuron import HodgkinHuxley, V_pos, Ca_pos
from Circuit import Circuit_tf
from NeuronOpt import NeuronOpt
import neuron_params
from Optimizer import Optimizer
from utils import plots_output_mult, OUT_SETTINGS, plots_output_double, plot_vars_syn
from data import DUMP_FILE, get_data_dump, FILE_CIRC
import tensorflow as tf
from tqdm import tqdm
import time

class CircuitOpt(Optimizer):

    """
    Optimization of a neuron circuit
    """
    dim_batch = 1

    def __init__(self, inits_p, conns, fixed=neuron_params.ALL, dt=0.1):
        self.circuit = Circuit_tf(inits_p, conns=conns, fixed=fixed, dt=dt)
        Optimizer.__init__(self, self.circuit)


    def build_loss(self, w, n_out):
        out, ca = [], []
        for n in n_out:
            out.append(self.res[:, V_pos, :, n])
            ca.append(self.res[:, Ca_pos, :, n])
        out = tf.stack(out, axis=2)
        ca = tf.stack(ca, axis=2)
        losses_v = w[0] * tf.square(tf.subtract(out, self.ys_[V_pos]))
        losses_ca = w[1] * tf.square(tf.subtract(ca, self.ys_[Ca_pos]))
        losses = losses_v + losses_ca
        self.loss = tf.reduce_mean(losses, axis=[0, 1, 2])


    """train 1 neuron"""
    def train_neuron(self, dir, opt, num, file):
        wv = 0.2
        wca = 0.8
        suffix = 'neuron%s'%num
        file = '%s%s'%(file,num)
        opt.optimize(dir, [wv, wca], epochs=20, suffix=suffix, step=0, file=file)
        for i in range(10):
            wv = 1 - wv
            wca = 1 - wca
            opt.optimize(dir, [wv, wca], reload=True, epochs=20, suffix=suffix, step=i+1, file=file)

    """optimize only neurons 1 by 1"""
    def opt_neurons(self, file):
        for i in range(self.circuit.neurons.num):
            self.train_neuron('Circuit_0', NeuronOpt(dt=self.circuit.neurons.dt), i, file)


    """optimize synapses"""
    def opt_circuits(self, subdir, file=DUMP_FILE, suffix='', epochs=400, n_out=[1], w=[1,0], l_rate=[0.9,9,0.95]):
        print(suffix)
        T, X, V, Ca = get_data_dump(file)

        yshape = [2, None, None, len(n_out)]

        self.init(subdir, suffix, file, l_rate, w, yshape)

        if (self.V is None):
            self.V = np.full(self.Ca.shape, -50.)
            w[0] = 0

        self.build_loss(w, n_out)
        self.build_train()
        self.summary = tf.summary.merge_all()

        with tf.Session() as sess:

            self.tdb = tf.summary.FileWriter(self.dir + '/tensorboard',
                                                 sess.graph)

            sess.run(tf.global_variables_initializer())
            losses = np.zeros((epochs, self.parallel))
            rates = np.zeros(epochs)
            vars = dict([(var, [val]) for var, val in self.optimized.init_p.items()])

            shapevars = [epochs, self.circuit.n_synapse]
            if(self.parallel>1):
                shapevars.append(self.parallel)
            vars = dict([(var, np.vstack((val, np.zeros(shapevars)))) for var, val in vars.items()])

            for i in tqdm(range(epochs)):
                results = self.train_and_gather(sess, i, losses, rates, vars)

                for b in range(self.n_batch):
                    plots_output_double(self.T, X[:,b,0], results[:,V_pos,b,n_out], V[:, b,0], results[:,Ca_pos,b,n_out],
                                        Ca[:, b,0], suffix='trace%s_%s'%(b, i), show=False, save=True)
                    # plots_output_mult(self.T, self.X[:,n_b], results[:,0,b,:], results[:,-1,b,:], suffix='circuit_%s_trace%s'%(i,n_b), show=False, save=True)

                if (i % 10 == 0 or i == epochs - 1):
                    for b in range(self.n_batch):
                        plots_output_mult(self.T, X[:, b,0], results[:, V_pos, b, :], results[:, Ca_pos, b, :],
                                          suffix='circuit_trace%s_%s' % (b, i), show=False, save=True)

                    self.plots_dump(sess, losses, rates, vars, i)

            with open(self.dir + 'time', 'w') as f:
                f.write(str(time.time() - self.start_time))

        return -1