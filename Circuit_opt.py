import numpy as np
from Neuron import HodgkinHuxley
from Circuit import Circuit_tf
from Neuron_opt import HH_opt
import params
from Optimizer import Optimizer
from utils import plots_output_mult, plot_loss_rate, plots_output_double, plot_vars_syn
from data import DUMP_FILE, get_data_dump, FILE_LV, SAVE_PATH
import tensorflow as tf
from tqdm import tqdm
import pickle

class Circuit_opt(Optimizer):

    """
    Optimization of a neuron circuit

    """
    dim_batch = 1

    def __init__(self, inits_p, conns, loop_func=HodgkinHuxley.loop_func, fixed=params.ALL, dt=0.1):
        Optimizer.__init__(self)
        self.circuit = Circuit_tf(inits_p, conns=conns, loop_func=loop_func, fixed=fixed, dt=dt)
        self.optimized = self.circuit


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
            self.train_neuron('Circuit_0', HH_opt(loop_func=self.circuit.neurons.loop_func, dt=self.circuit.neurons.dt), i, file)


    """optimize synapses"""
    def opt_circuits(self, subdir, file=DUMP_FILE, suffix='', epochs=200, n_out=1, w=[1,0], l_rate=[0.9,9,0.9]):
        self.init(subdir, suffix, l_rate, w, circuit=self.circuit)
        self.T, self.X, self.V, self.Ca = get_data_dump(file)


        n_batch = self.X.shape[self.dim_batch]

        # Xshape = [time, n_neuron, n_batch]
        self.xs_ = tf.placeholder(shape=[None, None, self.circuit.neurons.num], dtype=tf.float32, name='in_current')
        self.ys_ = tf.placeholder(shape=[2, None, None], dtype=tf.float32, name='out')
        init_state = self.circuit.neurons.init_state
        initshape = list(init_state.shape)
            # reshape init state
        initshape.insert(0, None)
        self.init_state = np.stack([init_state for _ in range(n_batch)], axis=0)
        self.init_state_ = tf.placeholder(shape=initshape, dtype=tf.float32, name='init_state')

        print('i : ', self.X.shape, 'V : ', self.V.shape, 'init : ', init_state.shape)

        #apply in parallel the batch
        def stepmap(hprev, x):
            lambdaData = (hprev,x)
            func = lambda x: (self.circuit.step(x[0], x[1]), 0)
            return tf.map_fn(func, lambdaData)[0]

        self.res = tf.scan(stepmap,
                      self.xs_,
                      initializer=self.init_state_)

        out = self.res[:, :, 0, n_out]
        losses = tf.square(tf.subtract(out, self.ys_[0]))
        self.loss = tf.reduce_mean(losses)
        self.build_train()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            losses = np.zeros(epochs)
            rates = np.zeros(epochs)
            vars = dict([(var, [val]) for var, val in self.circuit.init_p.items()])

            vars = dict([(var, np.vstack((val, np.zeros((epochs, self.circuit.num))))) for var, val in vars.items()])

            for i in tqdm(range(epochs)):
                results = self.train_and_gather(sess, i, losses, rates, vars)

                for n_b in range(n_batch):
                    plots_output_double(self.T, self.X[:,n_b], results[:,n_b,0,n_out], self.V[:, n_b], results[:,n_b,-1,n_out], self.Ca[:, n_b], suffix='%s_trace%s'%(i,n_b), show=False, save=True)
                    plots_output_mult(self.T, self.X[:,n_b], results[:,n_b,0,:], results[:,n_b,-1,:], suffix='circuit_%s_trace%s'%(i,n_b), show=False, save=True)

                if (i % 10 == 0 or i == epochs - 1):
                    with (open(self.dir + FILE_LV, 'wb')) as f:
                        pickle.dump([losses, rates, vars], f)
                    plot_vars_syn(dict([(name, val[:i + 2]) for name, val in vars.items()]), suffix=suffix,
                                  show=False, save=True)
                    plot_loss_rate(losses[:i + 1], rates[:i + 1], show=False, save=True)
                    saver.save(sess, '%s%s' % (self.dir, SAVE_PATH))