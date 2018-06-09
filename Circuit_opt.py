import numpy as np
from Hodghux import HodgkinHuxley
from Circuit import Circuit_tf
from HH_opt import HH_opt
import params
from utils import plots_output_mult, set_dir, plot_loss_rate, plots_output_double, OUT_SETTINGS
from data import get_data_dump
import tensorflow as tf
from tqdm import tqdm

class Circuit_opt():

    """
    neurons : objects to optimize

    """
    def __init__(self, inits_p, conns, loop_func=HodgkinHuxley.loop_func, fixed=params.ALL, dt=0.1):
        self.circuit = Circuit_tf(inits_p, conns=conns, loop_func=loop_func, fixed=fixed, dt=dt)


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


    def write_settings(self, dir, start_rate, decay_step, decay_rate, w):
        with open('%s%s.txt' % (dir, OUT_SETTINGS), 'w') as f:
            f.write('Nb of neurons : %s' % self.circuit.neurons.num + '\n' +
                    'Connections : \n %s \n %s' % (self.circuit.pres, self.circuit.posts) + '\n'+
                    'Initial synaptic params : %s' % self.circuit.connections + '\n' +
                    'Initial neuron params : %s' % self.circuit.neurons.init_p + '\n'+
                    'Fixed variables : %s' % [c for c in self.circuit.neurons.fixed] + '\n'+
                    'Initial state : %s' % self.circuit.neurons.init_state + '\n' +
                    'Constraints : %s' % self.circuit.neurons.constraints_dic + '\n' +
                    'Model solver : %s' % self.circuit.neurons.loop_func + '\n' +
                    'Weights (out, cac) : %s' % w + '\n' +
                    'Start rate : %s, decay_step : %s, decay_rate : %s' % (start_rate, decay_step, decay_rate) + '\n')


    """optimize synapses"""
    def opt_circuits(self, subdir, file, epochs=200, n_out=1, w=[1,0], l_rate=[0.9,9,0.9]):
        DIR = set_dir(subdir + '/')
        self.T, self.X, self.V, self.Ca = get_data_dump(file)

        tf.reset_default_graph()
        self.circuit.neurons.reset()
        self.circuit.reset()
        start_rate, decay_step, decay_rate = l_rate
        self.write_settings(DIR, start_rate, decay_step, decay_rate, w)

        xs_ = tf.placeholder(shape=[None, self.circuit.neurons.num], dtype=tf.float32, name='in_current')
        ys_ = tf.placeholder(shape=[2, None], dtype=tf.float32, name='out')
        init_state = tf.placeholder(shape=self.circuit.neurons.init_state.shape, dtype=tf.float32, name='init_state')

        res = tf.scan(self.circuit.step,
                      xs_,
                      initializer=init_state)

        out = res[:, 0, n_out]
        losses = tf.square(tf.subtract(out, ys_[0]))
        loss = tf.reduce_mean(losses)

        global_step = tf.Variable(0, trainable=False)
        # progressive learning rate
        learning_rate = tf.train.exponential_decay(
            start_rate,  # Base learning rate.
            global_step,  # Current index to the dataset.
            decay_step,  # Decay step.
            decay_rate,  # Decay rate.
            staircase=True)
        tf.summary.scalar('learning rate', learning_rate)
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        gvs = opt.compute_gradients(loss)
        tf.summary.histogram('gradients', gvs)
        # check if nan and clip the values
        grads, vars = zip(*gvs)
        # grads, vars = zip(*[(tf.cond(tf.is_nan(grad), lambda: 0., lambda: grad), var) for grad, var in gvs])
        grads_normed, _ = tf.clip_by_global_norm(grads, 5.)
        train_op = opt.apply_gradients(zip(grads_normed, vars), global_step=global_step)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            losses = np.zeros(epochs)
            rates = np.zeros(epochs)

            for i in tqdm(range(epochs)):
                results, _, train_loss = sess.run([res, train_op, loss], feed_dict={
                    xs_: self.X,
                    ys_: np.vstack((self.V, self.Ca)),
                    init_state: self.circuit.neurons.init_state
                })
                _ = sess.run(self.circuit.neurons.constraints)


                rates[i] = sess.run(learning_rate)
                losses[i] = train_loss
                print('[{}] loss : {}'.format(i, train_loss))

                plots_output_double(self.T, self.X, results[:,0,n_out], self.V, results[:,-1,n_out], self.Ca, suffix=i, show=False, save=True)
                plots_output_mult(self.T, self.X, results[:,0,:], results[:,-1,:], suffix='circuit_%s'%i, show=False, save=True)
                plot_loss_rate(losses[:i+1], rates[:i+1], show=False, save=True)