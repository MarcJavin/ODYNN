import numpy as np
from Hodghux import Neuron_fix
from HH_opt import HH_opt
import scipy as sp
import params
from utils import plots_output_mult, set_dir, plot_loss_rate
from data import FILE_LV, DUMP_FILE
import pickle
import tensorflow as tf
from tqdm import tqdm

class Circuit():

    """
    neurons : objects to optimize

    """
    def __init__(self, neurons, conns, i_injs, t, i_out=None, init_p=params.SYNAPSE, dt=0.1):
        assert(len(neurons) == i_injs.shape[0])
        self.param = init_p
        self.neurons = neurons
        self.connections = conns
        self.t = t
        self.i_injs = i_injs
        self.i_out = i_out
        self.dt = dt
        self.param = {}
        for (pre,post), p in conns.items():
            for k, v in p.items():
                name = '%s-%s__%s' % (pre, post, k)
                self.param[name] = tf.get_variable(name, initializer=v, dtype=tf.float32)


    """synaptic current"""
    def syn_curr(self, syn, vprev, vpost):
        g = self.param['%s__G'%syn] / (1 + sp.exp((self.param['%s__mdp'%syn] - vprev)/self.param['%s__scale'%syn]))
        i = g*(self.param['%s__E'%syn] - vpost)
        return i

    """run one time step"""
    def step(self, curs):
        next_curs = np.zeros(len(curs))
        #update neurons
        for i, n in enumerate(self.simuls):
            n.step(curs[i])
        #update synapses
        for pre, post in self.connections:
            vprev = self.simuls[pre].state[0]
            vpost = self.simuls[post].state[0]
            next_curs[post] += self.syn_curr('%s-%s'%(pre,post), vprev, vpost)
        return next_curs




    """runs the entire simulation"""
    def run_sim(self, simuls=None, show=True, dump=False):
        if(simuls is None):
            self.simuls = []
            for n in self.neurons:
                self.simuls.append(Neuron_fix(init_p=n.init_p, dt=self.dt))
        else:
            self.simuls = simuls
        states = dict(
            [(i, np.zeros((len(self.neurons[0].init_state), len(self.t)))) for i, n in enumerate(self.neurons)])
        curs = np.zeros(self.i_injs.shape)

        for t in range(self.i_injs.shape[1]):
            if(t == 0):
                curs[:, t] = self.step(self.i_injs[:, t])
            else:
                curs[:,t] = self.step(self.i_injs[:,t] + curs[:,t-1])

            for k,v in states.items():
                states[k][:,t] = self.simuls[k].state

        plots_output_mult(self.t, self.i_injs, [states[0][0, :], states[1][0, :]], [states[0][-1, :], states[1][-1, :]],
                          i_syn=curs, show=show)

        if(dump):
            for i, n in enumerate(self.neurons):
                todump = [self.t, self.i_injs[i,:]+curs[i,:], states[i][0,:], states[i][-1,:]]
                with open(DUMP_FILE+str(i), 'wb') as f:
                    pickle.dump(todump, f)
            return DUMP_FILE


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

    def opt_neurons(self):
        file = self.run_sim([Neuron_fix(), Neuron_fix()], dump=True)
        for i, n in enumerate(self.neurons):
            self.train_neuron('Circuit_0', HH_opt(n), i, file)


    """optimize synapses"""
    def opt_circuits(self, subdir, file, epochs=200, l_rate=[0.9,9,0.9]):
        DIR = set_dir(subdir + '/')
        tf.reset_default_graph()
        for n in self.neurons:
            n.reset()
        start_rate, decay_step, decay_rate = l_rate

        xs_ = tf.placeholder(shape=[None], dtype=tf.float32)
        ys_ = tf.placeholder(shape=[None], dtype=tf.float32)
        init_state = tf.placeholder(shape=[len(self.neurons[0].init_state)], dtype=tf.float32)

        res = tf.scan(self.step,
                      xs_,
                      initializer=init_state)


        out = res[:, 0]
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
        grads, vars = zip(*[(tf.cond(tf.is_nan(grad), lambda: 0., lambda: grad), var) for grad, var in gvs])
        grads_normed, _ = tf.clip_by_global_norm(grads, 5.)
        train_op = opt.apply_gradients(zip(grads_normed, vars), global_step=global_step)

        with tf.Session() as sess:

            losses = np.zeros(epochs)
            rates = np.zeros(epochs)

            for i in tqdm(range(epochs)):
                results, _, train_loss = sess.run([res, train_op, loss], feed_dict={
                    xs_: self.X,
                    ys_: np.vstack((self.V, self.Ca)),
                    init_state: self.neuron.init_state
                })
                _ = sess.run(self.neuron.constraints)

                rates[i] = sess.run(learning_rate)
                losses[i] = train_loss
                print('[{}] loss : {}'.format(i, train_loss))


                plot_loss_rate(losses, rates, show=False, save=True)