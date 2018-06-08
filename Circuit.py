import numpy as np
from Hodghux import Neuron_tf, HodgkinHuxley
from HH_opt import HH_opt
import params
from utils import plots_output_mult, set_dir, plot_loss_rate, plots_output_double, OUT_SETTINGS
from data import FILE_LV, DUMP_FILE, get_data_dump
import pickle
import tensorflow as tf
from tqdm import tqdm

class Circuit():

    """
    neurons : objects to optimize

    """
    def __init__(self, inits_p, conns, loop_func=HodgkinHuxley.loop_func, i_out=None, dt=0.1):
        self.neurons = Neuron_tf(inits_p, loop_func=loop_func, fixed=params.ALL, dt=dt)
        self.connections = conns
        self.i_out = i_out
        syns = zip(*[k for k in conns.iterkeys()])
        self.pres = np.array(syns[0], dtype=np.int32)
        self.posts = np.array(syns[1], dtype=np.int32)
        self.syns = ['%s-%s' % (a,b) for a,b in zip(self.pres, self.posts)]
        self.reset()

    """build graph variables"""
    def reset(self):
        self.param = {}
        for k in self.connections.values()[0].iterkeys():
            self.param[k] = tf.get_variable(k, initializer=[p[k] for n, p in self.connections.items()], dtype=tf.float32)

    """synaptic current"""
    def syn_curr(self, vprev, vpost):
        G = self.param['G']
        mdp = self.param['mdp']
        scale = self.param['scale']
        g = G * tf.sigmoid((vprev - mdp) / scale)
        i = g * (self.param['E'] - vpost)
        return i

    """run one time step"""
    def step(self, hprev, x):
        # update synapses
        # curs = tf.Variable(initial_value=np.zeros(self.neurons.num), trainable=False, dtype=tf.float32)
        idx_pres = np.vstack((np.zeros(self.pres.shape, dtype=np.int32), self.pres)).transpose()
        idx_post = np.vstack((np.zeros(self.pres.shape, dtype=np.int32), self.posts)).transpose()
        vpres = tf.gather_nd(hprev, idx_pres)
        vposts = tf.gather_nd(hprev, idx_post)
        curs_syn = self.syn_curr(vpres, vposts)
        curs_post = []
        for i in range(self.neurons.num):
            if i not in self.posts:
                curs_post.append(0.)
                continue
            curs_post.append(tf.reduce_sum(tf.gather(curs_syn, np.argwhere(self.posts==i))))

        # update neurons
        h = self.neurons.step(hprev, x + tf.stack(curs_post))
        return h

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
        for i in range(self.neurons.num):
            self.train_neuron('Circuit_0', HH_opt(loop_func=self.neurons.loop_func, dt=self.neurons.dt), i, file)


    def write_settings(self, dir, start_rate, decay_step, decay_rate, w):
        with open('%s%s.txt' % (dir, OUT_SETTINGS), 'w') as f:
            f.write('Nb of neurons : %s' % self.neurons.num + '\n' +
                    'Connections : \n %s \n %s' % (self.pres, self.posts) + '\n'+
                    'Initial synaptic params : %s' % self.connections + '\n' +
                    'Initial neuron params : %s' % self.neurons.init_p + '\n'+
                    'Fixed variables : %s' % [c for c in self.neurons.fixed] + '\n'+
                    'Initial state : %s' % self.neurons.init_state + '\n' +
                    'Constraints : %s' % self.neurons.constraints_dic + '\n' +
                    'Model solver : %s' % self.neurons.loop_func + '\n' +
                    'Weights (out, cac) : %s' % w + '\n' +
                    'Start rate : %s, decay_step : %s, decay_rate : %s' % (start_rate, decay_step, decay_rate) + '\n')


    """optimize synapses"""
    def opt_circuits(self, subdir, file, epochs=200, w=[1,0], l_rate=[0.9,9,0.9]):
        DIR = set_dir(subdir + '/')
        self.T, self.X, self.V, self.Ca = get_data_dump(file)

        tf.reset_default_graph()
        self.neurons.reset()
        self.reset()
        start_rate, decay_step, decay_rate = l_rate
        self.write_settings(DIR, start_rate, decay_step, decay_rate, w)

        xs_ = tf.placeholder(shape=[None, self.neurons.num], dtype=tf.float32, name='in_current')
        ys_ = tf.placeholder(shape=[2, None], dtype=tf.float32, name='out')
        init_state = tf.placeholder(shape=self.neurons.init_state.shape, dtype=tf.float32, name='init_state')

        res = tf.scan(self.step,
                      xs_,
                      initializer=init_state)

        out = res[:, 0, 1]
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
                    init_state: self.neurons.init_state
                })
                _ = sess.run(self.neurons.constraints)


                rates[i] = sess.run(learning_rate)
                losses[i] = train_loss
                print('[{}] loss : {}'.format(i, train_loss))

                plots_output_double(self.T, self.X, results[:,0,1], self.V, results[:,-1,1], self.Ca, suffix=i, show=False, save=True)
                plots_output_mult(self.T, self.X, results[:,0,:], results[:,-1,:], suffix='circuit_%s'%i, show=False, save=True)
                plot_loss_rate(losses, rates, show=False, save=True)