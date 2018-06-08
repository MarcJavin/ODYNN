import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from Hodghux import Neuron_tf
import tensorflow as tf
import numpy as np
from utils import  plots_output_double, OUT_SETTINGS, OUT_PARAMS, plot_loss_rate,set_dir, plot_vars
from data import get_data_dump, FILE_LV
import pickle
import params
from tqdm import tqdm
import time

SAVE_PATH = 'tmp/model.ckpt'
DUMP_V = 'final_params'


class HH_opt():
    """Full Hodgkin-Huxley Model implemented in Python"""


    def __init__(self, neuron=None, init_p=params.give_rand(), fixed=[], constraints=params.CONSTRAINTS, loop_func=None, dt=0.1):
        if(neuron is not None):
            self.neuron = neuron
        else:
            self.neuron = Neuron_tf(init_p, loop_func=loop_func, dt=dt, fixed=fixed, constraints=constraints)


    def write_settings(self, dir, start_rate, decay_step, decay_rate, w):
        with open('%s%s_%s.txt' % (dir, OUT_SETTINGS, self.suffix), 'w') as f:
            f.write('Nb of neurons : %s' % self.neuron.num + '\n' +
                    'Initial params : %s' % self.neuron.init_p + '\n'+
                    'Fixed variables : %s' % [c for c in self.neuron.fixed] + '\n'+
                    'Initial state : %s' % self.neuron.init_state + '\n' +
                    'Constraints : %s' % self.neuron.constraints_dic + '\n' +
                    'Model solver : %s' % self.neuron.loop_func + '\n' +
                    'Weights (out, cac) : %s' % w + '\n' +
                    'Start rate : %s, decay_step : %s, decay_rate : %s' % (start_rate, decay_step, decay_rate) + '\n')

    """Define how the loss is computed"""
    def build_loss(self, y, w):
        cac = self.res[:, -1]
        out = self.res[:, 0]
        self.loss = 0
        if(self.neuron.num == 1):
            losses_v = w[0] * tf.square(tf.subtract(out, y[0]))
            losses_ca = w[1] * tf.square(tf.subtract(cac, y[-1]))
            self.loss += tf.reduce_mean(losses_v + losses_ca)
        else:
            for i in range(self.neuron.num):
                losses_v = w[0] * tf.square(tf.subtract(out[:,i], y[0]))
                losses_ca = w[1] * tf.square(tf.subtract(cac[:,i], y[-1]))
                self.loss += tf.reduce_mean(losses_v + losses_ca)

    """learning rate and optimization"""
    def build_train(self, start_rate, decay_step, decay_rate):
        global_step = tf.Variable(0, trainable=False)
        # progressive learning rate
        self.learning_rate = tf.train.exponential_decay(
            start_rate,  # Base learning rate.
            global_step,  # Current index to the dataset.
            decay_step,  # Decay step.
            decay_rate,  # Decay rate.
            staircase=True)
        tf.summary.scalar('learning rate', self.learning_rate)
        opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        gvs = opt.compute_gradients(self.loss)
        grads, vars = zip(*gvs)
        # tf.summary.histogram('gradients', gvs)
        # check if nan and clip the values
        # grads, vars = zip(*[(tf.cond(tf.is_nan(grad), lambda: 0., lambda: grad), var) for grad, var in gvs])
        grads_normed, _ = tf.clip_by_global_norm(grads, 5.)
        self.train_op = opt.apply_gradients(zip(grads_normed, vars), global_step=global_step)


    def optimize(self, subdir, w=[1,0], epochs=200, l_rate=[0.9,9,0.9], suffix='', step='', file=None, reload=False):
        print(suffix, step)
        self.suffix = suffix
        DIR = set_dir(subdir+'/')
        tf.reset_default_graph()
        self.neuron.reset()
        start_rate, decay_step, decay_rate = l_rate

        self.write_settings(DIR, start_rate, decay_rate, decay_rate, w)

        if file is None:
            self.T, self.X, self.V, self.Ca = get_data_dump()
        else:
            self.T, self.X, self.V, self.Ca = get_data_dump(file)
        if(self.neuron.loop_func == self.neuron.ik_from_v):
            self.Ca = self.V
        assert(self.neuron.dt == self.T[1] - self.T[0])
        # inputs

        if(self.neuron.num == 1):
            xs_ = tf.placeholder(shape=[None], dtype=tf.float32)
            ys_ = tf.placeholder(shape=[2, None], dtype=tf.float32)
            init_state = tf.placeholder(shape=[len(self.neuron.init_state)], dtype=tf.float32)
        else:
            xs_ = tf.placeholder(shape=[None, self.neuron.num], dtype=tf.float32)
            ys_ = tf.placeholder(shape=[2, None], dtype=tf.float32)
            self.X = np.tile(self.X, (self.neuron.num, 1)).transpose()
            init_state = tf.placeholder(shape=self.neuron.init_state.shape, dtype=tf.float32)

        self.res = tf.scan(self.neuron.step,
                      xs_,
                     initializer=init_state)
        self.build_loss(ys_, w)
        self.build_train(start_rate, decay_step, decay_rate)


        summary = tf.summary.merge_all()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            if(reload):
                """Get variables and measurements from previous steps"""
                saver.restore(sess, '%s%s'%(DIR, SAVE_PATH))
                with open(DIR+FILE_LV, 'rb') as f:
                    l,r,vars = pickle.load(f)
                losses = np.concatenate((l, np.zeros(epochs)))
                rates = np.concatenate((r, np.zeros(epochs)))
                len_prev = len(l)
            else:
                sess.run(tf.global_variables_initializer())
                vars = dict([(var, [val]) for var, val in self.neuron.init_p.items()])
                losses = np.zeros(epochs)
                rates = np.zeros(epochs)
                len_prev = 0

            vars = dict([(var, np.vstack((val, np.zeros((epochs, self.neuron.num))))) for var, val in vars.items()])

            for i in tqdm(range(epochs)):
                results, _, train_loss = sess.run([self.res, self.train_op, self.loss], feed_dict={
                    xs_: self.X,
                    ys_: np.vstack((self.V, self.Ca)),
                    init_state: self.neuron.init_state
                })
                _ = sess.run(self.neuron.constraints)

                with open('%s%s_%s.txt' % (DIR, OUT_PARAMS, suffix), 'w') as f:
                    for name, v in self.neuron.param.items():
                        v_ = sess.run(v)
                        f.write('%s : %s\n' % (name, v_))

                        vars[name][len_prev + i + 1] = v_

                rates[len_prev+i] = sess.run(self.learning_rate)
                losses[len_prev+i] = train_loss
                print('[{}] loss : {}'.format(i, train_loss))

                plots_output_double(self.T, self.X, results[:,0], self.V, results[:,-1], self.Ca, suffix='%s_step%s_%s'%(suffix, step, i + 1), show=False, save=True)
                if(i%10==0 or i==epochs-1):
                    with (open(DIR+FILE_LV, 'wb')) as f:
                        pickle.dump([losses, rates, vars], f)
                    plot_vars(dict([(name, val[:len_prev + i + 1 + 1*(len_prev==0)]) for name,val in vars.items()]), suffix=suffix, show=False, save=True)
                    plot_loss_rate(losses[:len_prev+i+1], rates[:len_prev+i+1], suffix=suffix, show=False, save=True)
                    saver.save(sess, '%s%s' % (DIR, SAVE_PATH))


if __name__ == '__main__':
    runner = HH_opt(l_rate=[0.9,9,0.9])
    runner.loop_func = runner.integ_comp
    runner.Main2('test')
