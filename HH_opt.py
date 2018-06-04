import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from Hodghux import HodgkinHuxley
import tensorflow as tf
import numpy as np
from utils import  plots_output_double, OUT_SETTINGS, OUT_PARAMS, plot_loss_rate,set_dir, plot_vars
from data import get_data_dump
import pickle
import params
from tqdm import tqdm
import time

SAVE_PATH = 'tmp/model.ckpt'
FILE_LV = 'tmp/dump_lossratevars'


class HH_opt(HodgkinHuxley):
    """Full Hodgkin-Huxley Model implemented in Python"""


    def __init__(self, init_p=params.PARAMS_RAND, fixed=[], constraints=params.CONSTRAINTS, epochs=200, l_rate=[0.9,9,0.9], loop_func=None):
        HodgkinHuxley.__init__(self, init_p, tensors=True, loop_func=loop_func)
        self.fixed = fixed
        self.epochs = epochs
        self.start_rate, self.decay_step, self.decay_rate = l_rate
        self.constraints_dic = constraints
        self.init_p = init_p

    def condition(self, hprev, x, dt, index, mod):
        return tf.less(index * dt, self.DT)


    def step_long(self, hprev, x):
        index = tf.constant(0.)
        h = tf.while_loop(self.condition, self.loop_func, (hprev, x, self.dt, index, self))
        return h

    def step(self, hprev, x):
        return self.loop_func(hprev, x, self.DT, self)


    def Main(self, subdir, w=[1,0], sufix='', file=None, reload=False):
        print(sufix)
        DIR = set_dir(subdir+'/')
        self.param = {}
        self.constraints = []
        tf.reset_default_graph()
        for var, val in self.init_p.items():
            if (var in self.fixed):
                self.param[var] = tf.constant(val, name=var, dtype=tf.float32)
            else:
                self.param[var] = tf.get_variable(var, initializer=val, dtype=tf.float32)
                if var in self.constraints_dic:
                    con = self.constraints_dic[var]
                    self.constraints.append(
                        tf.assign(self.param[var], tf.clip_by_value(self.param[var], con[0], con[1])))

        with open('%s%s_%s.txt' % (DIR, OUT_SETTINGS, sufix), 'w') as f:
            f.write('Initial params : %s' % self.init_p + '\n'+
                    'Fixed variables : %s' % [c for c in self.fixed] + '\n'+
                    'Initial state : %s' % self.init_state + '\n' +
                    'Constraints : %s' % self.constraints_dic + '\n' +
                    'Model solver : %s' % self.loop_func + '\n' +
                    'Weights (out, cac) : %s' % w + '\n' +
                    'Start rate : %s, decay_step : %s, decay_rate : %s' % (self.start_rate, self.decay_step, self.decay_rate) + '\n')

        if file is None:
            self.T, self.X, self.V, self.Ca = get_data_dump()
        else:
            self.T, self.X, self.V, self.Ca = get_data_dump(file)
        if(self.loop_func == self.ik_from_v):
            self.Ca = self.V
        self.DT = params.DT
        # inputs
        xs_ = tf.placeholder(shape=[None], dtype=tf.float32)
        ys_ = tf.placeholder(shape=[2, None], dtype=tf.float32)
        init_state = tf.placeholder(shape=[len(self.init_state)], dtype=tf.float32)

        res = tf.scan(self.step,
                      xs_,
                     initializer=init_state)

        cac = res[:, -1]
        out = res[:, 0]
        losses_v = w[0]*tf.square(tf.subtract(out, ys_[0]))
        losses_ca = w[1]*tf.square(tf.subtract(cac, ys_[-1]))
        loss = tf.reduce_mean(losses_v + losses_ca)

        global_step = tf.Variable(0, trainable=False)
        #progressive learning rate
        learning_rate = tf.train.exponential_decay(
            self.start_rate,  # Base learning rate.
            global_step,  # Current index to the dataset.
            self.decay_step,  # Decay step.
            self.decay_rate,  # Decay rate.
            staircase=True)
        tf.summary.scalar('learning rate', learning_rate)
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        gvs = opt.compute_gradients(loss)
        tf.summary.histogram('gradients', gvs)
        #check if nan and clip the values
        grads, vars = zip(*[(tf.cond(tf.is_nan(grad), lambda : 0., lambda : grad), var) for grad, var in gvs])
        grads_normed, _ = tf.clip_by_global_norm(grads, 5.)
        train_op = opt.apply_gradients(zip(grads_normed, vars), global_step=global_step)

        summary = tf.summary.merge_all()
        saver = tf.train.Saver()




        with tf.Session() as sess:
            if(reload):
                """Get variables and measurements from previous steps"""
                saver.restore(sess, '%s%s'%(DIR, SAVE_PATH))
                with open(DIR+FILE_LV, 'rb') as f:
                    l,r,vars = pickle.load(f)
                losses = np.concatenate((l, np.zeros(self.epochs)))
                rates = np.concatenate((r, np.zeros(self.epochs)))
                len_prev = len(l)
            else:
                sess.run(tf.global_variables_initializer())
                vars = dict([(var, [val]) for var, val in self.init_p.items()])
                losses = np.zeros(self.epochs)
                rates = np.zeros(self.epochs)
                len_prev = 0

            vars = dict([(var, np.concatenate((val, np.zeros(self.epochs)))) for var, val in vars.items()])

            for i in tqdm(range(self.epochs)):
                results, _, train_loss = sess.run([res, train_op, loss], feed_dict={
                    xs_: self.X,
                    ys_: np.vstack((self.V, self.Ca)),
                    init_state: self.init_state
                })
                _ = sess.run(self.constraints)

                with open('%s%s_%s.txt' % (DIR, OUT_PARAMS, sufix), 'w') as f:
                    for name, v in self.param.items():
                        v_ = sess.run(v)
                        vars[name][len_prev + i ] = v_
                        f.write('%s : %s\n' % (name, v_))

                rates[len_prev+i] = sess.run(learning_rate)
                losses[len_prev+i] = train_loss
                print('[{}] loss : {}'.format(i, train_loss))

                plots_output_double(self.T, self.X, results[:,0], self.V, results[:,-1], self.Ca, suffix='%s_%s'%(sufix, i + 1), show=False, save=True)
                if(i%10==0 or i==self.epochs-1):
                    with (open(DIR+FILE_LV, 'wb')) as f:
                        pickle.dump([losses, rates, vars], f)
                    plot_vars(vars, len_prev+i, show=False, save=True)
                    plot_loss_rate(losses, rates, len_prev+i, show=False, save=True)
                    saver.save(sess, '%s%s' % (DIR, SAVE_PATH))





if __name__ == '__main__':
    runner = HH_opt(l_rate=[0.9,9,0.9])
    runner.loop_func = runner.integ_comp
    runner.Main2('test')
