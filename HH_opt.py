import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from Hodghux import HodgkinHuxley
import tensorflow as tf
import numpy as np
from utils import  plots_output_double, OUT_SETTINGS, OUT_PARAMS, plot_loss_rate, get_data_dump, set_dir, plot_vars

import params
from tqdm import tqdm
import time


NB_SER = 15
BATCH_SIZE = 60



class HH_opt(HodgkinHuxley):
    """Full Hodgkin-Huxley Model implemented in Python"""


    def __init__(self, init_p=params.PARAMS_RAND, fixed=[], constraints=params.CONSTRAINTS, epochs=200, l_rate=[0.9,9,0.9]):
        HodgkinHuxley.__init__(self, init_p, tensors=True)
        self.fixed = fixed
        self.epochs = epochs
        self.start_rate, self.decay_step, self.decay_rate = l_rate
        self.constraints = {}
        if (self.tensors):
            tf.reset_default_graph()
            self.init_p = init_p
            self.param = {}
            for var, val in self.init_p.items():
                if (var in fixed):
                    self.param[var] = tf.constant(val, name=var, dtype=tf.float32)
                else:
                    self.param[var] = tf.get_variable(var, initializer=val, dtype=tf.float32)
                    if var in constraints:
                        self.constraints[var] = constraints[var]


    def condition(self, hprev, x, dt, index, mod):
        return tf.less(index * dt, self.DT)


    def step_long(self, hprev, x):
        index = tf.constant(0.)
        h = tf.while_loop(self.condition, self.loop_func, (hprev, x, self.dt, index, self))
        return h

    def step(self, hprev, x):
        return self.loop_func(hprev, x, self.DT, self)

    def step_test(self, X, i):
        return self.loop_func(X, i, self.dt, self)

    def test(self):

        ts_ = tf.placeholder(shape=[None], dtype=tf.float32)
        init_state = tf.placeholder(shape=[len(self.init_state)], dtype=tf.float32)
        res = tf.scan(self.step_test,
                      self.i_inj,
                      initializer=init_state)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            start = time.time()
            results = sess.run(res, feed_dict={
                ts_: self.t,
                init_state: np.array(self.init_state)
            })
            print('time spent : %.2f s' % (time.time() - start))

            
    def apply_constraints(self):
        c = [tf.assign(self.param[var], tf.clip_by_value(self.param[var], con[0], con[1])) for var, con in self.constraints.items()]
        c = tf.Print(c, [tf.size(c)])
        return c


    def Main(self, subdir, w=[1,0], sufix=''):
        DIR = set_dir(subdir+'/')
        init = self.get_init_state()

        with open('%s%s_%s.txt' % (DIR, OUT_SETTINGS, sufix), 'w') as f:
            f.write('Initial params : %s' % self.init_p + '\n'+
                    'Fixed variables : %s' % [c for c in self.fixed] + '\n'+
                    'Initial state : %s' % init + '\n' +
                    'Model solver : %s' % self.loop_func + '\n' +
                    'Weights (out, cac) : %s' % w + '\n' +
                    'Start rate : %s, decay_step : %s, decay_rate : %s' % (self.start_rate, self.decay_step, self.decay_rate) + '\n')

        self.T, self.X, self.V, self.Ca = get_data_dump()
        if(self.loop_func == self.ik_from_v):
            self.Ca = self.V
        self.DT = params.DT
        # inputs
        xs_ = tf.placeholder(shape=[None], dtype=tf.float32)
        ys_ = tf.placeholder(shape=[2, None], dtype=tf.float32)
        init_state = tf.placeholder(shape=[len(init)], dtype=tf.float32)

        res = tf.scan(self.step,
                      xs_,
                     initializer=init_state)

        cac = res[:, -1]
        out = res[:, 0]
        losses = w[0]*tf.square(tf.subtract(out, ys_[0])) + w[1]*tf.square(tf.subtract(cac, ys_[-1]))
        loss = tf.reduce_mean(losses)

        global_step = tf.Variable(0, trainable=False)
        #progressive learning rate
        learning_rate = tf.train.exponential_decay(
            self.start_rate,  # Base learning rate.
            global_step,  # Current index to the dataset.
            self.decay_step,  # Decay step.
            self.decay_rate,  # Decay rate.
            staircase=True)
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        gvs = opt.compute_gradients(loss)
        #check if nan and clip the values
        grads, vars = zip(*[(tf.cond(tf.is_nan(grad), lambda : 0., lambda : grad), var) for grad, var in gvs])
        grads_normed, _ = tf.clip_by_global_norm(grads, 5.)
        train_op = opt.apply_gradients(zip(grads_normed, vars), global_step=global_step)

        constraints = self.apply_constraints()

        losses = np.zeros(self.epochs)
        rates = np.zeros(self.epochs)

        vars = {}
        for v in self.param.keys():
            vars[v] = np.zeros(self.epochs)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for i in tqdm(range(self.epochs)):
                results, _, train_loss = sess.run([res, train_op, loss], feed_dict={
                    xs_: self.X,
                    ys_: np.vstack((self.V, self.Ca)),
                    init_state: init
                })
                _ = sess.run(constraints)

                with open('%s%s_%s.txt' % (DIR, OUT_PARAMS, sufix), 'w') as f:

                    for name, v in self.param.items():
                        v_ = sess.run(v)
                        vars[name][i] = v_
                        f.write('%s : %s\n' % (name, v_))

                rates[i] = sess.run(learning_rate)
                losses[i] = train_loss
                print('[{}] loss : {}'.format(i, train_loss))
                train_loss = 0

                plots_output_double(self.T, self.X, results[:,0], self.V, results[:,-1], self.Ca, suffix='%s_%s'%(sufix, i + 1), show=False, save=True)
                if(i%10==0):
                    plot_vars(vars, i, suffix=sufix, show=False, save=True)
                    plot_loss_rate(losses, rates, i, suffix=sufix, show=False, save=True)



if __name__ == '__main__':
    runner = HH_opt()
    runner.loop_func = runner.integ_comp
    runner.Main('test')
