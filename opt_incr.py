import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
import numpy as np
from utils import plots_output, plots_output_double, plots_results, get_data, plot_loss, get_data_dump
import params
from tqdm import tqdm
import time

OUT = '_params.txt'
NB_SER = 15
BATCH_SIZE = 60
T, X, V, Ca = get_data_dump()
DT = T[1] - T[0]

# INIT_STATE = [-50, 0, 1, 0]
INIT_STATE = [-50., 0., 0.95, 0., 0., 1., 1.e-7]


class HodgkinHuxley():
    """Full Hodgkin-Huxley Model implemented in Python"""

    DECAY_CA = params.DECAY_CA
    RHO_CA = params.RHO_CA
    REST_CA = params.REST_CA

    dt = 0.1
    t = params.t
    i_inj = params.i_inj



    """ The time to  integrate over """

    def __init__(self):
        # build graph
        tf.reset_default_graph()
        self.C_m = tf.get_variable('Cm', initializer=params.params['C_m'])

        self.param = {}
        for var, val in params.params.items():
            self.param[var] = tf.get_variable(var, initializer=val, dtype=tf.float32)




    def inf(self, V, rate):
        mdp = self.param['%s__mdp' % rate]
        scale = self.param['%s__scale' % rate]
        return tf.sigmoid((V - mdp) / scale)

    def h(self, cac):
        """Channel gating kinetics. Functions of membrane voltage"""
        mdp = self.param['h__mdp']
        scale = self.param['h__scale']
        alpha = self.param['h__alpha']
        q = tf.sigmoid((cac - mdp) / scale)
        return 1 + (q - 1) * alpha


    def I_Ca(self, V, e, f, h):
        """
        Membrane current (in uA/cm^2)
        Sodium (Na = element name)
        """
        return self.param['g_Ca'] * e ** 2 * f * h * (V - self.param['E_Ca'])

    def I_Kf(self, V, p, q):
        """
        Membrane current (in uA/cm^2)
        Potassium (K = element name)
        """
        return self.param['g_Kf'] * p ** 4 * q * (V - self.param['E_K'])

    def I_Ks(self, V, n):
        """
        Membrane current (in uA/cm^2)
        Potassium (K = element name)
        """
        return self.param['g_Ks'] * n * (V - self.param['E_K'])

    #  Leak
    def I_L(self, V):
        """
        Membrane current (in uA/cm^2)
        Leak
        """
        return self.param['g_L'] * (V - self.param['E_L'])


    def integ_simp(self, X, i_sin, dt, index=0):
        """
        Integrate
        """
        index += 1

        V = X[0]
        e = X[-3]
        f = X[-2]
        cac = X[-1]

        h = self.h(cac)
        V += ((i_sin - self.I_Ca(V, e, f, h) - self.I_L(V)) / self.C_m) * dt
        cac = (self.DECAY_CA / (dt + self.DECAY_CA)) * (cac - self.I_Ca(V, e, f, h) * self.RHO_CA * dt + self.REST_CA * self.DECAY_CA / dt)
        tau = self.param['e__tau']
        e = ((tau * dt) / (tau + dt)) * ((e / dt) + (self.inf(V, 'e') / tau))
        tau = self.param['f__tau']
        f = ((tau * dt) / (tau + dt)) * ((f / dt) + (self.inf(V, 'f') / tau))
        return tf.stack([V, e, f, cac], 0)

    def notau_simp(self, X, i_sin, dt, index=0):
        """
        Integrate
        """
        index += 1

        V = X[0]
        e = X[-3]
        f = X[-2]
        cac = X[-1]

        h = self.h(cac)
        V += ((i_sin - self.I_Ca(V, e, f, h) - self.I_L(V)) / self.C_m) * dt
        cac = (self.DECAY_CA / (dt + self.DECAY_CA)) * (cac - self.I_Ca(V, e, f, h) * self.RHO_CA * dt + self.REST_CA * self.DECAY_CA / dt)
        e = self.inf(V, 'e')
        f = self.inf(V, 'f')
        return tf.stack([V, e, f, cac], 0)


    def integ_complete(self, X, i_sin, dt):
        """
        Integrate
        """
        V = X[0]
        p = X[1]
        q = X[2]
        n = X[3]
        e = X[4]
        f = X[5]
        cac = X[6]


        h = self.h(cac)
        V += ((i_sin - self.I_Ca(V, e, f, h) - self.I_Ks(V, n) - self.I_Kf(V, p, q) - self.I_L(V)) / self.C_m) * dt
        cac = (self.DECAY_CA/(dt+self.DECAY_CA)) * (cac - self.I_Ca(V, e, f, h)*self.RHO_CA*dt + self.REST_CA*self.DECAY_CA/dt)
        tau = self.param['p__tau']
        p = ((tau*dt) / (tau+dt)) * ((p/dt) + (self.inf(V, 'p')/tau))
        tau = self.param['q__tau']
        q = ((tau*dt) / (tau+dt)) * ((q/dt) + (self.inf(V, 'q')/tau))
        tau = self.param['e__tau']
        e = ((tau*dt) / (tau+dt)) * ((e/dt) + (self.inf(V, 'e')/tau))
        tau = self.param['f__tau']
        f = ((tau*dt) / (tau+dt)) * ((f/dt) + (self.inf(V, 'f')/tau))
        tau = self.param['n__tau']
        n = ((tau*dt) / (tau+dt)) * ((n/dt) + (self.inf(V, 'n')/tau))
        return tf.stack([V, p, q, n, e, f, cac], 0)

    def no_tau(self, X, i_sin, dt, index=0):
        """
        Integrate
        """
        index += 1
        V = X[0]
        p = X[1]
        q = X[2]
        n = X[3]
        e = X[4]
        f = X[5]
        cac = X[6]
        h = self.h(cac)
        V += ((i_sin - self.I_Ca(V, e, f, h) - self.I_Ks(V, n) - self.I_Kf(V, p, q) - self.I_L(
            V)) / self.C_m) * dt
        # cac += (-self.I_Ca(V, e, f, h) * self.RHO_CA - ((cac - self.REST_CA) / self.DECAY_CA)) * dt
        cac = (self.DECAY_CA / (dt + self.DECAY_CA)) * (
                    cac - self.I_Ca(V, e, f, h) * self.RHO_CA * dt + self.REST_CA * self.DECAY_CA / dt)
        p = self.inf(V, 'p')
        q = self.inf(V, 'q')
        e = self.inf(V, 'e')
        f = self.inf(V, 'f')
        n = self.inf(V, 'n')
        return tf.stack([V, p, q, n, e, f, cac], 0)

    loop_func = integ_complete


    def condition(self, hprev, x, dt, index):
        return tf.less(index * dt, DT)


    def step_long(self, hprev, x):
        div = DT/self.dt
        div= tf.cast(div, tf.float32)
        dt = DT / div
        index = tf.constant(0.)
        h = tf.while_loop(self.condition, self.loop_func, (hprev, x, dt, index))
        return h

    def step(self, hprev, x):
        return self.integ_complete(hprev, x, DT)

    def step_test(self, X, i):
        return self.loop_func(X, i, self.dt)

    def test(self):

        ts_ = tf.placeholder(shape=[None], dtype=tf.float32)
        init_state = tf.placeholder(shape=[7], dtype=tf.float32)
        res = tf.scan(self.step_test,
                      self.i_inj,
                      initializer=init_state)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            start = time.time()
            print(INIT_STATE)
            results = sess.run(res, feed_dict={
                ts_: self.t,
                init_state: np.array(INIT_STATE)
            })
            print('time spent : %.2f s' % (time.time() - start))
        plots_results(self, self.t, self.i_inj, results, cur=False)


    def Main(self, prefix):
        # inputs
        xs_ = tf.placeholder(shape=[None], dtype=tf.float32)
        ys_ = tf.placeholder(shape=[2, None], dtype=tf.float32)
        init_state = tf.placeholder(shape=[7], dtype=tf.float32)

        res = tf.scan(self.step,
                      xs_,
                     initializer=init_state)

        cac = res[:, -1]
        volt = res[:, 0]
        losses = tf.square(tf.subtract(volt, ys_[0])) + tf.square(tf.subtract(cac, ys_[-1]))
        loss = tf.reduce_mean(losses)
        loss = tf.Print(loss, [loss], 'loss : ')
        opt = tf.train.AdamOptimizer(learning_rate=0.01)
        grads = opt.compute_gradients(loss)
        # capped_grads = capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads]
        train_op = opt.apply_gradients(grads)

        epochs = 200
        losses = np.zeros(epochs)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            train_loss = 0

            for i in tqdm(range(epochs)):
                results, _, train_loss = sess.run([res, train_op, loss], feed_dict={
                    xs_: X,
                    ys_: np.vstack((V, Ca)),
                    init_state: INIT_STATE
                })

                with open(prefix + OUT, 'w') as f:
                    for v in tf.trainable_variables():
                        v_ = sess.run(v)
                        f.write('%s : %s\n' % (v, v_))

                # self.plots_output(T, X, cacl, Y)
                losses[i] = train_loss
                print('[{}] loss : {}'.format(i, train_loss))
                train_loss = 0

                plots_output_double(T, X, results[:,0], V, results[:,-1], Ca, suffix='%s_%s'%(prefix,i + 1), show=False, save=True)
                plot_loss(losses, suffix=prefix, show=False, save=True)
                # plots_results_ca(self, T, X, results, suffix=0, show=False, save=True)



if __name__ == '__main__':
    runner = HodgkinHuxley()
    # runner.loop_func = runner.no_tau
    runner.Main('own_data')
