import scipy as sp

import tensorflow as tf
import time
from tqdm import tqdm
from utils import plots_output, plots_results, get_data

NB_SER = 15
BATCH_SIZE = 50
T, X, Y = get_data()
DT = T[1] - T[0]

INIT_STATE = [-65, 0., 0.95, 0, 0, 1, 1e-7]

N_HILL = 0.189 #mM

from params import PARAM_GATES, PARAM_MEMB


class HodgkinHuxley():
    """Full Hodgkin-Huxley Model implemented in Python"""

    C_m = 1.0
    """membrane capacitance, in uF/cm^2"""

    g_Ca = 3.0
    """Calcium (Na) maximum conductances, in mS/cm^2"""

    g_Ks = 3.0
    g_Kf = 0.0712
    """Postassium (K) maximum conductances, in mS/cm^2"""

    g_L = 0.005
    """Leak maximum conductances, in mS/cm^2"""

    E_Ca = 40.0
    """Sodium (Na) Nernst reversal potentials, in mV"""

    E_K = -60.0
    """Postassium (K) Nernst reversal potentials, in mV"""

    E_L = -50.0
    """Leak Nernst reversal potentials, in mV"""

    DECAY_CA = 11.6  # ms
    RHO_CA = 0.000239e3  # mol_per_cm_per_uA_per_ms
    REST_CA = 0  # M

    dt = 0.1
    t = sp.arange(0.0, 450., dt)

    """ The time to  integrate over """

    def __init__(self):
        # build graph
        tf.reset_default_graph()
        self.rates = {}
        for var, val in PARAM_MEMB.items():
            self.rates[var] = tf.get_variable(var, initializer=val)
        for var, val in PARAM_GATES.items():
            self.rates[var] = tf.get_variable(var, initializer=val)

    def inf(self, V, rate):
        mdp = self.rates['%s__mdp'%rate]
        scale = self.rates['%s__scale'%rate]
        return tf.sigmoid((V - mdp)/scale)

    def h(self, cac):
        """Channel gating kinetics. Functions of membrane voltage"""
        mdp = self.rates['h__mdp']
        scale = self.rates['h__scale']
        alpha = self.rates['h__alpha']
        q = tf.sigmoid((cac - mdp)/scale)
        return 1 + (q - 1) * alpha

    def h_notensor(self, cac):
        """Channel gating kinetics. Functions of membrane voltage"""
        mdp = PARAM_GATES['h__mdp']
        scale = PARAM_GATES['h__scale']
        alpha = PARAM_GATES['h__alpha']
        q = 1 / (1 + sp.exp((mdp - cac) / scale))
        return 1 + (q - 1) * alpha

    def I_Ca(self, V, e, f, h):
        """
        Membrane current (in uA/cm^2)
        Sodium (Na = element name)

        |  :param V:
        |  :param m:
        |  :param h:
        |  :return:
        """
        return self.g_Ca * e**2 * f * h * (V - self.E_Ca)

    def I_Kf(self, V, p, q):
        """
        Membrane current (in uA/cm^2)
        Potassium (K = element name)

        |  :param V:
        |  :param h:
        |  :return:
        """
        return self.g_Kf * p ** 4 * q * (V - self.E_K)

    def I_Ks(self, V, n):
        """
        Membrane current (in uA/cm^2)
        Potassium (K = element name)

        |  :param V:
        |  :param h:
        |  :return:
        """
        return self.g_Ks * n * (V - self.E_K)

    #  Leak
    def I_L(self, V):
        """
        Membrane current (in uA/cm^2)
        Leak

        |  :param V:
        |  :param h:
        |  :return:
        """
        return self.g_L * (V - self.E_L)

    def I_inj(self, t, no_tensor=False):
        """
        External Current

        |  :param t: time
        |  :return: step up to 10 uA/cm^2 at t>100
        |           step down to 0 uA/cm^2 at t>200
        |           step up to 35 uA/cm^2 at t>300
        |           step down to 0 uA/cm^2 at t>400
        """
        if no_tensor:
            return 10 * (t > 100) - 10 * (t > 200) + 35 * (t > 300) - 35 * (t > 400)
        else:
            # idx = tf.minimum(tf.cast((t / DT), tf.int32), X.shape[0] - 1)
            # return tf.cast(tf.gather(X, idx), tf.float32)
            return 10. * tf.cast((t > 100), tf.float32) - 10. * tf.cast((t > 200), tf.float32) + 35. * tf.cast((t > 300),
                                                            tf.float32) - 35. * tf.cast((t > 400), tf.float32)

    #@staticmethod
    def dALLdt(self, X, t):
        """
        Integrate

        |  :param X:
        |  :param t:
        |  :return: calculate membrane potential & activation variables
        """
        V = X[0]
        p = X[1]
        q = X[2]
        n = X[3]
        e = X[4]
        f = X[5]
        cac = X[6]

        h = self.h(cac)
        dVdt = (self.I_inj(t) - self.I_Ca(V, e, f, h) - self.I_Ks(V, n) - self.I_Kf(V, p, q) - self.I_L(V)) / self.C_m
        dpdt = (self.inf(V, 'p') - p) / self.rates['p__tau']
        dqdt = (self.inf(V, 'q') - q) / self.rates['q__tau']
        dedt = (self.inf(V, 'e') - e) / self.rates['e__tau']
        dfdt = (self.inf(V, 'f') - f) / self.rates['f__tau']
        dndt = (self.inf(V, 'n') - n) / self.rates['n__tau']
        dcacdt = - self.I_Ca(V, e, f, h) * self.RHO_CA - ((cac - self.REST_CA) / self.DECAY_CA)
        return tf.stack([dVdt, dpdt, dqdt, dndt, dedt, dfdt, dcacdt], 0)


    def integ_comp(self, X, i_sin, dt, index=0):
        """
        Integrate

        |  :param X:
        |  :param t:
        |  :return: calculate membrane potential & activation variables
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
        V += ((i_sin - self.I_Ca(V, e, f, h) - self.I_Ks(V, n) - self.I_Kf(V, p, q) - self.I_L(V)) / self.C_m) * dt
        #cac += (-self.I_Ca(V, e, f, h) * self.RHO_CA - ((cac - self.REST_CA) / self.DECAY_CA)) * dt
        cac = (self.DECAY_CA/(dt+self.DECAY_CA)) * (cac - self.I_Ca(V, e, f, h)*self.RHO_CA*dt + self.REST_CA*self.DECAY_CA/dt)
        tau = self.rates['p__tau']
        p = ((tau*dt) / (tau+dt)) * ((p/dt) + (self.inf(V, 'p')/tau))
        tau = self.rates['q__tau']
        q = ((tau*dt) / (tau+dt)) * ((q/dt) + (self.inf(V, 'q')/tau))
        tau = self.rates['e__tau']
        e = ((tau*dt) / (tau+dt)) * ((e/dt) + (self.inf(V, 'e')/tau))
        tau = self.rates['f__tau']
        f = ((tau*dt) / (tau+dt)) * ((f/dt) + (self.inf(V, 'f')/tau))
        tau = self.rates['n__tau']
        n = ((tau*dt) / (tau+dt)) * ((n/dt) + (self.inf(V, 'n')/tau))
        return tf.stack([V, p, q, n, e, f, cac], 0), i_sin, dt, index


    def condition(self, hprev, x, dt, index):
        return tf.less(index * dt, DT)


    def step(self, hprev, x):
        div = DT/self.dt
        div= tf.cast(div, tf.float32)
        dt = DT / div
        index = tf.constant(0.)
        # t = sp.arange(0, DT, self.dt)
        # h = tf.contrib.integrate.odeint(self.dALLdt, hprev, t)[-1] #put x in the object
        h = tf.while_loop(self.condition, self.integ_comp, (hprev, x, dt, index))[0]
        return h

    def step_test(self, X, t):
        return self.integ_comp(X, self.I_inj(t), self.dt)[0]


    def test(self):

        ts_ = tf.placeholder(shape=[None], dtype=tf.float32)
        init_state = tf.placeholder(shape=[7], dtype=tf.float32)

        res = tf.scan(self.step_test,
                      ts_,
                      initializer=init_state)
        # res = tf.contrib.integrate.odeint(self.dALLdt, init_state, self.t)


        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            start = time.time()
            results = sess.run(res, feed_dict={
                ts_: self.t,
                init_state: INIT_STATE
            })
            print('time spent : %.2f s' % (time.time() - start))
        plots_results(self, self.t, [self.I_inj(t, True) for t in self.t], results)


    def Main(self, prefix = ""):
        """
        Main demo for the Hodgkin Huxley neuron model
        """
        # inputs
        xs_ = tf.placeholder(shape=[None], dtype=tf.float32)
        ys_ = tf.placeholder(shape=[None], dtype=tf.float32)
        init_state = tf.placeholder(shape=[7], dtype=tf.float32)

        # res = tf.contrib.integrate.odeint(self.dALLdt, init_state, T)
        res = tf.scan(self.step,
                     xs_,
                    initializer=init_state)

        cacs = res[:, -1]
        cacs_pow = tf.pow(cacs, 3.8)
        cac_lum = cacs #cacs_pow / (cacs_pow + N_HILL)
        cac_lum = cac_lum * 0.01
        losses = tf.square(tf.subtract(cac_lum, ys_))
        loss = tf.reduce_mean(losses)
        loss = tf.Print(loss, [loss], 'loss : ')
        opt = tf.train.AdamOptimizer(0.1)
        grads = opt.compute_gradients(loss)
        # capped_grads = capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads]
        train_op = opt.apply_gradients(grads)

        epochs = 200
        start = time.time()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            train_loss = 0

            results, cacl = sess.run([res, cac_lum], feed_dict={
                xs_: X,
                ys_: Y,
                init_state: INIT_STATE
            })
            plots_results(self, T, X, results, suffix='%s_integ_0'%prefix, show=False, save=True)
            plots_output(T, X, cacl, Y, suffix='%s_integ_0'%prefix, show=False, save=True)
            print(time.time() - start)

            for i in tqdm(range(epochs)):
                final_state = INIT_STATE
                results, cacl, train_loss, grad, _ = sess.run([res, cac_lum, loss, grads, train_op], feed_dict={
                    xs_: X,
                    ys_: Y,
                    init_state: final_state
                })
                for v, g in grad:
                    print(v, g)
                final_state = results[-1, :]
                train_loss += train_loss
                # for j in range(0, X.shape[0], BATCH_SIZE):
                #
                #     results, train_loss, grad, _ = sess.run([res, loss, grads, train_op], feed_dict={
                #         xs_: X[j:j+BATCH_SIZE],
                #         ys_: Y[j:j+BATCH_SIZE],
                #         init_state: final_state
                #     })
                #     for v, g in grad:
                #         print(v, g)
                #     final_state = results[-1,:]
                #     train_loss += train_loss

                # self.plots_output(T, X, cacl, Y)
                print('[{}] loss : {}'.format(i, train_loss))
                train_loss = 0

                # results, cacl = sess.run([res, cac_lum], feed_dict={
                #     xs_: X,
                #     ys_: Y,
                #     init_state: INIT_STATE
                # })
                plots_results(self, T, X, results, suffix="%s_integ_%s"%(prefix, i+1), show=False, save=True)
                plots_output(T, X, cacl, Y, suffix="%s_integ_%s"%(prefix, i+1), show=False, save=True)




if __name__ == '__main__':
    runner = HodgkinHuxley()
    runner.test()