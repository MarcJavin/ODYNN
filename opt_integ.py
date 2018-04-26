import scipy as sp
import pylab as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import time

FILE = 'AVAL_test.csv'
df = pd.read_csv(FILE)
Y = np.array(df['trace'])
X = np.array(df['inputCurrent'])*10
T = np.array(df['timeVector'])*1000
DT = T[1] - T[0]

N_HILL = 189e-9 #M

from params import PARAM_CHANNELS, RATE_COLORS


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
    RHO_CA = 0.000239e-11  # mol_per_cm_per_A_per_s
    REST_CA = 0  # M

    dt = 0.01
    t = sp.arange(0.0, 450, dt)

    """ The time to  integrate over """

    def __init__(self):
        # build graph
        tf.reset_default_graph()
        self.index = tf.constant(0, dtype=tf.int32)
        self.rates = {}
        for rate in ['p', 'q', 'n', 'e', 'f']:
            self.rates['%s__mdp'%rate] = tf.get_variable('%s__midpoint' % rate, initializer=PARAM_CHANNELS['%s__midpoint' % rate])
            self.rates['%s__scale'%rate] = tf.get_variable('%s__scale' % rate, initializer=PARAM_CHANNELS['%s__scale' % rate])
            self.rates['%s__tau'%rate] = tf.get_variable('%s__tau' % rate, initializer=PARAM_CHANNELS['%s__tau' % rate])
        self.rates['h__mdp'] = tf.get_variable('h__midpoint', initializer=PARAM_CHANNELS['h__ca_half'])
        self.rates['h__scale'] = tf.get_variable('h__scale', initializer=PARAM_CHANNELS['h__k'])
        self.rates['h__alpha'] = tf.get_variable('h__alpha', initializer=PARAM_CHANNELS['h__alpha'])

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
        mdp = PARAM_CHANNELS['h__ca_half']
        scale = PARAM_CHANNELS['h__k']
        alpha = PARAM_CHANNELS['h__alpha']
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
            return 10. * tf.cast((t > 100), tf.float32) - 10. * tf.cast((t > 200), tf.float32) + 35. * tf.cast((t > 300),
                                                            tf.float32) - 35. * tf.cast((t > 400), tf.float32)

    #@staticmethod
    def dALLdt(self, Xt, t):
        """
        Integrate

        |  :param X:
        |  :param t:
        |  :return: calculate membrane potential & activation variables
        """
        V = Xt[0]
        p = Xt[1]
        q = Xt[2]
        n = Xt[3]
        e = Xt[4]
        f = Xt[5]
        cac = Xt[6]



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
        cac += (-self.I_Ca(V, e, f, h) * self.RHO_CA - ((cac - self.REST_CA) / self.DECAY_CA)) * dt
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
        div= tf.cast(DT, tf.float32)
        dt = DT / div
        index = tf.constant(0.)
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


        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            start = time.time()
            results = sess.run(res, feed_dict={
                ts_: self.t,
                init_state: [-65, 0., 0.95, 0, 0, 1, 0]
            })
            print('time spent : %.2f s' % (time.time() - start))
        self.plots_results(results, self.t, [self.I_inj(t, True) for t in self.t])


    def Main(self):
        """
        Main demo for the Hodgkin Huxley neuron model
        """
        # inputs
        xs_ = tf.placeholder(shape=[None], dtype=tf.float32)
        ys_ = tf.placeholder(shape=[None], dtype=tf.float32)
        init_state = tf.placeholder(shape=[7], dtype=tf.float32)

        #res = tf.contrib.integrate.odeint(self.dALLdt, [-65, 0., 0.95, 0, 0, 1, 0], T)
        res = tf.scan(self.step,
                      xs_,
                     initializer=init_state)

        cacs = res[:, -1]
        cacs_pow = tf.pow(cacs, 3.8)
        cac_lum = cacs / (cacs_pow + N_HILL)
        losses = tf.square(tf.subtract(cac_lum, ys_))
        loss = tf.reduce_mean(losses)
        loss = tf.Print(loss, [loss], 'loss : ')
        train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

        epochs = 2
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            train_loss = 0

            for i in range(epochs):
                for j in range(1):
                    print(sess.run(self.rates['p__tau']))
                    resu, cacl, _, train_loss, p = sess.run([res, cac_lum, train_op, loss, self.rates['p__tau']], feed_dict={
                        xs_: X,
                        ys_: Y,
                        init_state: [-65, 0., 0.95, 0, 0, 1, 0]
                    })
                self.plots_output(T, X, cacl, Y)
                print('[{}] loss : {}'.format(i, train_loss))
                train_loss = 0

            self.plots_results(T, X, resu)






    def plots_output(self, ts, i_inj, cac_lum, y_cac_lum):

        plt.figure()


        plt.subplot(3, 1, 1)
        plt.plot(ts, cac_lum, 'r')
        plt.ylabel('Ca2+ concentration predicted')

        plt.subplot(3, 1, 2)
        plt.plot(ts, y_cac_lum, 'r')
        plt.ylabel('Ca2+ concentration true')

        plt.subplot(3, 1, 3)
        plt.plot(ts, i_inj, 'k')
        plt.xlabel('t (ms)')
        plt.ylabel('$I_{inj}$ ($\\mu{A}/cm^2$)')
        plt.ylim(-1, 40)

        plt.show()


    def plots_results(self, ts, i_inj_values, results):
        V = results[:, 0]
        p = results[:, 1]
        q = results[:, 2]
        n = results[:, 3]
        e = results[:, 4]
        f = results[:, 5]
        cac = results[:, 6]

        h = self.h_notensor(cac)
        ica = self.I_Ca(V, e, f, h)
        ik = self.I_Ks(V, n) + self.I_Kf(V, p, q)
        il = self.I_L(V)

        plt.figure()

        plt.subplot(5, 1, 1)
        plt.title('Hodgkin-Huxley Neuron')
        plt.plot(ts, V, 'k')
        plt.ylabel('V (mV)')

        plt.subplot(5, 1, 2)
        plt.plot(ts, cac, 'r')
        plt.ylabel('Ca2+ concentration')

        plt.subplot(5, 1, 3)
        plt.plot(ts, ica, 'c', label='$I_{Ca}$')
        plt.plot(ts, ik, 'y', label='$I_{K}$')
        plt.plot(ts, il, 'm', label='$I_{L}$')
        plt.ylabel('Current')
        plt.legend()

        plt.subplot(5, 1, 4)
        plt.plot(ts, p, RATE_COLORS['p'], label='p')
        plt.plot(ts, q, RATE_COLORS['q'], label='q')
        plt.plot(ts, n, RATE_COLORS['n'], label='n')
        plt.plot(ts, e, RATE_COLORS['e'], label='e')
        plt.plot(ts, f, RATE_COLORS['f'], label='f')
        plt.plot(ts, h, RATE_COLORS['h'], label='h')
        plt.ylabel('Gating Value')
        plt.legend()

        plt.subplot(5, 1, 5)
        plt.plot(ts, i_inj_values, 'k')
        plt.xlabel('t (ms)')
        plt.ylabel('$I_{inj}$ ($\\mu{A}/cm^2$)')
        plt.ylim(-1, 40)

        plt.show()

if __name__ == '__main__':
    runner = HodgkinHuxley()
    runner.Main()