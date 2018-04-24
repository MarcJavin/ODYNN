import scipy as sp
import pylab as plt
import numpy as np
from scipy.integrate import odeint
import tensorflow as tf


N_HILL = 189e-9 #M

PARAM_CHANNELS = {
    'p__tau': 2.25518,
    'p__scale': 7.42636,
    'p__midpoint': -8.05232,

    'q__tau': 149.963,
    'q__scale': -9.97468,
    'q__midpoint': -15.6456,

    'n__tau': 25.0007,
    'n__scale': 15.8512,
    'n__midpoint': 19.8741,

    'e__tau': 0.100027,
    'e__scale': 6.74821,
    'e__midpoint': -3.3568,

    'f__tau': 150.88,
    'f__scale': -5.03176,
    'f__midpoint': 25.1815,

    'e__tau': 0.100027,
    'e__scale': 6.74821,
    'e__midpoint': -3.3568,

    'h__alpha': 0.282473,
    'h__k': -1.00056e-11,
    'h__ca_half': 6.41889e-11
}


class HodgkinHuxley():
    """Full Hodgkin-Huxley Model implemented in Python"""

    C_m = 1.0
    """membrane capacitance, in uF/cm^2"""

    g_Ca = .0
    """Calcium (Na) maximum conductances, in mS/cm^2"""

    g_Ks = 3.0
    g_Kf = 0.07
    """Postassium (K) maximum conductances, in mS/cm^2"""

    g_L = 0.005
    """Leak maximum conductances, in mS/cm^2"""

    E_Ca = 40.0
    """Sodium (Na) Nernst reversal potentials, in mV"""

    E_K = -60.0
    """Postassium (K) Nernst reversal potentials, in mV"""

    E_L = -50.0
    """Leak Nernst reversal potentials, in mV"""

    DECAY_CA = 11.6  # s
    RHO_CA = 0.000239e-2  # mol_per_cm_per_A_per_s
    REST_CA = 0  # M

    t = sp.arange(0.0, 450.0, 0.01)
    """ The time to integrate over """

    def __init__(self):
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
        q = 1 + (1 + sp.exp((mdp - cac) / scale))
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
        return self.g_Ca * e ** 2 * f * h * (V - self.E_Ca)

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

    def I_inj(self, t):
        """
        External Current

        |  :param t: time
        |  :return: step up to 10 uA/cm^2 at t>100
        |           step down to 0 uA/cm^2 at t>200
        |           step up to 35 uA/cm^2 at t>300
        |           step down to 0 uA/cm^2 at t>400
        """
        return 10. * tf.cast((t > 100), tf.float32) - 10. * tf.cast((t > 200), tf.float32) + 35. * tf.cast((t > 300), tf.float32) - 35. * tf.cast((t > 400), tf.float32)


    def I_inj_notensor(self, t):
        """
        External Current

        |  :param t: time
        |  :return: step up to 10 uA/cm^2 at t>100
        |           step down to 0 uA/cm^2 at t>200
        |           step up to 35 uA/cm^2 at t>300
        |           step down to 0 uA/cm^2 at t>400
        """
        return 10*(t>100) - 10*(t>200) + 35*(t>300) - 35*(t>400)

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
        dcacdt = self.I_Ca(V, e, f, h) * self.RHO_CA - ((cac - self.REST_CA) / self.DECAY_CA)
        return tf.stack([dVdt, dpdt, dqdt, dndt, dedt, dfdt, dcacdt], 0)

    def Main(self):
        """
        Main demo for the Hodgkin Huxley neuron model
        """

        res = tf.contrib.integrate.odeint(self.dALLdt, [-65, 0.5, 0.5, 0.5, 0.5, 0.5, 1e-10], self.t)

        cac = res[:,-1]
        cac_pow = tf.pow(cac, 3.8)
        cac_lum = cac / (cac_pow + N_HILL)
        """losses = tf.square(tf.subtract(cac_lum, ys_))
        loss = tf.reduce_mean(losses)
        loss = tf.Print(loss, [loss], 'loss : ')
        train_op = tf.train.AdamOptimizer(learning_rate=0.02).minimize(loss)"""

        epochs = 1
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            train_loss = 0
            try:
                for i in range(epochs):
                    for j in range(1):
                        X = sess.run(res)
                        V = X[:,0]
                        p = X[:, 1]
                        q = X[:, 2]
                        n = X[:, 3]
                        e = X[:, 4]
                        f = X[:, 5]
                        cac = X[:, 6]

                        """plt.subplot(411)
                        plt.title('Potential')
                        plt.plot(t, res[:, 0], 'b')
                        plt.subplot(412)
                        plt.title('calcium concentration')
                        plt.plot(t, cac_l, 'r')
                        plt.subplot(413)
                        plt.title('input current')
                        plt.plot(t, xs, 'g')
                        plt.subplot(414)
                        plt.title('p')
                        plt.plot(t, ys, 'b')
                        #plt.plot(t, res[:,-4], 'black', t, res[:,-3], 'orange', t, res[:,-2], 'purple')
                        plt.show()"""
            except:
                pass


        h = self.h_notensor(cac)
        ica = self.I_Ca(V, e, f, h)
        ik = self.I_Ks(V, n) + self.I_Kf(V, p, q)
        il = self.I_L(V)


        plt.figure()

        plt.subplot(5, 1, 1)
        plt.title('Hodgkin-Huxley Neuron')
        plt.plot(self.t, V, 'k')
        plt.ylabel('V (mV)')

        print('yolo')

        plt.subplot(5, 1, 2)
        plt.plot(self.t, ica, 'c', label='$I_{Na}$')
        plt.plot(self.t, ik, 'y', label='$I_{K}$')
        plt.plot(self.t, il, 'm', label='$I_{L}$')
        plt.ylabel('Current')
        plt.legend()
        print('yolo')

        plt.subplot(5, 1, 3)
        plt.plot(self.t, p, 'r', label='p')
        plt.plot(self.t, q, 'g', label='q')
        plt.plot(self.t, n, 'b', label='n')
        plt.ylabel('Gating Value')
        plt.legend()
        print('yolo')

        plt.subplot(5, 1, 4)
        plt.plot(self.t, e, 'r', label='e')
        plt.plot(self.t, f, 'g', label='f')
        plt.plot(self.t, h, 'b', label='h')
        plt.ylabel('Gating Value')
        plt.legend()
        print('yolo')

        plt.subplot(5, 1, 5)
        i_inj_values = [self.I_inj_notensor(t) for t in self.t]
        plt.plot(self.t, i_inj_values, 'k')
        plt.xlabel('t (ms)')
        plt.ylabel('$I_{inj}$ ($\\mu{A}/cm^2$)')
        plt.ylim(-1, 40)

        plt.show()


if __name__ == '__main__':
    runner = HodgkinHuxley()
    runner.Main()