"""
.. module:: self
    :synopsis: Module defining a self for C. elegans neurons

.. moduleauthor:: Marc Javin
"""


import numpy as np
import random
import tensorflow as tf
import scipy as sp

from .model import Model, V_pos, Ca_pos
from . import utils
from pylab import plt



REST_CA = 0.
INITS = {
    'i': 0,
    'V': -50.,
    'p': 0.,
    'q': 0.95,
    'n': 0.,
    'e': 0.,
    'f': 1.,
    'h': 0.,
    'cac': 1.e-7
}
INIT_STATE = np.array([INITS[p] for p in ['V', 'p', 'q', 'n', 'e', 'f', 'cac']])
INIT_STATE_ica = [INITS[p] for p in ['i', 'e', 'f', 'h', 'cac']]
INIT_STATE_ik = [INITS[p] for p in ['i', 'p', 'q', 'n']]



CONSTRAINTS = {
    'decay_ca': [1e-3, np.infty],
    'rho_ca': [1e-3, np.infty],
    'C_m': [5e-1, np.infty],
    'e__scale': [1e-3, np.infty],
    'e__tau': [1e-3, np.infty],
    'f__scale': [-np.infty, 1e-3],
    'f__tau': [1e-3, np.infty],
    'g_Ca': [1e-5, np.infty],
    'g_Kf': [1e-5, np.infty],
    'g_Ks': [1e-5, np.infty],
    'g_L': [1e-5, np.infty],
    'h__alpha': [0, 1],
    'h__scale': [-np.infty, 1e-3],
    'n__scale': [1e-3, np.infty],
    'n__tau': [1e-3, np.infty],
    'p__scale': [1e-3, np.infty],
    'p__tau': [1e-3, np.infty],
    'q__scale': [-np.infty, 1e-3],
    'q__tau': [1e-3, np.infty]
}

DEFAULT_2 = {
    'decay_ca': 110.,
    'rho_ca': 0.23,
    'p__tau': 100.,  # ms
    'p__scale': 7.43,  # mV
    'p__mdp': -8.05,  # mV
    'q__tau': 100.,
    'q__scale': -9.97,
    'q__mdp': -15.6,
    'n__tau': 1050.,
    'n__scale': 20.,
    'n__mdp': 2.,
    'f__tau': 301.,
    'f__scale': -20.03,
    'f__mdp': 5.2,
    'e__tau': 20.,
    'e__scale': 15.,
    'e__mdp': -6.,
    'h__alpha': 0.282,  # None
    'h__scale': -1.,  # mol per m3
    'h__mdp': 302.,
    'C_m': 20.0,
    'g_Ca': 3.,
    'g_Ks': 6.0,
    'g_Kf': 0.07,
    'g_L': 0.005,
    'E_Ca': 20.0,
    'E_K': -60.0,
    'E_L': -60.0
}
DEFAULT = {
    'decay_ca': 110.,
    'rho_ca': 0.23,

    'p__tau': 100.,  # ms
    'p__scale': 7.43,  # mV
    'p__mdp': -8.05,  # mV

    'q__tau': 150.,
    'q__scale': -9.97,
    'q__mdp': -15.6,

    'n__tau': 25.,
    'n__scale': 15.9,
    'n__mdp': 19.9,

    'f__tau': 151.,
    'f__scale': -5.03,
    'f__mdp': 25.2,

    'e__tau': 10.,
    'e__scale': 6.75,
    'e__mdp': -3.36,

    'h__alpha': 0.282,  # None
    'h__scale': -1.,  # mol per m3
    'h__mdp': 6.42,

    'C_m': 20.0,
    # membrane capacitance, in uF/cm$^2$

    'g_Ca': 3.0,
    # Calcium (Na) maximum conductances, in mS/cm$^2$

    'g_Ks': 10.0,
    'g_Kf': 0.07,
    # Postassium (K) maximum conductances, in mS/cm$^2$

    'g_L': 0.005,
    # Leak maximum conductances, in mS/cm$^2$

    'E_Ca': 20.0,
    # Sodium (Na) Nernst reversal potentials, in mV

    'E_K': -60.0,
    # Postassium (K) Nernst reversal potentials, in mV

    'E_L': -60.0
    # Leak Nernst reversal potentials, in mV
}

ALL = set(DEFAULT.keys())

MAX_TAU = 200.
MIN_SCALE = 1.
MAX_SCALE = 50.
MIN_MDP = -40.
MAX_MDP = 30.
MAX_G = 10.

def give_rand():
    return {
        'decay_ca': 110.,
        'rho_ca': 0.23e-2,
        'p__tau': random.uniform(0.1, MAX_TAU),
        'p__scale': random.uniform(MIN_SCALE, MAX_SCALE),
        'p__mdp': random.uniform(MIN_MDP, MAX_MDP),

        'q__tau': random.uniform(0.1, MAX_TAU),
        'q__scale': random.uniform(-MAX_SCALE, -MIN_SCALE),
        'q__mdp': random.uniform(MIN_MDP, MAX_MDP),

        'n__tau': random.uniform(MAX_TAU, 5*MAX_TAU),
        'n__scale': random.uniform(MIN_SCALE, MAX_SCALE),
        'n__mdp': random.uniform(MIN_MDP, MAX_MDP),

        'f__tau': random.uniform(0.1, MAX_TAU),
        'f__scale': random.uniform(-MAX_SCALE, -MIN_SCALE),
        'f__mdp': random.uniform(MIN_MDP, MAX_MDP),

        'e__tau': random.uniform(0.1, MAX_TAU),
        'e__scale': random.uniform(MIN_SCALE, MAX_SCALE),
        'e__mdp': random.uniform(-30., 0.),

        'h__alpha': random.uniform(0.1, 0.9),
        'h__scale': random.uniform(-MAX_SCALE, -MIN_SCALE),
        'h__mdp': random.uniform(1, 100),

        'C_m': random.uniform(0.1, 40.),
        'g_Ca': random.uniform(0.1, MAX_G),
        'g_Ks': random.uniform(0.1, MAX_G),
        'g_Kf': random.uniform(0.1, MAX_G),
        'g_L': random.uniform(0.001, 0.5),
        'E_Ca': random.uniform(0., 40),
        'E_K': random.uniform(-80, -40.),
        'E_L': random.uniform(-80, -40.),
    }



class HodgkinHuxley(Model):
    """Full Hodgkin-Huxley Model implemented for C. elegans"""

    REST_CA = REST_CA
    _init_state = INIT_STATE
    """initial state for neurons : voltage, rates and $[Ca^{2+}]$"""
    default_params = DEFAULT
    """default parameters as a dictionnary"""
    _constraints_dic = CONSTRAINTS
    """constraints to be applied when optimizing"""

    def __init__(self, init_p=None, tensors=False, dt=0.1):
        Model.__init__(self, init_p=init_p, tensors=tensors, dt=dt)

    def inf(self, V, rate):
        """
        Compute the steady state value of a gate activation rate
        Parameters
        ----------
        V : float
            voltage
        rate : string
            name of the rate
        Returns
        ----------
        float, value of the rate steady state
        """
        mdp = self._param['%s__mdp' % rate]
        scale = self._param['%s__scale' % rate]
        if self._tensors:
            # print('V : ', V)
            # print('mdp : ', mdp)
            return tf.sigmoid((V - mdp) / scale)
        else:
            return 1 / (1 + sp.exp((mdp - V) / scale))

    def h(self, cac):
        """Channel gating kinetics. Functions of membrane voltage"""
        q = self.inf(cac, 'h')
        return 1 + (q - 1) * self._param['h__alpha']

    def g_Ca(self, e, f, h):
        return self._param['g_Ca'] * e ** 2 * f * h

    def I_Ca(self, V, e, f, h):
        """
        Membrane current (in uA/cm^2)
        Calcium (Ca = element name)
        """
        return self._param['g_Ca'] * e ** 2 * f * h * (V - self._param['E_Ca'])

    def g_Kf(self, p, q):
        return self._param['g_Kf'] * p ** 4 * q

    def I_Kf(self, V, p, q):
        """
        Membrane current (in uA/cm^2)
        Potassium (K = element name)
        """
        return self._param['g_Kf'] * p ** 4 * q * (V - self._param['E_K'])

    def g_Ks(self, n):
        return self._param['g_Ks'] * n

    def I_Ks(self, V, n):
        """
        Membrane current (in uA/cm^2)
        Potassium (K = element name)
        """
        return self._param['g_Ks'] * n * (V - self._param['E_K'])

    #  Leak
    def I_L(self, V):
        """
        Membrane current (in uA/cm^2)
        Leak
        """
        return self._param['g_L'] * (V - self._param['E_L'])

    """default self"""

    @staticmethod
    def step_model(X, i_inj, self):
        """
        Integrate and update voltage after one time step
        Parameters
        ----------
        X : array
            state vector
        i_inj : array
            input current
        self : neuron object
        Returns
        ----------
        An updated state vector
        """
        V = X[V_pos]
        p = X[1]
        q = X[2]
        n = X[3]
        e = X[4]
        f = X[5]
        cac = X[Ca_pos]

        h = self.h(cac)
        # V = V * (i_inj + self.g_Ca(e,f,h)*self._param['E_Ca'] + (self.g_Ks(n)+self.g_Kf(p,q))*self._param['E_K'] + self._param['g_L']*self._param['E_L']) / \
        #     ((self._param['C_m']/self.dt) + self.g_Ca(e,f,h) + self.g_Ks(n) + self.g_Kf(p,q) + self._param['g_L'])
        V += ((i_inj - self.I_Ca(V, e, f, h) - self.I_Ks(V, n) - self.I_Kf(V, p, q) - self.I_L(V)) / self._param[
            'C_m']) * self.dt

        cac += (-self.I_Ca(V, e, f, h) * self._param['rho_ca'] - (
                    (cac - self.REST_CA) / self._param['decay_ca'])) * self.dt
        tau = self._param['p__tau']
        p = ((tau * self.dt) / (tau + self.dt)) * ((p / self.dt) + (self.inf(V, 'p') / tau))
        tau = self._param['q__tau']
        q = ((tau * self.dt) / (tau + self.dt)) * ((q / self.dt) + (self.inf(V, 'q') / tau))
        tau = self._param['e__tau']
        e = ((tau * self.dt) / (tau + self.dt)) * ((e / self.dt) + (self.inf(V, 'e') / tau))
        tau = self._param['f__tau']
        f = ((tau * self.dt) / (tau + self.dt)) * ((f / self.dt) + (self.inf(V, 'f') / tau))
        tau = self._param['n__tau']
        n = ((tau * self.dt) / (tau + self.dt)) * ((n / self.dt) + (self.inf(V, 'n') / tau))

        if self._tensors:
            return tf.stack([V, p, q, n, e, f, cac], 0)
        else:
            return [V, p, q, n, e, f, cac]

    def plot_results(self, ts, i_inj_values, results, ca_true=None, dir='.', suffix="", show=True, save=False):
        """plot all dynamics"""
        V = results[:, 0]
        p = results[:, 1]
        q = results[:, 2]
        n = results[:, 3]
        e = results[:, 4]
        f = results[:, 5]
        cac = results[:, 6]

        h = self.h(cac)
        ica = self.I_Ca(V, e, f, h)
        ik = self.I_Ks(V, n) + self.I_Kf(V, p, q)
        il = self.I_L(V)

        plt.figure()

        plt.subplot(5, 1, 1)
        plt.title('Hodgkin-Huxley Neuron')
        if (V.ndim == 1):
            plt.plot(ts, V, 'k')
        else:
            plt.plot(ts, V)
        plt.ylabel('V (mV)')

        plt.subplot(5, 1, 2)
        if (ca_true is not None):
            plt.plot(ts, ca_true, 'Navy', linestyle='-.', label='real data')
            plt.legend()
        if (V.ndim == 1):
            plt.plot(ts, cac, 'r')
        else:
            plt.plot(ts, cac)
        plt.ylabel('[$Ca^{2+}$]')

        plt.subplot(5, 1, 3)
        plt.plot(ts, ica, utils.RATE_COLORS['f'], label='$I_{Ca}$')
        plt.plot(ts, ik, 'm', label='$I_{K}$')
        plt.plot(ts, il, 'g', label='$I_{L}$')
        plt.ylabel('Current')
        plt.legend()

        plt.subplot(5, 1, 4)
        plt.plot(ts, p, utils.RATE_COLORS['p'], label='p')
        plt.plot(ts, q, utils.RATE_COLORS['q'], label='q')
        plt.plot(ts, n, utils.RATE_COLORS['n'], label='n')
        plt.plot(ts, e, utils.RATE_COLORS['e'], label='e')
        plt.plot(ts, f, utils.RATE_COLORS['f'], label='f')
        plt.plot(ts, h, utils.RATE_COLORS['h'], label='h')
        plt.ylabel('Gating Value')
        plt.legend()

        plt.subplot(5, 1, 5)
        plt.plot(ts, i_inj_values, 'b')
        plt.xlabel('t (ms)')
        plt.ylabel('$I_{inj}$ ($\\mu{A}/cm^2$)')
        # plt.ylim(-1, 40)

        if save:
            plt.savefig('{}results_{}.png'.format(utils.current_dir, suffix), dpi=300)
        if show:
            plt.show()
        plt.close()

    @staticmethod
    def get_random():
        """Returns a dictionnary of random parameters"""
        return give_rand()

    @staticmethod
    def plot_vars(*args, **kwargs):
        """Plot functions for the parameters"""
        return utils.plot_vars(*args, **kwargs)


    @staticmethod
    def ica_from_v(X, v_fix, self):
        e = X[1]
        f = X[2]
        cac = X[Ca_pos]

        h = self.h(cac)
        tau = self._param['e__tau']
        e = ((tau * self.dt) / (tau + self.dt)) * ((e / self.dt) + (self.inf(v_fix, 'e') / tau))
        tau = self._param['f__tau']
        f = ((tau * self.dt) / (tau + self.dt)) * ((f / self.dt) + (self.inf(v_fix, 'f') / tau))
        ica = self.I_Ca(v_fix, e, f, h)
        cac += (-self.I_Ca(v_fix, e, f, h) * self._param['rho_ca'] - (
                (cac - self.REST_CA) / self._param['decay_ca'])) * self.dt

        if self._tensors:
            return tf.stack([ica, e, f, h, cac], 0)
        else:
            return [ica, e, f, h, cac]

    @staticmethod
    def ik_from_v(X, v_fix, self):
        p = X[1]
        q = X[2]
        n = X[3]

        tau = self._param['p__tau']
        p = ((tau * self.dt) / (tau + self.dt)) * ((p / self.dt) + (self.inf(v_fix, 'p') / tau))
        tau = self._param['q__tau']
        q = ((tau * self.dt) / (tau + self.dt)) * ((q / self.dt) + (self.inf(v_fix, 'q') / tau))
        tau = self._param['n__tau']
        n = ((tau * self.dt) / (tau + self.dt)) * ((n / self.dt) + (self.inf(v_fix, 'n') / tau))
        ik = self.I_Kf(v_fix, p, q) + self.I_Ks(v_fix, n)

        if self._tensors:
            return tf.stack([ik, p, q, n], 0)
        else:
            return [ik, p, q, n]