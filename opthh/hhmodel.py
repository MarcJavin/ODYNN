"""
.. module:: hhmodel
    :synopsis: Module defining a self for C. elegans neurons

.. moduleauthor:: Marc Javin
"""


import numpy as np
import random

import pylab as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
import scipy as sp
from matplotlib import gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter
import collections

from .utils import box, save_show
from . import utils, model
from pylab import plt

RATE_COLORS = {'p' : '#00ccff',
               'q' : '#0000ff',
               'n' : '#cc00ff',
               'e' : '#b30000',
               'f' : '#ff9900',
               'h' : '#ff1a1a'
                }
GATES = ['e', 'f', 'n', 'p', 'q']
CONDS = ['g_Ks', 'g_Kf', 'g_Ca', 'g_L']
MEMB = ['C_m', 'E_K', 'E_Ca', 'E_L']

REST_CA = 0.
INITS = {
    'i': 0,
    'V': -60.,
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

MIN_TAU = 1.
MAX_TAU = 1000.
MAX_G = 10.
MIN_SCALE = 1.
MAX_SCALE = 50.
MIN_MDP = -50.
MAX_MDP = 50.


def give_rand():
    rand_par = {
        'decay_ca': random.uniform(10.,500.),
        'rho_ca': random.uniform(1e-5,10.),
        'p__tau': random.uniform(MIN_TAU, MAX_TAU/5),
        'p__scale': random.uniform(MIN_SCALE, MAX_SCALE),
        'p__mdp': random.uniform(MIN_MDP, MAX_MDP),

        'q__tau': random.uniform(MIN_TAU, MAX_TAU/5),
        'q__scale': random.uniform(-MAX_SCALE, -MIN_SCALE),
        'q__mdp': random.uniform(MIN_MDP, MAX_MDP),

        'n__tau': random.uniform(MAX_TAU/5, MAX_TAU),
        'n__scale': random.uniform(MIN_SCALE, MAX_SCALE),
        'n__mdp': random.uniform(MIN_MDP, MAX_MDP),

        'f__tau': random.uniform(MIN_TAU, MAX_TAU/5),
        'f__scale': random.uniform(-MAX_SCALE, -MIN_SCALE),
        'f__mdp': random.uniform(MIN_MDP, MAX_MDP),

        'e__tau': random.uniform(MIN_TAU, MAX_TAU/5),
        'e__scale': random.uniform(MIN_SCALE, MAX_SCALE),
        'e__mdp': random.uniform(-30., 0.),

        'h__alpha': random.uniform(0.1, 0.9),
        'h__scale': random.uniform(-MAX_SCALE, -MIN_SCALE),
        'h__mdp': random.uniform(1, 100),

        'C_m': random.uniform(0.5, 40.),
        'g_Ca': random.uniform(0.1, MAX_G),
        'g_Ks': random.uniform(0.1, MAX_G),
        'g_Kf': random.uniform(0.1, MAX_G),
        'g_L': random.uniform(0.0001, 0.5),
        'E_Ca': random.uniform(0., 40),
        'E_K': random.uniform(-80, -40.),
        'E_L': random.uniform(-80, -40.),
    }
    return collections.OrderedDict(sorted(rand_par.items(), key=lambda t: t[0]))

CONSTRAINTS = {
    'decay_ca': [1e-1, np.infty],
    'rho_ca': [1e-5, 100.],
    'C_m': [5e-1, np.infty],
    'e__scale': [MIN_SCALE, np.infty],
    'e__tau': [MIN_TAU, np.infty],
    'f__scale': [-np.infty, -MIN_SCALE],
    'f__tau': [MIN_TAU, MAX_TAU],
    'g_Ca': [1e-5, MAX_G],
    'g_Kf': [1e-5, MAX_G],
    'g_Ks': [1e-5, MAX_G],
    'g_L': [1e-5, MAX_G],
    'h__alpha': [0, 1],
    'h__scale': [-np.infty, -MIN_SCALE],
    'n__scale': [MIN_SCALE, np.infty],
    'n__tau': [MIN_TAU, MAX_TAU],
    'p__scale': [MIN_SCALE, np.infty],
    'p__tau': [MIN_TAU, MAX_TAU],
    'q__scale': [-np.infty, -MIN_SCALE],
    'q__tau': [MIN_TAU, MAX_TAU]
}
CONSTRAINTS = collections.OrderedDict(sorted(CONSTRAINTS.items(), key=lambda t: t[0]))

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
DEFAULT = collections.OrderedDict(sorted(DEFAULT.items(), key=lambda t: t[0]))

DEFAULT_3 = {
'decay_ca': 110.,
    'rho_ca': 0.23,
    'p__tau': 2.25518, #ms
    'p__scale': 7.42636, #mV
    'p__mdp': -8.05232, #mV

    'q__tau': 149.963,
    'q__scale': -9.97468,
    'q__mdp': -15.6456,

    'n__tau': 25.0007,
    'n__scale': 15.8512,
    'n__mdp': 19.8741,

    'e__tau': 0.100027,
    'e__scale': 6.74821,
    'e__mdp': -3.3568,

    'f__tau': 150.88,
    'f__scale': -5.03176,
    'f__mdp': 25.1815,

    'h__alpha': 0.282,  # None
    'h__scale': -1.,  # mol per m3
    'h__mdp': 6.42,

    'E_K' : -60,
    'E_Ca' : 40,
    'E_L' : -50,
    'C_m' : 1,
    'g_Ca': 3.,

    'g_Ks': 3.0,
    'g_Kf': 0.07,

    'g_L': 0.005

}


ALL = set(DEFAULT.keys())



class CElegansNeuron(model.BioNeuron):
    """Full Hodgkin-Huxley Model implemented for C. elegans"""


    REST_CA = REST_CA
    _ions = {'$Ca^{2+}$': -1}
    Ca_pos = -1
    """int, Default position of the calcium concentration in state vectors"""
    default_init_state = INIT_STATE
    """initial state for neurons : voltage, rates and $[Ca^{2+}]$"""
    default_params = DEFAULT
    """default parameters as a dictionnary"""
    _constraints_dic = CONSTRAINTS
    """constraints to be applied when optimizing"""

    def __init__(self, init_p=None, tensors=False, dt=0.1):
        model.BioNeuron.__init__(self, init_p=init_p, tensors=tensors, dt=dt)

    def _inf(self, V, rate):
        """Compute the steady state value of a gate activation rate"""
        mdp = self._param['%s__mdp' % rate]
        scale = self._param['%s__scale' % rate]
        if self._tensors:
            # print('V : ', V)
            # print('mdp : ', mdp)
            return tf.sigmoid((V - mdp) / scale)
        else:
            return 1 / (1 + sp.exp((mdp - V) / scale))

    def _h(self, cac):
        """Channel gating kinetics. Functions of membrane voltage"""
        q = self._inf(cac, 'h')
        return 1 + (q - 1) * self._param['h__alpha']

    def _i_ca(self, V, e, f, h):
        """Membrane current (in uA/cm^2) for Calcium"""
        return self._param['g_Ca'] * e ** 2 * f * h * (V - self._param['E_Ca'])

    def _i_kf(self, V, p, q):
        """Membrane current (in uA/cm^2) for Potassium"""
        return self._param['g_Kf'] * p ** 4 * q * (V - self._param['E_K'])

    def _i_ks(self, V, n):
        """Membrane current (in uA/cm^2) for Potassium"""
        return self._param['g_Ks'] * n * (V - self._param['E_K'])

    def _i_leak(self, V):
        """Membrane leak current (in uA/cm^2)"""
        return self._param['g_L'] * (V - self._param['E_L'])

    def step(self, X, i_inj):
        """Integrate and update voltage after one time step

        Args:
          X(np.array or tensor): state vector
          i_inj(np.array or tensor): input current
        
        """
        V = X[self.V_pos]
        p = X[1]
        q = X[2]
        n = X[3]
        e = X[4]
        f = X[5]
        cac = X[self.Ca_pos]

        h = self._h(cac)
        # V = V * (i_inj + self.g_Ca(e,f,h)*self._param['E_Ca'] + (self.g_Ks(n)+self.g_Kf(p,q))*self._param['E_K'] + self._param['g_L']*self._param['E_L']) / \
        #     ((self._param['C_m']/self.dt) + self.g_Ca(e,f,h) + self.g_Ks(n) + self.g_Kf(p,q) + self._param['g_L'])
        V = V + ((i_inj - self._i_ca(V, e, f, h) - self._i_ks(V, n) - self._i_kf(V, p, q) - self._i_leak(V)) / self._param[
            'C_m']) * self.dt

        cac = cac + (-self._i_ca(V, e, f, h) * self._param['rho_ca'] - (
                    (cac - self.REST_CA) / self._param['decay_ca'])) * self.dt
        tau = self._param['p__tau']
        p = ((tau * self.dt) / (tau + self.dt)) * ((p / self.dt) + (self._inf(V, 'p') / tau))
        tau = self._param['q__tau']
        q = ((tau * self.dt) / (tau + self.dt)) * ((q / self.dt) + (self._inf(V, 'q') / tau))
        tau = self._param['e__tau']
        e = ((tau * self.dt) / (tau + self.dt)) * ((e / self.dt) + (self._inf(V, 'e') / tau))
        tau = self._param['f__tau']
        f = ((tau * self.dt) / (tau + self.dt)) * ((f / self.dt) + (self._inf(V, 'f') / tau))
        tau = self._param['n__tau']
        n = ((tau * self.dt) / (tau + self.dt)) * ((n / self.dt) + (self._inf(V, 'n') / tau))

        if self._tensors:
            return tf.stack([V, p, q, n, e, f, cac], 0)
        else:
            return np.array([V, p, q, n, e, f, cac])

    def plot_results(self, ts, i_inj_values, results, ca_true=None, suffix="", show=True, save=False):
        """plot all dynamics

        Args:
          ts: 
          i_inj_values: 
          results: 
          ca_true:  (Default value = None)
          suffix:  (Default value = "")
          show(bool): If True, show the figure (Default value = True)
          save(bool): If True, save the figure (Default value = False)

        Returns:

        """
        V = results[:, 0]
        p = results[:, 1]
        q = results[:, 2]
        n = results[:, 3]
        e = results[:, 4]
        f = results[:, 5]
        cac = results[:, 6]

        h = self._h(cac)
        ica = self._i_ca(V, e, f, h)
        ik = self._i_ks(V, n) + self._i_kf(V, p, q)
        il = self._i_leak(V)

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
        plt.plot(ts, ica, RATE_COLORS['f'], label='$I_{Ca}$')
        plt.plot(ts, ik, 'm', label='$I_{K}$')
        plt.plot(ts, il, 'g', label='$I_{L}$')
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
        plt.plot(ts, i_inj_values, 'b')
        plt.xlabel('t (ms)')
        plt.ylabel('$I_{inj}$ ($\\mu{A}/cm^2$)')
        # plt.ylim(-1, 40)

        utils.save_show(show, save, name='Results_{}'.format(suffix), dpi=300)

    @staticmethod
    def get_random():
        """Returns a dictionnary of random parameters"""
        return give_rand()

    @staticmethod
    def boxplot_vars(var_dic, suffix="", show=False, save=True):
        df = pd.DataFrame.from_dict(var_dic)
        plt.figure()
        plt.subplot(121)
        cols = [RATE_COLORS['n'], RATE_COLORS['n'], RATE_COLORS['f'], 'k']
        box(df, cols, CONDS)
        plt.title('Conductances')
        plt.subplot(122)
        cols = ['b', RATE_COLORS['n'], RATE_COLORS['f'], 'k']
        box(df, cols, MEMB)
        plt.title('Membrane')
        save_show(show, save, name='{}MEmbrane_{}'.format(utils.NEUR_DIR, suffix).format(suffix), dpi=300)

        plt.figure()
        plt.subplot(211)
        box(df, ['#666666'], ['rho_ca'])
        plt.title('Rho_ca')
        plt.yscale('log')
        plt.subplot(212)
        box(df, ['#0000ff'], ['decay_ca'])  # , 'b')
        plt.title('Decay_ca')
        plt.tight_layout()
        save_show(show, save, name='{}CalciumPump_{}'.format(utils.NEUR_DIR, suffix), dpi=300)

        plt.figure()
        for i, type in enumerate(['mdp', 'scale', 'tau']):
            plt.subplot(3, 1, i + 1)
            plt.title(type)
            labels = ['{}__{}'.format(rate, type) for rate in RATE_COLORS.keys()]
            cols = RATE_COLORS.values()
            if (type == 'tau'):
                labels = ['h__alpha' if x == 'h__tau' else x for x in labels]
                plt.yscale('log')
            box(df, cols, labels)
        save_show(show, save, name='{}Rates_{}'.format(utils.NEUR_DIR, suffix), dpi=300)

    @classmethod
    def plot_vars(cls, var_dic, suffix="evolution", show=False, save=True, func=utils.plot):
        """plot variation/comparison/boxplots of all variables organized by categories

        Args:
          var_dic:
          suffix:  (Default value = "")
          show(bool): If True, show the figure (Default value = True)
          save(bool): If True, save the figure (Default value = False)
          func:  (Default value = plot)

        Returns:

        """
        fig = plt.figure()
        grid = plt.GridSpec(2, 3)
        for nb in range(len(GATES)):
            gate = GATES[nb]
            cls.plot_vars_gate(gate, var_dic['{}__mdp'.format(gate)], var_dic['{}__scale'.format(gate)],
                           var_dic['{}__tau'.format(gate)], fig, grid[nb], (nb % 3 == 0), func=func)
        cls.plot_vars_gate('h', var_dic['h__mdp'], var_dic['h__scale'],
                       var_dic['h__alpha'], fig, grid[5], False, func=func)
        plt.tight_layout()
        save_show(show, save, name='{}Rates_{}'.format(utils.NEUR_DIR, suffix), dpi=300)

        fig = plt.figure()
        grid = plt.GridSpec(1, 2)
        subgrid = gridspec.GridSpecFromSubplotSpec(4, 1, grid[0], hspace=0.1)
        for i, var in enumerate(CONDS):
            ax = plt.Subplot(fig, subgrid[i])
            func(ax, var_dic[var])  # )
            ax.set_ylabel(var)
            if (i == 0):
                ax.set_title('Conductances')
            fig.add_subplot(ax)
        subgrid = gridspec.GridSpecFromSubplotSpec(4, 1, grid[1], hspace=0.1)
        for i, var in enumerate(MEMB):
            ax = plt.Subplot(fig, subgrid[i])
            func(ax, var_dic[var])  # )
            ax.set_ylabel(var)
            if (i == 0):
                ax.set_title('Membrane')
            ax.yaxis.tick_right()
            fig.add_subplot(ax)
        plt.tight_layout()
        save_show(show, save, name='{}Membrane_{}'.format(utils.NEUR_DIR, suffix), dpi=300)

        plt.figure()
        ax = plt.subplot(211)
        func(ax, var_dic['rho_ca'])  # , 'r')
        plt.ylabel('Rho_ca')
        plt.yscale('log')
        ax = plt.subplot(212)
        func(ax, var_dic['decay_ca'])  # , 'b')
        plt.ylabel('Decay_ca')
        save_show(show, save, name='{}CalciumPump_{}'.format(utils.NEUR_DIR, suffix), dpi=300)

    @staticmethod
    def plot_vars_gate(name, mdp, scale, tau, fig, pos, labs, func=utils.plot):
        """plot the gates variables

        Args:
          name:
          mdp:
          scale:
          tau:
          fig:
          pos:
          labs:
          func:  (Default value = plot)

        Returns:

        """
        subgrid = gridspec.GridSpecFromSubplotSpec(3, 1, pos, hspace=0.1)
        vars = [('Midpoint', mdp), ('Scale', scale), ('Tau', tau)]
        keys = ['mdp', 'scale', 'tau']
        if name=='h':
            keys[-1] = 'alpha'
        for i, var in enumerate(vars):
            ax = plt.Subplot(fig, subgrid[i])
            func(ax, var[1])  # , 'r')
            ax.set_xlabel([])
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            if (labs):
                ax.set_ylabel(var[0])
            if (i == 0):
                ax.set_title(name)
            fig.add_subplot(ax)

    @classmethod
    def study_vars(cls, p, suffix='', show=False, save=True):
        cls.plot_vars(p, func=utils.bar, suffix='compared_%s'%suffix, show=show, save=save)
        cls.boxplot_vars(p, suffix='boxes_%s'%suffix, show=show, save=save)

        if p['C_m'].shape != (1,):
            corr = pd.DataFrame(p).corr()
            plt.subplots(figsize=(11, 9))
            # Generate a custom diverging colormap
            cmap = sns.diverging_palette(220, 10, as_cmap=True)
            # Draw the heatmap with the mask and correct aspect ratio
            sns.heatmap(corr, cmap=cmap, center=0,
                        square=True, linewidths=.5, cbar_kws={"shrink": .5})
            save_show(show=show, save=save, name='{}Correlation_{}'.format(utils.NEUR_DIR, suffix))

    @staticmethod
    def ica_from_v(X, v_fix, self):
        e = X[1]
        f = X[2]
        cac = X[self.Ca_pos]

        h = self._h(cac)
        tau = self._param['e__tau']
        e = ((tau * self.dt) / (tau + self.dt)) * ((e / self.dt) + (self._inf(v_fix, 'e') / tau))
        tau = self._param['f__tau']
        f = ((tau * self.dt) / (tau + self.dt)) * ((f / self.dt) + (self._inf(v_fix, 'f') / tau))
        ica = self._i_ca(v_fix, e, f, h)
        cac += (-self._i_ca(v_fix, e, f, h) * self._param['rho_ca'] - (
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
        p = ((tau * self.dt) / (tau + self.dt)) * ((p / self.dt) + (self._inf(v_fix, 'p') / tau))
        tau = self._param['q__tau']
        q = ((tau * self.dt) / (tau + self.dt)) * ((q / self.dt) + (self._inf(v_fix, 'q') / tau))
        tau = self._param['n__tau']
        n = ((tau * self.dt) / (tau + self.dt)) * ((n / self.dt) + (self._inf(v_fix, 'n') / tau))
        ik = self._i_kf(v_fix, p, q) + self._i_ks(v_fix, n)

        if self._tensors:
            return tf.stack([ik, p, q, n], 0)
        else:
            return [ik, p, q, n]


def plots_ica_from_v(ts, V, results, name="ica", show=False, save=True):
    """plot i_ca and Ca conc depending on the voltage

    Args:
      ts:
      V:
      results:
      name:  (Default value = "ica")
      show(bool): If True, show the figure (Default value = True)
      save(bool): If True, save the figure (Default value = False)

    Returns:

    """
    ica = results[:, 0]
    e = results[:, 1]
    f = results[:, 2]
    h = results[:, 3]
    cac = results[:, -1]

    plt.figure()

    plt.subplot(4, 1, 1)
    plt.title('Hodgkin-Huxley Neuron : I_ca from a fixed V')
    plt.plot(ts, ica, 'b')
    plt.ylabel('I_ca')

    plt.subplot(4, 1, 2)
    plt.plot(ts, cac, 'r')
    plt.ylabel('$Ca^{2+}$ concentration')

    plt.subplot(4, 1, 3)
    plt.plot(ts, e, RATE_COLORS['e'], label='e')
    plt.plot(ts, f, RATE_COLORS['f'], label='f')
    plt.plot(ts, h, RATE_COLORS['h'], label='h')
    plt.ylabel('Gating Value')
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(ts, V, 'k')
    plt.ylabel('V (input) (mV)')
    plt.xlabel('t (ms)')

    save_show(show, save, name)


def plots_ik_from_v(ts, V, results, name="ik", show=True, save=False):
    """plot i_k depending on the voltage

    Args:
      ts:
      V:
      results:
      name:  (Default value = "ik")
      show(bool): If True, show the figure (Default value = True)
      save(bool): If True, save the figure (Default value = False)

    Returns:

    """
    ik = results[:, 0]
    p = results[:, 1]
    q = results[:, 2]
    n = results[:, 3]

    plt.figure()

    plt.subplot(3, 1, 1)
    plt.title('Hodgkin-Huxley Neuron : I_ca from a fixed V')
    plt.plot(ts, ik, 'b')
    plt.ylabel('I_k')

    plt.subplot(3, 1, 2)
    plt.plot(ts, p, RATE_COLORS['p'], label='p')
    plt.plot(ts, q, RATE_COLORS['q'], label='q')
    plt.plot(ts, n, RATE_COLORS['n'], label='n')
    plt.ylabel('Gating Value')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(ts, V, 'k')
    plt.ylabel('V (input) (mV)')
    plt.xlabel('t (ms)')

    save_show(show, save, name)
    plt.close()