"""
.. module:: 
    :synopsis: Module doing stuff...

.. moduleauthor:: Marc Javin
"""

from .model import NeuronModel
from odynn import utils
from pylab import plt
import random
import numpy as np
import torch
import collections

MIN_TAU = 1.
MAX_TAU = 1000.
MIN_SCALE = 1.
MAX_SCALE = 200.

# Class for our new model
class HodgHuxSimple(NeuronModel):
    # Our model has membrane conductance as its only parameter
    default_params = {'C_m': 1., 'g_L': 0.1, 'E_L': -60.,
                      'g_K': 0.5,
                      'E_K': 30.,
                      'a__mdp': -30.,
                      'a__scale': 20.,
                      'a__tau': 500.,
                      'b__mdp': -5.,
                      'b__scale': -3.,
                      'b__tau': 30.,
                      }
    default_params = collections.OrderedDict(sorted(default_params.items(), key=lambda t: t[0]))
    # Initial value for the voltage
    default_init_state = np.array([-60., 0., 1.])
    _constraints = {'C_m': [0.5, 40.],
                        'g_L': [1e-9, 10.],
                        'g_K': [1e-9, 10.],
                        'a__scale': [MIN_SCALE, MAX_SCALE],
                        'a__tau': [MIN_TAU, MAX_TAU],
                        'b__scale': [-MAX_SCALE, -MIN_SCALE],
                        'b__tau': [MIN_TAU, MAX_TAU]
                    }
    _random_bounds =  {'C_m': [0.5, 40.],
            'g_L': [1e-5, 10.],
            'g_K': [1e-5, 10.],
            'E_L': [-70., -45.],
            'E_K': [-40., 30.],
            'a__tau': [MIN_TAU, MAX_TAU],
            'a__scale': [MIN_SCALE, MAX_SCALE],
            'a__mdp': [-50., 0.],
            'b__tau': [MIN_TAU, MAX_TAU],
            'b__scale': [-MAX_SCALE, -MIN_SCALE],
            'b__mdp': [-30., 20.],
            }

    def __init__(self, init_p, tensors=False, dt=0.1):
        NeuronModel.__init__(self, init_p=init_p, tensors=tensors, dt=dt)

    def _i_K(self, a, b, V):
        return self._param['g_K'] * a**3 * b * (self._param['E_K'] - V)

    def _i_L(self, V):
        return self._param['g_L'] * (self._param['E_L'] - V)

    def step(self, X, i_inj):
        # Update the voltage
        V = X[0]
        a = X[1]
        b = X[2]
        V = V + self.dt * (i_inj + self._i_L(V) + self._i_K(a, b, V)) / self._param['C_m']
        a = self._update_gate(a, 'a', V)
        b = self._update_gate(b, 'b', V)
        if self._tensors:
            return torch.stack([V, a, b])
        else:
            return np.array([V, a, b])

    def plot_results(self, ts, i_inj_values, results, ca_true=None, suffix="", show=True, save=False):

        V = results[:, 0]
        a = results[:, 1]
        b = results[:, 2]

        il = self._i_L(V)
        ik = self._i_K(a, b, V)

        plt.figure()

        plt.subplot(4, 1, 1)
        plt.plot(ts, V, 'k')
        plt.title('Leaky Integrator Neuron')
        plt.ylabel('V (mV)')

        plt.subplot(4, 1, 2)
        plt.plot(ts, il, 'g', label='$I_{L}$')
        plt.plot(ts, ik, 'c', label='$I_{K}$')
        plt.ylabel('Current')
        plt.legend()

        plt.subplot(4, 1, 3)
        plt.plot(ts, a, 'c', label='a')
        plt.plot(ts, b, 'b', label='b')
        plt.ylabel('Gating Value')
        plt.legend()

        plt.subplot(4, 1, 4)
        plt.plot(ts, i_inj_values, 'b')
        plt.xlabel('t (ms)')
        plt.ylabel('$I_{inj}$ ($\\mu{A}/cm^2$)')
        # plt.ylim(-1, 40)

        utils.save_show(show, save, name='Results_{}'.format(suffix), dpi=300)