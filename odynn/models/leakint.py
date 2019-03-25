"""
.. module:: 
    :synopsis: Module doing stuff...

.. moduleauthor:: Marc Javin
"""

from .model import NeuronModel
from odynn import utils
from pylab import plt
import numpy as np
import torch


# Class for our new model
class LeakyIntegrate(NeuronModel):

    default_params = {'C_m': 5., 'g_L': 0.272, 'E_L': -40.8}
    # Initial value for the voltage
    default_init_state = np.array([-35.])
    _constraints = {'C_m': [0.5, 40.],
                        'g_L': [1e-9, 1.]}
    _random_bounds = {'C_m': [0.5, 5],
                'g_L': [1e-5, 1.],
                'E_L': [-70., -45.]}

    def __init__(self, init_p=None, tensors=False, dt=0.1):
        NeuronModel.__init__(self, init_p=init_p, tensors=tensors, dt=dt)

    def _i_L(self, V):
        return self._param['g_L'] * (self._param['E_L'] - V)

    def step(self, X, i_inj):
        # Update the voltage
        V = X[0]
        V = (V * (self._param['C_m'] / self.dt) + (i_inj + self._param['g_L'] * self._param['E_L'])) /\
            ((self._param['C_m'] / self.dt) + self._param['g_L'])
        # V = V + self.dt*(i_inj + self._i_L(V))/self._param['C_m']

        return self._lib.stack([V])

    def plot_results(self, ts, i_inj_values, X, ca_true=None, suffix="", show=True, save=False):

        V = X[:,0]
        il = self._i_L(V)

        plt.figure()

        plt.subplot(3, 1, 1)
        plt.plot(ts, V, 'k')
        plt.title('Leaky Integrator Neuron')
        plt.ylabel('V (mV)')

        plt.subplot(3, 1, 2)
        plt.plot(ts, il, 'g', label='$I_{L}$')
        plt.ylabel('Current')
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(ts, i_inj_values, 'b')
        plt.xlabel('t (ms)')
        plt.ylabel('$I_{inj}$ ($\\mu{A}/cm^2$)')
        # plt.ylim(-1, 40)

        utils.save_show(show, save, name='Results_{}'.format(suffix), dpi=300)