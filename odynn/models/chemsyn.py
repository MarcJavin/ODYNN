"""
.. module::
    :synopsis: Module doing stuff...

.. moduleauthor:: Marc Javin
"""

from .model import Model
import random
import numpy as np
import torch


SYNAPSE = {
    'G': 0.05,
    'mdp': -50.,
    'scale': 10.,
    'E': 1.
}
SYNAPSE_inhib = {
    'G': 0.05,
    'mdp': -40.,
    'scale': 1.,
    'E': -80.
}

SYN_VARS = list(SYNAPSE.keys())

MIN_SCALE = 0.05
MAX_SCALE = 100
MIN_MDP = -60.
MAX_MDP = -20.
MIN_G = 1.e-2
MAX_G = 2.

# Class for our new model
class ChemSyn(Model):

    default_params = SYNAPSE_inhib
    _constraints = {'G': np.array([MIN_G, MAX_G]),
                        'scale': np.array([MIN_SCALE, np.infty]),
                        'E' : np.array([-100., -60.])}
    _random_bounds = {
            'G': [MIN_G, MAX_G],
            'mdp': [MIN_MDP, MAX_MDP],
            'scale': [MIN_SCALE, MAX_SCALE],
            'E': [-100., -60.]
        }

    def __init__(self, pres, posts, init_p=None, tensors=False, dt=0.1):
        Model.__init__(self, init_p=init_p, tensors=tensors, dt=dt)
        self._pres = pres
        self._posts = posts

    def _curs(self, vprev, vpost):
        """
        Compute the synaptic current

        Args:
          vprev(ndarray or torch.Tensor): presynaptic voltages
          vpost(ndarray or torch.Tensor): postsynaptic voltages

        Returns:
            ndarray of tf.Tensor: synaptic currents [batch, neuron, model]

        """
        G = self._param['G']
        mdp = self._param['mdp']
        scale = self._param['scale']
        g = G * self._lib.sigmoid((vprev - mdp) / scale)
        return g * (self._param['E'] - vpost)

    def step(self, h, i=None):
        """run one time step

        For tensor :


        Args:
          hprev(ndarray or torch.Tensor): previous state vector

        Returns:
            ndarray or torch.Tensor: updated state vector
        """

        # update synapses
        vpres = h[0, :, self._pres]
        vposts = h[0, :, self._posts]
        if not self._tensors:
            vpres = np.transpose(vpres, (1, 0, 2))
            vposts = np.transpose(vposts, (1, 0, 2))
        curs_intern = self._curs(vpres, vposts)

        curs_post = self._lib.zeros(h.shape[1:])

        for n, i in enumerate(self._posts):
            curs_post[:,i,:] += curs_intern[:,n]

        return curs_post