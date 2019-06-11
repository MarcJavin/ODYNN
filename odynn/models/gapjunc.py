"""
.. module::
    :synopsis: Module doing stuff...

.. moduleauthor:: Marc Javin
"""

from .model import Model
import random
import numpy as np
import torch


MIN_G = 0.01
MAX_G = 2.

# Class for our new model
class GapJunction(Model):

    default_params = {'G_gap': 0.5, 'factor':1}
    _constraints = {'G_gap': np.array([MIN_G, MAX_G])}
    _random_bounds = {'G_gap': [MIN_G, MAX_G/4], 'factor': [0.5,2.]}

    def __init__(self, pres, posts, init_p=None, tensors=False, dt=0.1):
        Model.__init__(self, init_p=init_p, tensors=tensors, dt=dt)
        self._pres = pres
        self._posts = posts

    def _curs(self, vprev, vpost, previ=None):
        """
        Compute the synaptic current

        Args:
          vprev(ndarray or torch.Tensor): presynaptic voltages
          vpost(ndarray or torch.Tensor): postsynaptic voltages

        Returns:
            ndarray of tf.Tensor: synaptic currents [batch, neuron, model]

        """
        G = self._param['G_gap']
        fac = self._param['factor']
        return G * (vprev * fac - vpost)

    def step(self, h, x=None):
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

        curs_post = self._lib.zeros((h.shape[1], h.shape[2], self.parallel))

        for n, (i,j) in enumerate(zip(self._pres, self._posts)):
            curs_post[:,j,:] += curs_intern[:,n]
            # curs_post[:, i, :] -= curs_intern[:, n]

        return curs_post, x


class GapJunctionTau(GapJunction):

    tau = 200.

    def __init__(self, pres, posts, init_p=None, tensors=False, dt=0.1):
        GapJunction.__init__(self, pres, posts, init_p=init_p, tensors=tensors, dt=dt)

    def _curs(self, vprev, vpost, previ):
        """
        Compute the synaptic current

        Args:
          vprev(ndarray or torch.Tensor): presynaptic voltages
          vpost(ndarray or torch.Tensor): postsynaptic voltages

        Returns:
            ndarray of tf.Tensor: synaptic currents [batch, neuron, model]

        """
        cur_inf = GapJunction._curs(self, vprev, vpost)
        return previ + (cur_inf - previ) / self.tau

    def step(self, h, x):
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
        x = self._curs(vpres, vposts, x)

        curs_post = self._lib.zeros(h.shape[1:])

        for n, (i,j) in enumerate(zip(self._pres, self._posts)):
            curs_post[:,j,:] += x[:,n]
            curs_post[:, i, :] -= x[:, n]

        return curs_post, x