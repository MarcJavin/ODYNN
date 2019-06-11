"""
.. module::
    :synopsis: Module doing stuff...

.. moduleauthor:: Marc Javin
"""

from .model import Model
import numpy as np


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

MIN_SCALE = 0.2
MAX_SCALE = 100
MIN_MDP = 0.
MAX_MDP = 1.
MIN_G = 0.01
MAX_G = 1.

# Class for our new model
class ChemSyn(Model):

    default_params = SYNAPSE_inhib
    _constraints = {'G': np.array([MIN_G, MAX_G]),
                    'mdp': [MIN_MDP, MAX_MDP],
                        'scale': np.array([MIN_SCALE, np.infty]),
                        'E' : np.array([-0.5, 2.])}
    _random_bounds = {
            'G': [MIN_G, MAX_G/4],
            'mdp': [MIN_MDP, MAX_MDP],
            'scale': [MIN_SCALE, MAX_SCALE],
            'E': [-0.5, 1.]
        }

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
        G = self._param['G']
        mdp = self._param['mdp']
        scale = self._param['scale']
        g = G * self._lib.sigmoid((vprev - mdp) / scale)
        return g * (self._param['E'] - vpost)

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

        for n, i in enumerate(self._posts):
            curs_post[:,i] += curs_intern[:,n]

        return curs_post, x

# Class for our new model
class ChemSynTau(ChemSyn):

    tau = 200.

    def __init__(self, pres, posts, init_p=None, tensors=False, dt=0.1):
        ChemSyn.__init__(self, pres, posts, init_p=init_p, tensors=tensors, dt=dt)

    def _curs(self, vprev, vpost, previ):
        """
        Compute the synaptic current

        Args:
          vprev(ndarray or torch.Tensor): presynaptic voltages
          vpost(ndarray or torch.Tensor): postsynaptic voltages

        Returns:
            ndarray of tf.Tensor: synaptic currents [batch, neuron, model]

        """
        cur_inf = ChemSyn._curs(self, vprev, vpost)
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

        for n, i in enumerate(self._posts):
            curs_post[:,i,:] += x[:,n]

        return curs_post, x



class ChemSynLin(ChemSyn):

    def __init__(self, pres, posts, init_p=None, tensors=False, dt=0.1):
        ChemSyn.__init__(self, pres, posts, init_p=init_p, tensors=tensors, dt=dt)
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
        G = self._param['G']
        mdp = self._param['mdp']
        scale = self._param['scale']
        slope = (vprev - mdp) * scale + 0.5
        g = G * self._lib.clamp(slope, 0, 1)
        return g * (self._param['E'] - vpost)
