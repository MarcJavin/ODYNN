"""
.. module:: cls
    :synopsis: Module containing basic cls abstract class

.. moduleauthor:: Marc Javin
"""

import numpy as np
import torch
from odynn.models import neur_models, syn_models

class Circuit():

    def __init__(self, neurons, synapses):
        assert neurons._tensors == synapses._tensors
        assert neurons._parallel == synapses._parallel
        nb_neurons = len(np.unique(np.hstack((synapses._pres, synapses._posts))))
        assert nb_neurons == neurons.num
        self._tensors = neurons._tensors
        self._parallel = neurons._parallel
        self._synapses = synapses
        self._neurons = neurons
        self._init_state = self._neurons.init_state

    @classmethod
    def create_random(cls, synapses, parallel=1, neuron_model='leakint', syn_model='classic', tensors=True, dt=0.1):
        syns = list(zip(*synapses))
        nb_neurons = len(np.unique(np.hstack((syns[0], syns[1]))))
        n_synapse = len(syns[0])
        nmod = neur_models[neuron_model]
        neurons = nmod.create_random(nb_neurons, parallel, tensors=tensors, dt=dt)
        syns = syn_models[syn_model].create_random(n_synapse, parallel, tensors=tensors)
        return Circuit(neurons, syns)

    @property
    def parameters(self):
        return {**self._neurons.parameters, **self._synapses.parameters}

    def step(self, X, i):
        syn_cur = self._synapses.step(X)
        return self._neurons.step(X, i + syn_cur)

    def calculate(self, i_inj):
        init = np.repeat(self._neurons._init_state[:, None], self._neurons._num, axis=-1)
        init = np.repeat(init[:, :, None], self._parallel, axis=-1)
        init = init[:, None]
        X = [torch.Tensor(init)]
        # if i_inj.ndim == 1:
        #     i_inj = i_inj[:,None,None,None]
        # elif i_inj.ndim == 2:
        i_inj = i_inj[:,None,:,None]
        # elif i_inj.ndim == 3:
        #     i_inj = i_inj[:,:,:,None]
        for i in i_inj:
            X.append(self.step(X[-1], i))
        if self._tensors:
            return torch.cat(X[1:])
        else:
            return np.array(X[1:])

    def apply_constraints(self):
        self._neurons.apply_constraints()
        self._synapses.apply_constraints()


            