"""
.. module:: cls
    :synopsis: Module containing basic cls abstract class

.. moduleauthor:: Marc Javin
"""

import numpy as np
import torch
from odynn.models import neur_models, syn_models

class Circuit(object):

    def __init__(self, synapses=[], neuron_model='leakint', syn_model='classic', tensors=False, dt=0.1):
        self._tensors = tensors
        syns = list(zip(*synapses))
        self._synapses = syn_models[syn_model](syns[0], syns[1], tensors=tensors)
        nb_neurons = len(np.unique(np.hstack((syns[0], syns[1]))))
        self.n_synapse = len(syns[0])
        nmod = neur_models[neuron_model]
        self._neurons = nmod(init_p=[nmod.get_random() for _ in range(nb_neurons)], tensors=tensors, dt=dt)
        self._init_state = self._neurons.init_state


    def calculate(self, i_inj):
        for k, v in self._neurons._param.items():
            self._neurons._param[k] = v[...,None]
        init = self._neurons.init_state[:,None,:,None]
        X = [init]
        if i_inj.ndim == 1:
            i_inj = i_inj[:,None,None,None]
        elif i_inj.ndim == 2:
            i_inj = i_inj[:,None,:,None]
        elif i_inj.ndim == 3:
            i_inj = i_inj[:,:,:,None]
        for i in i_inj:
            syn_cur = self._synapses.step(X[-1])
            i += syn_cur
            X.append(self._neurons.step(X[-1], i))
        if self._tensors:
            return torch.cat(X[1:])
        else:
            return np.array(X[1:])
        
            