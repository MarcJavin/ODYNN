"""
.. module:: cls
    :synopsis: Module containing basic cls abstract class

.. moduleauthor:: Marc Javin
"""

import numpy as np
import torch
from odynn.models import neur_models, syn_models, gap_models

class Circuit():

    def __init__(self, neurons, synapses, gaps):
        assert neurons._tensors == synapses._tensors
        assert neurons._parallel == synapses._parallel, '%s,%s'%(neurons._parallel, synapses._parallel)
        assert neurons._parallel == gaps._parallel, '%s,%s'%(neurons._parallel, gaps._parallel)
        nb_neurons = len(np.unique(np.hstack((synapses._pres, synapses._posts, gaps._pres, gaps._posts))))
        assert nb_neurons == neurons.num, '%s,%s'%(nb_neurons, neurons.num)
        self._tensors = neurons._tensors
        self._parallel = neurons._parallel
        self._synapses = synapses
        self._gaps = gaps
        self._neurons = neurons
        self._init_state = self._neurons.init_state

    @classmethod
    def create_random(cls, synapses, gaps, parallel=1, neuron_model='leakint', syn_model='classic', gap_model='classic', tensors=True, dt=0.1):
        #TODO : initialize synapses
        syns = list(zip(*synapses))
        nb_neurons = len(np.unique(np.hstack((syns[0], syns[1]))))
        n_synapse = len(syns[0])
        n_gap = len(gaps[0])
        nmod = neur_models[neuron_model]
        neurons = nmod.create_random(nb_neurons, parallel, tensors=tensors, dt=dt)
        syns = syn_models[syn_model].create_random(n_synapse, parallel, tensors=tensors)
        gaps = gap_models[gap_model].create_random(n_gap, parallel, tensors=tensors)
        return Circuit(neurons, syns, gaps)

    @property
    def parameters(self):
        return {**self._neurons.parameters, **self._synapses.parameters}

    def step(self, X, i_inj, xsyn, xgap):
        syn_cur, xsyn = self._synapses.step(X, xsyn)
        gap_cur, xgap = self._gaps.step(X, xgap)
        return self._neurons.step(X, i_inj + syn_cur + gap_cur), xsyn, xgap

    def calculate(self, i_inj, init=None):
        if init is None:
            init = np.repeat(self._neurons._init_state[:, None], self._neurons._num, axis=-1)
            init = np.repeat(init[:, :, None], self._parallel, axis=-1)
            init = init[:, None]
        i_inj = np.repeat(i_inj[:, :, None], self._parallel, axis=-1)
        i_inj = i_inj[:, None]
        if self._tensors:
            init = torch.Tensor(init)
            i_inj = torch.Tensor(i_inj)
        X = [init]
        print('Initial states shape : ', init.shape, 'Input current shape : ', i_inj.shape)
        xsyn = 0.
        xgap = 0.
        for i in i_inj:
            x, xsyn, xgap = self.step(X[-1], i, xsyn, xgap)
            X.append(x)
        if self._tensors:
            return torch.stack(X[1:])
        else:
            return np.array(X[1:])

    def apply_constraints(self):
        self._neurons.apply_constraints()
        self._synapses.apply_constraints()
        self._gaps.apply_constraints()

    def plot(self, labels=None, img_size=15):
        """
        Plot the circuit representation and its connections
        :param labels: dict, names for the neurons
        :param img_size: int, size of the image
        """
        import pylab as plt
        import networkx as nx
        NODE_SIZE = 1500
        plt.figure(figsize=(img_size, img_size))

        G = nx.Graph()
        syns = list(zip(self._synapses._pres, self._synapses._posts))
        G.add_edges_from(syns)
        gaps = list(zip(self._gaps._pres, self._gaps._posts))
        G.add_edges_from(gaps)

        pos = nx.layout.spring_layout(G)

        def draw_nodes(shape='o', col='k'):
            nx.draw_networkx_nodes(G, pos, node_shape=shape, node_color=col,
                                   node_size=NODE_SIZE, alpha=0.9)

        draw_nodes()
        nx.draw_networkx_edges(G, pos, edge_color='r', node_size=NODE_SIZE, arrowstyle='->', edgelist=syns,
                               alpha=0.9, width=1, arrowsize=60)
        nx.draw_networkx_edges(G, pos, edge_color='Gold', node_size=NODE_SIZE, edgelist=gaps,
                               alpha=0.9, width=1)

        if labels is not None:
            G = nx.relabel_nodes(G, labels)
            pos = {labels[k]: v for k,v in pos.items()}
        nx.draw_networkx_labels(G, pos, font_color='w', font_weight='bold')

        plt.axis('off')
        plt.draw()
        plt.show()
        plt.close()


            