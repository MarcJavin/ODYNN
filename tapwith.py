"""
.. module:: 
    :synopsis: Module doing stuff...

.. moduleauthor:: Marc Javin
"""
import sys

import numpy as np
import scipy as sp

import psutil
from opthh import neuron as nr
from opthh import circuit
from opthh import datas
from opthh import cfg_model, utils
from opthh.coptim import CircuitOpt, CircuitTf
import opthh.csimul as sim

p = cfg_model.NEURON_MODEL.default_params
rand = cfg_model.NEURON_MODEL.get_random


if __name__=='__main__':

    name = sys.argv[1]

    n_parallel = 50
    syns = {(0, 1): circuit.SYNAPSE_inhib,
            (0, 2): circuit.SYNAPSE_inhib,
            (0, 4): circuit.SYNAPSE_inhib,
            (1, 3): circuit.SYNAPSE_inhib,
            (1, 4): circuit.SYNAPSE_inhib,
            (1, 6): circuit.SYNAPSE_inhib,
            (2, 0): circuit.SYNAPSE,
            (2, 4): circuit.SYNAPSE,
            (2, 5): circuit.SYNAPSE,
            (3, 2): circuit.SYNAPSE_inhib,
            (3, 4): circuit.SYNAPSE_inhib,
            (3, 5): circuit.SYNAPSE_inhib,
            (4, 2): circuit.SYNAPSE_inhib,
            (4, 5): circuit.SYNAPSE_inhib,
            (4, 6): circuit.SYNAPSE_inhib,
            (5, 4): circuit.SYNAPSE_inhib,
            (5, 6): circuit.SYNAPSE_inhib,
            (6, 4): circuit.SYNAPSE,
            (6, 5): circuit.SYNAPSE,
            (7, 2): circuit.SYNAPSE_inhib,
            (7, 5): circuit.SYNAPSE_inhib,
            (8, 2): circuit.SYNAPSE_inhib,
            (8, 6): circuit.SYNAPSE_inhib,
            }
    syns_opt = [{(0, 1): circuit.get_syn_rand(False),
                 (0, 2): circuit.get_syn_rand(False),
                 (0, 4): circuit.get_syn_rand(False),
                 (1, 3): circuit.get_syn_rand(False),
                 (1, 4): circuit.get_syn_rand(False),
                 (1, 6): circuit.get_syn_rand(False),
                 (2, 0): circuit.get_syn_rand(True),
                 (2, 4): circuit.get_syn_rand(True),
                 (2, 5): circuit.get_syn_rand(True),
                 (3, 2): circuit.get_syn_rand(False),
                 (3, 4): circuit.get_syn_rand(False),
                 (3, 5): circuit.get_syn_rand(False),
                 (4, 2): circuit.get_syn_rand(False),
                 (4, 5): circuit.get_syn_rand(False),
                 (4, 6): circuit.get_syn_rand(False),
                 (5, 4): circuit.get_syn_rand(False),
                 (5, 6): circuit.get_syn_rand(False),
                 (6, 4): circuit.get_syn_rand(True),
                 (6, 5): circuit.get_syn_rand(True),
                 (7, 2): circuit.get_syn_rand(False),
                 (7, 5): circuit.get_syn_rand(False),
                 (8, 2): circuit.get_syn_rand(False),
                 (8, 6): circuit.get_syn_rand(False),
                 } for i in range(n_parallel)]
    gaps = {(1, 2): circuit.GAP,
            (2, 3): circuit.GAP,
            (2, 4): circuit.GAP,
            (3, 5): circuit.GAP,
            (6, 7): circuit.GAP,
            (7, 8): circuit.GAP,
            }
    gaps_opt = [{(1, 2): circuit.get_gap_rand(),
                 (2, 3): circuit.get_gap_rand(),
                 (2, 4): circuit.get_gap_rand(),
                 (3, 5): circuit.get_gap_rand(),
                 (6, 7): circuit.get_gap_rand(),
                 (7, 8): circuit.get_gap_rand()
                 } for i in range(n_parallel)]
    labels = {0: 'PVD',
              1: 'PLM',
              2: 'PVC',
              3: 'DVA',
              4: 'AVA',
              5: 'AVB',
              6: 'AVD',
              7: 'AVM',
              8: 'ALM'}
    # plt.rcParams['figure.facecolor'] = 'Indigo'

    # c = circuit.CircuitFix([p for _ in range(9)], synapses=syns, gaps=gaps, labels=labels)
    # c.plot()

    dt = 0.2
    t, i = datas.full4(dt=dt, nb_neuron_zero=5)
    i[:, :, :] = i[:, :, [0, 1, 4, 5, 6, 7, 8, 2, 3]]
    _, itest = datas.full4_test(dt=dt, nb_neuron_zero=5)
    itest[:, :, :] = itest[:, :, [0, 1, 4, 5, 6, 7, 8, 2, 3]]


    dir = utils.set_dir('Tapwith_' + name)
    train = sim.simul(pars=[p for _ in range(9)], t=t, i_injs=i, synapses=syns, gaps=gaps, n_out=[4, 5], labels=labels)
    test = sim.simul(pars=[p for _ in range(9)], t=t, i_injs=itest, synapses=syns, gaps=gaps, n_out=[4, 5], labels=labels)
    n = nr.BioNeuronTf([rand() for _ in range(9)], dt=dt)
    ctf = CircuitTf(neurons=n, synapses=syns_opt, gaps=gaps_opt, labels=labels, commands={4, 5},
                    sensors={0, 1, 7, 8})
    copt = CircuitOpt(circuit=ctf)
    copt.opt_circuits(subdir=dir, train=train, test=test, n_out=[4, 5])