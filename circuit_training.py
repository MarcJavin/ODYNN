"""
.. module:: circuit_training
    :synopsis: Module containing functions to organize the training of circuits

.. moduleauthor:: Marc Javin
"""

import sys

import numpy as np
import scipy as sp

from opthh import neuron as nr
from opthh import circuit
from opthh import datas
from opthh import config_model, utils
from opthh.circopt import CircuitOpt, CircuitTf
import opthh.circsimul as sim

p = config_model.NEURON_MODEL.default_params

def inhibit():
    inhib =circuit.SYNAPSE_inhib
    connections = {(0,1) : inhib, (1,0) : inhib}
    t = np.array(sp.arange(0.0, 2000.,datas.DT))
    i0 = 10.*((t>300)&(t<350)) + 20.*((t>900)&(t<950))
    i1 = 10.*((t>500)&(t<550)) + 20.*((t>700)&(t<750)) + 6.*((t>1100)&(t<1300)) + 7.5*((t>1600)&(t<1800))
    i_injs = np.array([i0, i1]).transpose()
    sim.simul(t, i_injs, [p, p], connections, show=True, save=False)


def opt_neurons():
    connections = {(0, 1):circuit.SYNAPSE_inhib,
                   (1, 0):circuit.SYNAPSE}
    t = np.array(sp.arange(0.0, 1000.,datas.DT))
    i0 = 10. * ((t > 200) & (t < 400)) + 30. * ((t > 500) & (t < 600))
    i1 = 30. * ((t > 700) & (t < 800))
    i_injs = np.array([i0, i1]).transpose()
    f = sim.simul(t, i_injs, [p, p], connections, dump=True)
    c = CircuitOpt([p, p], connections)
    c.opt_neurons(f)



def test(nb_neuron, conns, conns_opt, dir, t, i_injs, n_out=[1]):
    pars = [p for _ in range(nb_neuron)]
    dir = utils.set_dir(dir)
    print("Feed with current of shape : ", i_injs.shape)

    train = sim.simul(t, i_injs, pars, conns, n_out=n_out, dump=False, show=False)
    n = nr.BioNeuronTf(init_p=pars, fixed='all', dt=t[1]-t[0])
    cr = CircuitTf(n, synapses=conns_opt)
    c = CircuitOpt(cr)
    c.opt_circuits(dir, n_out=n_out, train=train)

def full4to1():
    t,i = datas.full4(nb_neuron_zero=1)
    n_neuron = 5
    conns = {(0, 4):circuit.SYNAPSE,
             (1, 4):circuit.SYNAPSE,
             (2, 4):circuit.SYNAPSE,
             (3, 4):circuit.SYNAPSE,
             }
    conns_opt = {(0, 4):circuit.get_syn_rand(),
             (1, 4):circuit.get_syn_rand(),
             (2, 4):circuit.get_syn_rand(),
             (3, 4):circuit.get_syn_rand(),
             }
    dir = '4to1-test'
    test(n_neuron, conns, conns_opt, dir, t, i, n_out=[4])


def full441():
    t, i =datas.full4(nb_neuron_zero=6)
    print(i.shape)
    n_neuron = 10
    conns = {(0, 4):circuit.SYNAPSE,
             (1, 4):circuit.SYNAPSE,
             (2, 4):circuit.SYNAPSE,
             (3, 4):circuit.SYNAPSE,
             (0, 5):circuit.SYNAPSE1,
             (1, 5):circuit.SYNAPSE1,
             (2, 5):circuit.SYNAPSE1,
             (3, 5):circuit.SYNAPSE1,
             (0, 6):circuit.SYNAPSE2,
             (1, 6):circuit.SYNAPSE2,
             (2, 6):circuit.SYNAPSE2,
             (3, 6):circuit.SYNAPSE2,
             (0, 7):circuit.SYNAPSE,
             (1, 7):circuit.SYNAPSE1,
             (2, 7):circuit.SYNAPSE,
             (3, 7):circuit.SYNAPSE1,
             (4, 8):circuit.SYNAPSE1,
             (5, 8):circuit.SYNAPSE,
             (6, 8):circuit.SYNAPSE,
             (7, 8):circuit.SYNAPSE2,
             (4, 9):circuit.SYNAPSE,
             (5, 9):circuit.SYNAPSE2,
             (6, 9):circuit.SYNAPSE1,
             (7, 9):circuit.SYNAPSE,
             }
    conns_opt = dict([(k,circuit.get_syn_rand()) for k in conns.keys()])
    dir = '4to4to2-test'
    test(n_neuron, conns, conns_opt, dir, t, i, n_out=[8, 9])
    exit(0)


def with_LSTM():
    dir = utils.set_dir('withLSTM')
    conns = {(0, 1): circuit.SYNAPSE}
    conns_opt = {(0, 1): circuit.get_syn_rand(True)}

    dt = 0.5
    t, i = datas.give_train(dt=dt)
    i_1 = np.zeros(i.shape)
    i_injs = np.stack([i, i_1], axis=2)
    train = sim.simul(t, i_injs, [p, p], conns, n_out=[0, 1], dump=False, show=False)

    neurons = nr.Neurons([nr.BioNeuronTf(config_model.NEURON_MODEL.get_random(), fixed=[], dt=dt), nr.BioNeuronTf(p, fixed='all', dt=dt)])
    c = CircuitTf(neurons=neurons, synapses=conns_opt)
    co = CircuitOpt(circuit=c)
    co.opt_circuits(dir, train=train, n_out=[0,1], l_rate=(0.01,9,0.95))


def tap_with(dir, name):

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

    dt = 0.5
    t, i = datas.full4(dt=dt, nb_neuron_zero=5)
    i[:,:,:] = i[:,:,[0,1,4,5,6,7,8,2,3]]
    t, itest = datas.full4_test(dt=dt, nb_neuron_zero=5)
    itest[:, :, :] = itest[:, :, [0, 1, 4, 5, 6, 7, 8, 2, 3]]


    dir = utils.set_dir(dir + '_' + name)
    train = sim.simul(pars=[p for _ in range(9)], t=t, i_injs=i, synapses=syns, gaps=gaps, n_out=[4,5], labels=labels)
    test = sim.simul(pars=[p for _ in range(9)], t=t, i_injs=itest, synapses=syns, gaps=gaps, n_out=[4, 5], labels=labels)
    n = nr.BioNeuronTf([p for _ in range(9)], fixed='all', dt=dt)
    ctf = CircuitTf(neurons=n, synapses=syns_opt, gaps=gaps_opt, labels=labels, commands=set([4,5]), sensors=set([0,1,7,8]))
    copt = CircuitOpt(circuit=ctf)
    copt.opt_circuits(subdir=dir, train=train, test=test, n_out=[4,5])



if __name__ == '__main__':

    tap_with()
    exit(0)


    xp = sys.argv[1]
    if(xp == '21exc'):
        n_neuron = 2
        conns = {(0,1) :circuit.SYNAPSE}
        conns_opt = {(0,1) :circuit.get_syn_rand(True)}
        dir = '2n-1exc-test'
    elif(xp == 'tw'):
        tap_with(dir = 'Tapwith', name=sys.argv[2])
    elif(xp=='21inh'):
        n_neuron = 2
        conns = {(0,1):circuit.SYNAPSE_inhib}
        conns_opt = {(0,1):circuit.get_syn_rand(False)}
        dir = '2n-1inh-test'
    elif(xp == '22exc'):
        n_neuron = 2
        conns = {(0,1):circuit.SYNAPSE,
                 (1,0):circuit.SYNAPSE}
        conns_opt = [{(0,1):circuit.get_syn_rand(True),
                      (1,0):circuit.get_syn_rand(True)} for _ in range(100)]
        dir = '2n-2exc-test'
    elif(xp == '21exc1inh'):
        n_neuron = 2
        conns = {(0, 1):circuit.SYNAPSE,
                 (1, 0):circuit.SYNAPSE_inhib}
        conns_opt = {(0, 1):circuit.get_syn_rand(True),
                     (1, 0):circuit.get_syn_rand(False)}
        dir = '2n-1exc1inh-test'
    elif (xp == '22inh'):
        n_neuron = 2
        conns = {(0, 1):circuit.SYNAPSE_inhib,
                 (1, 0):circuit.SYNAPSE_inhib}
        conns_opt = {(0, 1):circuit.get_syn_rand(False),
                     (1, 0):circuit.get_syn_rand(False)}
        dir = '2n-2inh-test'
    t, i =datas.give_train(dt=0.5)
    i_1 = np.zeros(i.shape)
    i_injs = np.stack([i, i_1], axis=2)
    test(n_neuron, conns, conns_opt, dir, t, i_injs)
