"""
.. module:: 
    :synopsis: Module doing stuff...

.. moduleauthor:: Marc Javin
"""
import sys

from odin import circuit
from odin.circuit import CircuitTf
from odin import datas
from odin import utils
from odin import neuron as nr
from odin.models import cfg_model
from odin.coptim import CircuitOpt
import odin.csimul as sim

p = cfg_model.NEURON_MODEL.default_params
rand = cfg_model.NEURON_MODEL.get_random
dt = 0.3
n_parallel = 1

syns = {(0, 1): circuit.SYNAPSE_inhib,
        (0, 2): circuit.SYNAPSE_inhib,
        (0, 4): circuit.SYNAPSE_inhib2,
        (1, 3): circuit.SYNAPSE_inhib,
        (1, 4): circuit.SYNAPSE_inhib2,
        (1, 6): circuit.SYNAPSE_inhib,
        (2, 0): circuit.SYNAPSE2,
        (2, 4): circuit.SYNAPSE2,
        (2, 5): circuit.SYNAPSE1,
        (3, 2): circuit.SYNAPSE_inhib,
        (3, 4): circuit.SYNAPSE_inhib2,
        (3, 5): circuit.SYNAPSE_inhib,
        (4, 2): circuit.SYNAPSE_inhib,
        (4, 5): circuit.SYNAPSE_inhib,
        (4, 6): circuit.SYNAPSE_inhib2,
        (5, 4): circuit.SYNAPSE_inhib,
        (5, 6): circuit.SYNAPSE_inhib2,
        (6, 4): circuit.SYNAPSE1,
        (6, 5): circuit.SYNAPSE2,
        (7, 2): circuit.SYNAPSE_inhib,
        (7, 5): circuit.SYNAPSE_inhib,
        (8, 2): circuit.SYNAPSE_inhib2,
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


def run():
    name = sys.argv[1]

    # plt.rcParams['figure.facecolor'] = 'Indigo'

    # c = circuit.CircuitFix([p for _ in range(9)], synapses=syns, gaps=gaps, labels=labels)
    # c.plot()

    t, i = datas.full4(dt=dt, nb_neuron_zero=5)
    i[:, :, :] = i[:, :, [0, 1, 4, 5, 6, 7, 8, 2, 3]]
    _, itest = datas.full4_test(dt=dt, nb_neuron_zero=5)
    itest[:, :, :] = itest[:, :, [0, 1, 4, 5, 6, 7, 8, 2, 3]]

    dir = utils.set_dir('Tapwith_' + name)
    n_out = [4, 5]
    train = sim.simul(pars=[p for _ in range(9)], t=t, i_injs=i, synapses=syns, gaps=gaps, n_out=n_out, suffix='train', labels=labels)
    test = sim.simul(pars=[p for _ in range(9)], t=t, i_injs=itest, synapses=syns, gaps=gaps, n_out=n_out,
                     labels=labels, suffix='test')
    exit(0)
    # n = nr.BioNeuronTf([rand() for _ in range(9)], dt=dt)
    n = nr.Neurons(
        [nr.NeuronLSTM(dt=dt), nr.NeuronLSTM(dt=dt), nr.BioNeuronTf([p for _ in range(5)], dt=dt), nr.NeuronLSTM(dt=dt),
         nr.NeuronLSTM(dt=dt)])
    ctf = CircuitTf(n, synapses=syns_opt[0], gaps=gaps_opt[0], labels=labels, commands={4, 5}, sensors={0, 1, 7, 8})

    # ctf = CircuitTf.create_random(neurons = n, n_neuron=9, dt=dt, syn_keys={k: v['E']>-60 for k,v in syns.items()}, gap_keys=gaps.keys(), labels=labels, commands={4, 5},
    #                 sensors={0, 1, 7, 8}, n_rand=n_parallel)
    copt = CircuitOpt(circuit=ctf)
    copt.optimize(subdir=dir, train=train, test=test, n_out=[4, 5], l_rate=(0.1, 9, 0.95))

def analyse(dir):
    from odin import optim
    import pandas as pd
    import seaborn as sns
    import pylab as plt
    dir = utils.set_dir(dir)
    dic = optim.get_vars(dir, loss=True)

    train, test = optim.get_data(dir)
    df = pd.DataFrame.from_dict({'loss': dic['loss']})  # .head(4)
    df = df.sort_values('loss').reset_index(drop=True)
    # df = df.dropna()
    sns.barplot(x=df.index, y='loss', data=df)
    plt.show()

    dic = optim.get_best_result(dir)
    ctf = optim.get_model(dir)
    ctf.init_params = dic
    sim.simul(t=train[0], i_injs=train[1], circuit=ctf, show=True, save=False)
    sim.simul(t=test[0], i_injs=test[1], circuit=ctf, show=True, save=False)

if __name__=='__main__':
    analyse('Tapwith_dt0.5')

