from Circuit_simul import Circuit_simul
from Circuit_opt import Circuit_opt
import utils
import numpy as np
import scipy as sp
import params
import data
import sys

p = params.DEFAULT

def inhibit():
    inhib = params.SYNAPSE_inhib
    connections = {(0,1) : inhib, (1,0) : inhib}
    t = np.array(sp.arange(0.0, 2000., params.DT))
    i0 = 10.*((t>300)&(t<350)) + 20.*((t>900)&(t<950))
    i1 = 10.*((t>500)&(t<550)) + 20.*((t>700)&(t<750)) + 6.*((t>1100)&(t<1300)) + 7.5*((t>1600)&(t<1800))
    i_injs = np.array([i0, i1]).transpose()
    c = Circuit_simul(p, connections, t, i_injs)
    c.run_sim()


def opt_neurons():
    connections = {(0, 1): params.SYNAPSE_inhib,
                   (1, 0): params.SYNAPSE}
    t = np.array(sp.arange(0.0, 1000., params.DT))
    i0 = 10. * ((t > 200) & (t < 400)) + 30. * ((t > 500) & (t < 600))
    i1 = 30. * ((t > 700) & (t < 800))
    i_injs = np.array([i0, i1]).transpose()
    c = Circuit_simul([p, p], connections, t, i_injs)
    f = c.run_sim(dump=True, general=False)
    c = Circuit_opt([p, p], connections)
    c.opt_neurons(f)

def comp_pars(dir):
    p = data.get_vars(dir)
    utils.set_dir(dir)
    utils.plot_vars_syn(p, func=utils.bar, suffix='compared', show=False, save=True)
    utils.plot_vars_syn(p, func=utils.boxplot, suffix='boxes', show=False, save=True)


def test(nb_neuron, conns, conns_opt, dir):
    pars = [p for _ in range(nb_neuron)]
    n_out = 1
    utils.set_dir(dir)
    #
    # t, i = params.give_train()
    # i_1 = np.zeros(i.shape)
    # i_injs = np.stack([i, i_1], axis=1)
    #
    # c = Circuit_simul(pars, conns, t, i_injs)
    # c.run_sim(n_out=1, dump=True)
    c = Circuit_opt(pars, conns_opt)
    c.opt_circuits(dir, n_out=n_out, file=data.DUMP_FILE)
    comp_pars(dir)

if __name__ == '__main__':


    xp = sys.argv[1]
    if(xp == '21exc'):
        n_neuron = 2
        conns = {(0,1) : params.SYNAPSE}
        conns_opt = {(0,1) : params.get_syn_rand(True)}
        dir = '2n-1exc-test'
    elif(xp=='21inh'):
        n_neuron = 2
        conns = {(0,1): params.SYNAPSE_inhib}
        conns_opt = {(0,1): params.get_syn_rand(False)}
        dir = '2n-1inh-test'
    elif(xp == '22exc'):
        n_neuron = 2
        conns = {(0,1): params.SYNAPSE,
                 (1,0): params.SYNAPSE}
        conns_opt = {(0,1): params.get_syn_rand(True),
                     (1,0): params.get_syn_rand(True)}
        dir = '2n-2exc-test'
    elif(xp == '21exc1inh'):
        n_neuron = 2
        conns = {(0, 1): params.SYNAPSE,
                 (1, 0): params.SYNAPSE_inhib}
        conns_opt = {(0, 1): params.get_syn_rand(True),
                     (1, 0): params.get_syn_rand(False)}
        dir = '2n-1exc1inh-test'
    elif (xp == '22inh'):
        n_neuron = 2
        conns = {(0, 1): params.SYNAPSE_inhib,
                 (1, 0): params.SYNAPSE_inhib}
        conns_opt = {(0, 1): params.get_syn_rand(False),
                     (1, 0): params.get_syn_rand(False)}
        dir = '2n-2inh-test'
    test(n_neuron, conns, conns_opt, dir)
