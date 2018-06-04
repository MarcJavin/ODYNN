import numpy as np
from HH_simul import HH_simul
import scipy as sp
import params

class Circuit():

    """
    neurons : objects to optimize

    """
    def __init__(self, neurons, conns, i_injs, init_p=params.SYNAPSE, dt=0.1):
        assert(len(neurons) == i_injs.shape[0])
        self.param = init_p
        self.neurons = neurons
        self.connections = conns
        self.i_injs = i_injs
        self.datas = np.array(neurons.shape)
        self.dt = dt

    def run_sim(self):
        simuls = []
        for n in self.neurons:
            simuls.append(HH_simul(init_p=n.init_p))
        init_states = np.full(len(simuls), simuls[0].get_init_state())
        curs = self.i_injs[0]
        for t in range(len(self.i_injs[0])):
            for n in range(len(simuls)):
                s = simuls[n]
                init_states[n] = s.loop_func(init_states[n], curs[n], self.dt)


    def step(self, curs):
        next_curs = np.zeros[len(curs)]
        #update neurons
        for i, n in enumerate(self.neurons):
            n.step(n.state, curs[i], self.dt, n)
        #update synapses
        for i, n in enumerate(self.neurons):
            vprev = n.state[0]
            syn, post = self.connections[i]
            vpost = self.neurons[post].state[0]
            next_curs[post] += self.syn_curr(self.connections[n], vprev, vpost)
        return next_curs




    def syn_curr(self, syn, vprev, vpost):
        g = self.param['%s__G'%syn] / (1 + sp.exp((self.param['%s_q_mdp'%syn] - vprev)/self.param['%s__scale'%syn]))
        i = g*(self.param['%s__E'%syn] - vpost)
        return i


    def optimize(self):
