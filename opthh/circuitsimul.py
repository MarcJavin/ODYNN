"""
.. module:: circuitsimul
    :synopsis: Module for simulation of neural circuits

.. moduleauthor:: Marc Javin
"""

import numpy as np

from . import datas
from . import circuit
from .neuron import V_pos, Ca_pos
from .circuit import CircuitFix
from .utils import plots_output_mult
from .datas import DUMP_FILE
import pickle
import time


class CircuitSimul():
    """
    Simulation of a neuron circuit

    """

    def __init__(self, inits_p, conns, t, i_injs, dt=0.1):
        assert(dt == t[1] - t[0])
        self.circuit = CircuitFix(inits_p=inits_p, conns=conns, dt=dt)
        self.batch = False
        if i_injs.ndim > 2:
            self.batch = True
            self.n_batch = i_injs.shape[1]
            i_injs = np.moveaxis(i_injs, 1, 0)
            self.calculate = np.vectorize(self.calculate, signature='(t,n)->(t,s,n),(t,n)')
        assert (len(inits_p) == i_injs.shape[-1])
        self.connections = conns
        self.t = t
        #[(batch,) t, neuron]
        self.i_injs = i_injs

    def circuit_step(self, curs):
        return self.circuit.step(None, curs)

    def calculate(self, i_inj):
        self.circuit._neurons.reset()
        states = np.zeros((np.hstack((len(self.t), self.circuit._neurons._init_state.shape))))
        curs = np.zeros(i_inj.shape)

        for t in range(len(self.t)):
            if t == 0:
                curs[t, :] = self.circuit_step(curs=i_inj[t, :])
            else:
                curs[t, :] = self.circuit_step(curs=i_inj[t, :] + curs[t - 1, :])
            states[t, :, :] = self.circuit._neurons.state
        return states, curs

    def simul(self, n_out, dump=False, suffix='', show=False, save=True):
        """runs the entire simulation"""
        #[(batch,) time, state, neuron]
        start = time.time()
        states, curs = self.calculate(self.i_injs)
        print(time.time() - start)

        if self.batch:
            for i in range(self.i_injs.shape[0]):
                plots_output_mult(self.t, self.i_injs[i], states[i,:,V_pos,:], states[i,:,Ca_pos,:],
                          i_syn=curs[i], show=show, save=save, suffix='TARGET_%s%s'%(suffix,i))
            # [t, state, (batch,) neuron]
            states = np.moveaxis(states, 0, -2)
            i_injs = np.moveaxis(self.i_injs, 0, 1)
        else:
            plots_output_mult(self.t, self.i_injs, states[:,V_pos,:], states[:,Ca_pos,:],
                          i_syn=curs, show=show, save=save, suffix='TARGET_%s'%suffix)
            #reshape for batch dimension
            states = states[:,:,np.newaxis,:]
            i_injs = self.i_injs[:,np.newaxis,:]

        V = np.stack([states[:, 0, :, n] for n in n_out], axis=-1)
        Ca = np.stack([states[:, -1, :, n] for n in n_out], axis=-1)
        todump = [self.t, i_injs, V, Ca]
        if dump:
            with open(DUMP_FILE, 'wb') as f:
                pickle.dump(todump, f)
            return DUMP_FILE
        else:
            return todump



if __name__ == '__main__':
    from . import hhmodel

    p = hhmodel.DEFAULT
    pars = [p, p]
    t,i = datas.give_train()
    connections = {(0, 1): circuit.SYNAPSE,
                   (1, 0): circuit.SYNAPSE}
    t, i = datas.give_train(nb_neuron_zero=1)
    print("i_inj : ", i.shape)
    c = CircuitSimul(pars, connections, t, i)
    c.simul(dump=False, n_out=1, show=True, save=False)