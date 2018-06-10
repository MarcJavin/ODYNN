import numpy as np
from Neuron import HodgkinHuxley
from Circuit import Circuit_fix
from utils import plots_output_mult
from data import DUMP_FILE
import pickle


class Circuit_simul():
    """
    Simulation of a neuron circuit

    """

    def __init__(self, inits_p, conns, t, i_injs, loop_func=HodgkinHuxley.loop_func, dt=0.1):
        self.circuit = Circuit_fix(inits_p=inits_p, conns=conns, loop_func=loop_func, dt=dt)
        self.batch = False
        if (i_injs.ndim > 2):
            self.batch = True
            self.n_batch = i_injs.shape[-1]
            i_injs = np.moveaxis(i_injs, -1, 0)
            print(i_injs.shape)
            self.run_one = np.vectorize(self.run_one, signature='(t,n)->(t,s,n),(t,n)')
        assert (len(inits_p) == i_injs.shape[-1])
        self.connections = conns
        self.t = t
        #[(batch,) t, neuron]
        self.i_injs = i_injs

    def run_one(self, i_inj):
        self.circuit.neurons.reset()
        states = np.zeros((np.hstack((len(self.t), self.circuit.neurons.init_state.shape))))
        curs = np.zeros(i_inj.shape)
        print(curs.shape)

        for t in range(len(self.t)):
            if (t == 0):
                curs[t, :] = self.circuit.step(curs=i_inj[t, :])
            else:
                curs[t, :] = self.circuit.step(curs=i_inj[t, :] + curs[t - 1, :])
            states[t, :, :] = self.circuit.neurons.state
        return states, curs

    """runs the entire simulation"""

    def run_sim(self, n_out, dump=False, show=False, save=True):
        #[(batch,) time, state, neuron]
        states, curs = self.run_one(self.i_injs)

        if(self.batch):
            for i in range(self.i_injs.shape[0]):
                plots_output_mult(self.t, self.i_injs[i], states[i,:,0,:], states[i,:,-1,:],
                          i_syn=curs[i], show=show, save=save, suffix='TARGET')
        else:
            plots_output_mult(self.t, self.i_injs, states[:,0,:], states[:,-1,:],
                          i_syn=curs, show=show, save=save, suffix='TARGET')

        if (dump):
            #[t, state, neuron(, batch)]
            states = np.moveaxis(states, 0, -1)
            i_injs = np.moveaxis(self.i_injs, 0, 1)
            todump = [self.t, i_injs, states[:,0,n_out], states[:,-1,n_out]]
            with open(DUMP_FILE, 'wb') as f:
                pickle.dump(todump, f)
            return DUMP_FILE


if __name__ == '__main__':
    import params

    p = params.DEFAULT
    pars = [p, p]
    t,i = params.give_train()
    connections = {(0, 1): params.SYNAPSE,
                   (1, 0): params.SYNAPSE}
    t, i = params.give_train()
    i_0 = np.zeros(i.shape)
    i_injs = np.stack([i, i_0], axis=1)
    print("i_inj : ", i_injs.shape)
    c = Circuit_simul(pars, connections, t, i_injs)
    c.run_sim(dump=True, n_out=1, show=True, save=False)