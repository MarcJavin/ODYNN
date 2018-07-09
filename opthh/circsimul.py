"""
.. module:: circsimul
    :synopsis: Module for simulation of neural circuits

.. moduleauthor:: Marc Javin
"""

import numpy as np
from .circuit import CircuitFix
from .utils import plots_output_mult
from .datas import DUMP_FILE
import pickle
import time



# def __init__(self, inits_p, conns, t, i_injs, dt=0.1):
#     assert(dt == t[1] - t[0])
#     circuit = CircuitFix(inits_p=inits_p, conns=conns, dt=dt)
#     self.batch = False
#     if i_injs.ndim > 2:
#         self.batch = True
#         self.n_batch = i_injs.shape[1]
#         i_injs = np.moveaxis(i_injs, 1, 0)
#         self.calculate = np.vectorize(self.calculate, signature='(t,n)->(t,s,n),(t,n)')
#     print(len(inits_p), i_injs.shape)
#     assert (len(inits_p) == i_injs.shape[-1])
#     self.connections = conns
#     t = t
#     #[(batch,) t, neuron]
#     i_injs = i_injs

def simul(pars, conns, t, i_injs, circuit=None, n_out=[0], dump=False, suffix='', show=False, save=True):
    """runs the entire simulation

    Args:
      n_out(list): neurons to register
      dump(bool): If True, dump the measurement (Default value = False)
      suffix(str): suffix for the figure files (Default value = '')
      show(bool): If True, show the figure (Default value = False)
      save(bool): If True, save the figure (Default value = True)

    Returns:
        str or list: If dump, return the file name,
        otherwise, return the measurements as a list [time, input, voltage, calcium]
    """
    if circuit is None:
        circuit = CircuitFix(pars, conns, dt=t[1]-t[0])

    #[(batch,) time, state, neuron]
    print('Circuit Simulation'.center(40, '_'))
    start = time.time()
    states, curs = circuit.calculate(i_injs)
    print('Simulation time : {}'.format(time.time() - start))

    if states.ndim > 3:
        for i in range(i_injs.shape[1]):
            plots_output_mult(t, i_injs[:,i], states[:,circuit.neurons.V_pos,i,:], states[:,circuit.neurons.Ca_pos,i,:],
                      i_syn=curs[:,i], show=show, save=save, suffix='TARGET_%s%s'%(suffix,i))
        # [t, state, (batch,) neuron]
    else:
        plots_output_mult(t, i_injs, states[:,circuit.neurons.V_pos,:], states[:,circuit.neurons.Ca_pos,:],
                      i_syn=curs, show=show, save=save, suffix='TARGET_%s'%suffix)
        #reshape for batch dimension
        states = states[:,:,np.newaxis,:]
        i_injs = i_injs[:,np.newaxis,:]

    V = np.stack([states[:, 0, :, n] for n in n_out], axis=-1)
    Ca = np.stack([states[:, -1, :, n] for n in n_out], axis=-1)
    todump = [t, i_injs, V, Ca]
    if dump:
        with open(DUMP_FILE, 'wb') as f:
            pickle.dump(todump, f)
        return DUMP_FILE
    else:
        return todump


