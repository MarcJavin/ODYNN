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



def simul(t, i_injs, pars=None, synapses={}, gaps={}, circuit=None, n_out=[0], dump=False, suffix='', show=False,
          save=True, labels=None):
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
        circuit = CircuitFix(pars, dt=t[1] - t[0], synapses=synapses, gaps=gaps, labels=labels)

    #[(batch,) time, state, neuron]
    print('Circuit Simulation'.center(40, '_'))
    start = time.time()
    states, curs = circuit.calculate(i_injs)
    print('Simulation time : {}'.format(time.time() - start))

    if states.ndim > 3:
        for i in range(i_injs.shape[1]):
            circuit.plots_output_mult(t, i_injs[:,i], states[:,circuit.neurons.V_pos,i,:], states[:,circuit.neurons.Ca_pos,i,:],
                      i_syn=curs[:,i], show=show, save=save, suffix='TARGET_%s%s'%(suffix,i))
        # [t, state, (batch,) neuron]
    else:
        circuit.plots_output_mult(t, i_injs, states[:,circuit.neurons.V_pos,:], states[:,circuit.neurons.Ca_pos,:],
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


