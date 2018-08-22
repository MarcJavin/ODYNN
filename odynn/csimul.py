"""
.. module:: csimul
    :synopsis: Module for simulation of neural circuits

.. moduleauthor:: Marc Javin
"""

import numpy as np
from .circuit import Circuit
from .neuron import PyBioNeuron
import time


def simul(t, i_injs, pars=None, synapses={}, gaps={}, circuit=None, n_out=[0], suffix='', show=False,
          save=True, labels=None):
    """
    Simulate a circuit with input current `i_injs` and return the outputs of neurons contained in `n_out`

    Args:
        t(ndarray): time
        i_injs(ndarray): input currents of shape [time, batch, neuron]
        pars(dict or list): parameters for the neurons
        synapses(dict or list): parameters of the chemical synapses
        gaps(dict or list): parameters of the gap junctions
        circuit(:obj: Circuit): if not None, ignore the 3 previous arguments
        n_out(list): neurons which output have to be saved
        suffix(str): suffix for the plots
        show(bool): If True, show the plot
        save(bool): If True, save the plot
        labels: labels for the circuit's neurons

    Returns:
        list : measurements as a list [time, input currents, [voltage(, calcium)]]
    """
    if circuit is None:
        circuit = Circuit(PyBioNeuron(pars, dt=t[1] - t[0]), synapses=synapses, gaps=gaps, labels=labels)

    #[(batch,) time, state, neuron]
    print('Circuit Simulation'.center(40, '_'))
    start = time.time()
    curs = None
    try:
        states, curs = circuit.calculate(i_injs)
    except:
        states = circuit.calculate(i_injs)
    print('Simulation time : {}'.format(time.time() - start))

    if states.ndim > 3:
        for i in range(i_injs.shape[1]):
            if curs is not None:
                c = curs[:, i]
            else:
                c = curs
            circuit.plots_output_mult(t, i_injs[:,i], states[:,:,i], i_syn=c, show=show, save=save,
                                      suffix='TARGET_%s%s'%(suffix,i))
        # [t, state, (batch,) neuron]
    else:
        circuit.plots_output_mult(t, i_injs, states, i_syn=curs, show=show, save=save, suffix='TARGET_%s'%suffix)
        #reshape for batch dimension
        states = states[:,:,np.newaxis,:]
        i_injs = i_injs[:,np.newaxis,:]

    V = np.moveaxis(states[:, 0, :, np.array(n_out)], 0, -1)
    Ca = np.moveaxis(states[:, -1, :, np.array(n_out)], 0, -1)
    return [t, i_injs, [V, Ca]]

