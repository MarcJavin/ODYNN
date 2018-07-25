"""
.. module:: neursimul
    :synopsis: Module for simulation of neurons

.. moduleauthor:: Marc Javin
"""

from .neuron import BioNeuronFix
import time
from . import datas
import numpy as np
import pickle
import scipy as sp

DT = 0.1
t_len = 5000.
t = np.array(sp.arange(0.0, t_len, DT))
i_inj = 10. * ((t > 100) & (t < 750)) + 20. * ((t > 1500) & (t < 2500)) + 40. * ((t > 3000) & (t < 4000))


def comp_pars(ps, t=None, dt=DT, i_inj=i_inj, show=True, save=False):
    """Compare different parameter sets on the same experiment

    Args:
      ps(list of dict): list of parameters to compare
      dt(float): time step
      i_inj(ndarray): input currents
      show(bool): If True, show the figure (Default value = True)
      save(bool): If True, save the figure (Default value = False)

    """
    start = time.time()
    if t is not None:
        dt = t[1] - t[0]
    else:
        t = sp.arange(0, len(i_inj) * dt, dt)
    neurons = BioNeuronFix(init_p=ps, dt=dt)
    X = neurons.calculate(i_inj)
    print("Simulation time : ", time.time() - start)
    neurons.plot_output(t, i_inj, X, show=show, save=save)

def comp_pars_targ(p, p_targ, t=None, dt=DT, i_inj=i_inj, suffix='', save=False, show=True):
    """Compare parameter sets with a target

    Args:
      p(dict or list of dict): parameter(s) to compare with the target
      p_targ(dict): target parameters
      dt(float): time step
      i_inj(ndarray): input currents
      suffix(str): suffix for the saved figure (Default value = '')
      save(bool): If True, save the figure (Default value = False)
      show(bool): If True, show the figure (Default value = True)

    """
    if(isinstance(p, list)):
        p.append(p_targ)
    else:
        p = [p, p_targ]
    if t is not None:
        dt = t[1] - t[0]
    else:
        t = sp.arange(0, len(i_inj) * dt, dt)

    start = time.time()
    neurons = BioNeuronFix(init_p=p, dt=dt)
    X = neurons.calculate(i_inj)
    print("Simulation time : ", time.time() - start)
    neurons.plot_output(t, i_inj, X[:, :, :-1], np.moveaxis(X[:, :, -1],1,0), suffix=suffix, save=save, show=show, l=2, lt=2, targstyle='-.')

def comp_neurons(neurons, i_inj=i_inj, show=True, save=False):
    """Compare different neurons on the same experiment

    Args:
      neurons(list of object NeuronModel): neurons to compare
      dt(float): time step
      i_inj(ndarray): input currents
      show(bool): If True, show the figure (Default value = True)
      save(bool): If True, save the figure (Default value = False)

    """
    start = time.time()
    X = []
    for n in neurons:
        X.append(n.calculate(i_inj))
    X = np.stack(X, axis=2)
    print("Simulation time : ", time.time() - start)
    t = sp.arange(0, len(i_inj)*neurons.dt, neurons.dt)
    neurons[0].plot_output(t, i_inj, X, show=show, save=save)

def comp_neuron_trace(neuron, trace, i_inj=i_inj, scale=False, show=True, save=False):
    """Compare a neuron with a given measured trace after scaling

    Args:
      neuron(NeuronModel object): neuron to compare
      trace: recordings to plot
      dt(float): time step
      i_inj(ndarray): input currents
      scale:  (Default value = False)
      show(bool): If True, show the figure (Default value = True)
      save(bool): If True, save the figure (Default value = False)

    """
    start = time.time()
    X = neuron.calculate(i_inj)
    print("Simulation time : ", time.time() - start)
    if scale:
        for i, t in enumerate(trace):
            if t is not None:
                t *= np.max(X[:,i]) / np.max(t)
    ts = sp.arange(0, len(i_inj)*neuron.dt, neuron.dt)
    neuron.plot_output(ts, i_inj, X, trace, show=show, save=save)


"""Runs and plot the neuron"""
def simul(p=None, neuron=None, t=None, dt=DT, i_inj=i_inj, dump=False, suffix='', show=False, save=True, ca_true=None):
    """Main demo for the Hodgkin Huxley neuron model

    Args:
        p(dict): parameters of the neuron to simulate
        neuron(NeuronModel object): neuron to simulate
        dt(float): time step
      i_inj(ndarray): input currents
      dump:  (Default value = False)
      suffix:  (Default value = '')
      show(bool): If True, show the figure (Default value = False)
      save:  (Default value = True)
      ca_true:  (Default value = None)

    Returns:
        ndarray: records if dump is False
        str: If dump is True, name of the file where the records have been dumped


    """
    if t is not None:
        dt = t[1] - t[0]
    else:
        t = sp.arange(0, len(i_inj) * dt, dt)
    if(neuron is None):
        neuron = BioNeuronFix(p, dt=dt)
    print('Neuron Simulation'.center(40,'_'))
    start = time.time()

    #[t,s,batch]
    X = neuron.calculate(i_inj)

    print("Simulation time : {}".format(time.time() - start))

    # if neuron.loop_func == neuron.ica_from_v:
    #     plots_ica_from_v(t, i_inj, np.array(X), suffix='target_%s' % suffix, show=show, save=save)
    # elif neuron.loop_func == neuron.ik_from_v:
    #     plots_ik_from_v(t, i_inj, np.array(X), suffix='target_%s' % suffix, show=show, save=save)
    # else:
    if True:
        if i_inj.ndim > 1:
            for i in range(i_inj.shape[1]):
                neuron.plot_results(t, i_inj[:,i], np.array(X[:,:,i]), suffix='target_%s%s' % (suffix,i), show=show,
                              save=save)
        else:
            neuron.plot_results(t, i_inj, np.array(X), suffix='target_%s' % suffix, show=show, save=save, ca_true=ca_true)

    todump = [t, i_inj, [X[:, neuron.V_pos], X[:, neuron.Ca_pos]]]
    if dump:
        with open(datas.DUMP_FILE, 'wb') as f:
            pickle.dump(todump, f)
        return datas.DUMP_FILE
    else:
        return todump


if __name__ == "__main__":
    import doctest
    doctest.testmod()