import Circuit_not
import Circuit
from Hodghux import Neuron_tf
import numpy as np
import scipy as sp
import params
import data

def inhibit():
    inhib = params.SYNAPSE_inhib
    connections = {(0,1) : inhib, (1,0) : inhib}
    t = np.array(sp.arange(0.0, 2000., params.DT))
    i0 = 10.*((t>300)&(t<350)) + 20.*((t>900)&(t<950))
    i1 = 10.*((t>500)&(t<550)) + 20.*((t>700)&(t<750)) + 6.*((t>1100)&(t<1300)) + 7.5*((t>1600)&(t<1800))
    i_out = 10.*((t>350)&(t<700))
    i_injs = np.array([i0, i1])
    neurons = [Neuron_tf(init_p=params.DEFAULT, fixed=params.ALL),
               Neuron_tf(init_p=params.DEFAULT, fixed=params.ALL)]
    c = Circuit_not.Circuit(neurons, connections, i_injs, t, i_out)
    c.run_sim()


def opt_neurons():
    connections = {(0, 1): params.SYNAPSE_inhib,
                   (1, 0): params.SYNAPSE}
    t = np.array(sp.arange(0.0, 1000., params.DT))
    i0 = 10. * ((t > 200) & (t < 400)) + 30. * ((t > 500) & (t < 600))
    i1 = 30. * ((t > 700) & (t < 800))
    i_injs = np.array([i0, i1])
    neurons = [Neuron_tf(),
               Neuron_tf()]
    c = Circuit_not.Circuit(neurons, connections, i_injs, t)
    c.opt_neurons()



if __name__ == '__main__':
    connections = {(0, 1) : params.SYNAPSE}
    t = np.array(sp.arange(0.0, 800., params.DT))
    i0 = 10. * ((t > 200) & (t < 400)) + 30. * ((t > 500) & (t < 600))
    i1 = np.zeros(t.shape)
    i_injs = np.array([i0, i1])
    neurons = [Neuron_tf(fixed=params.ALL),
               Neuron_tf(fixed=params.ALL)]
    c = Circuit_not.Circuit(neurons, connections, i_injs, t)
    c.run_sim(dump=True)

    c = Circuit.Circuit(neurons, connections, i_injs, t)
    c.opt_circuits('YOLO', data.DUMP_FILE+str(1))