import Circuit_not
import Circuit
import utils
import numpy as np
import scipy as sp
import params
import data

p = [params.DEFAULT, params.DEFAULT]

def inhibit():
    inhib = params.SYNAPSE_inhib
    connections = {(0,1) : inhib, (1,0) : inhib}
    t = np.array(sp.arange(0.0, 2000., params.DT))
    i0 = 10.*((t>300)&(t<350)) + 20.*((t>900)&(t<950))
    i1 = 10.*((t>500)&(t<550)) + 20.*((t>700)&(t<750)) + 6.*((t>1100)&(t<1300)) + 7.5*((t>1600)&(t<1800))
    i_injs = np.array([i0, i1]).transpose()
    c = Circuit_not.Circuit(p, connections, i_injs, t)
    c.run_sim()


def opt_neurons():
    connections = {(0, 1): params.SYNAPSE_inhib,
                   (1, 0): params.SYNAPSE}
    t = np.array(sp.arange(0.0, 1000., params.DT))
    i0 = 10. * ((t > 200) & (t < 400)) + 30. * ((t > 500) & (t < 600))
    i1 = 30. * ((t > 700) & (t < 800))
    i_injs = np.array([i0, i1]).transpose()
    c = Circuit_not.Circuit([p,p], connections, i_injs, t)
    f = c.run_sim(dump=True, general=False)
    c = Circuit.Circuit([p,p], connections)
    c.opt_neurons(f)



if __name__ == '__main__':
    p = params.DEFAULT
    pars = [p,p]
    n_out = 1
    dir = '2n-2exc'
    utils.set_dir(dir)

    connections = {(0, 1) : params.SYNAPSE,
                   (1,0) : params.SYNAPSE}
    t = np.array(sp.arange(0.0, 800., params.DT))
    i0 = 10. * ((t > 200) & (t < 400)) + 30. * ((t > 500) & (t < 600))
    i1 = np.zeros(t.shape)
    i_injs = np.array([i0, i1]).transpose()
    c = Circuit_not.Circuit(pars, connections, i_injs, t)
    connections = {(0, 1): params.get_syn_rand(),
                   (1,0): params.get_syn_rand()}
    c.run_sim(dump=True, show=True, save=True)
    c = Circuit.Circuit(pars, connections)
    c.opt_circuits(dir, n_out=n_out, file=data.DUMP_FILE+str(n_out))