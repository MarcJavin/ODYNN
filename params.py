import scipy as sp
import numpy as np
import random

# '[k|c]a?_[^_]*__(.*)': ['"](.*) .*["']




SYNAPSE1 = {
    'G': 1.,
    'mdp': -30.,
    'scale': 2.,
    'E': 20.
}

SYNAPSE2 = {
    'G': 10.,
    'mdp': 0.,
    'scale': 5.,
    'E': -10.
}

SYNAPSE = {
    'G': 5.,
    'mdp': -25.,
    'scale': 2.,
    'E': 0.
}

SYNAPSE_inhib = {
    'G': 1.,
    'mdp': -35.,
    'scale': -2.,
    'E': 20.
}

def give_constraints_syn(conns):
    """constraints for synapse parameters"""
    scale_con = np.array([const_scale(True) if p['scale'] > 0 else const_scale(False) for p in conns.values()])
    return {'G': np.array([1e-5, np.infty]),
            'scale': scale_con.transpose()}


def const_scale(exc=True):
    if (exc):
        return [1e-3, np.infty]
    else:
        return [-np.infty, -1e-3]



MAX_TAU = 200.
MIN_SCALE = 1.
MAX_SCALE = 50.
MIN_MDP = -40.
MAX_MDP = 30.
MAX_G = 10.

def get_syn_rand(exc=True):
    """Random parameters for a synapse"""
    # scale is negative if inhibitory
    if (exc):
        scale = random.uniform(MIN_SCALE, MAX_SCALE)
    else:
        scale = random.uniform(-MAX_SCALE, -MIN_SCALE)
    return {
        'G': random.uniform(0.01, MAX_G),
        'mdp': random.uniform(MIN_MDP, MAX_MDP),
        'scale': scale,
        'E': random.uniform(-20., 50.),
    }





DT = 0.1
t_train = np.array(sp.arange(0.0, 1200., DT))
i_inj_train = 10. * ((t_train > 100) & (t_train < 300)) + 20. * ((t_train > 400) & (t_train < 600)) + 40. * (
            (t_train > 800) & (t_train < 950))
i_inj_train = np.array(i_inj_train, dtype=np.float32)
i_inj_train2 = 30. * ((t_train > 100) & (t_train < 500)) + 25. * ((t_train > 800) & (t_train < 900))
i_inj_train3 = np.sum([(10. + (n * 2 / 100)) * ((t_train > n) & (t_train < n + 50)) for n in range(100, 1100, 100)],
                      axis=0)
i_inj_trains = np.stack([i_inj_train, i_inj_train2, i_inj_train3], axis=1)

def give_train(dt=DT, nb_neuron_zero=None, max_t=1200.):
    """time and currents for optimization"""
    t = np.array(sp.arange(0.0, max_t, dt))
    i = 10. * ((t > 100) & (t < 300)) + 20. * ((t > 400) & (t < 600)) + 40. * ((t > 800) & (t < 950))
    i2 = 30. * ((t > 100) & (t < 500)) + 25. * ((t > 800) & (t < 900))
    i3 = np.sum([(10. + (n * 2 / 100)) * ((t > n) & (t < n + 50)) for n in range(100, 1100, 100)], axis=0)
    i4 = 15. * ((t > 400) & (t < 800))
    i_fin = np.stack([i, i2, i3, i4], axis=1)
    if(nb_neuron_zero is not None):
        i_zeros = np.zeros(i_fin.shape)
        i_fin= np.stack([i_fin, i_zeros], axis=2)
    return t, i_fin


def full4(dt=DT, nb_neuron_zero=None):
    t = np.array(sp.arange(0.0, 1200., dt))
    i1 = 10. * ((t > 200) & (t < 600))
    i2 = 10. * ((t > 300) & (t < 700))
    i3 = 10. * ((t > 400) & (t < 800))
    i4 = 10. * ((t > 500) & (t < 900))
    is_ = np.stack([i1, i2, i3, i4], axis=1)
    is_2 = is_ * 2
    i1 = np.sum([10. * ((t > n) & (t < n + 100)) for n in range(70, 1100, 200)], axis=0)
    i2 = np.sum([(10. + (n * 1 / 100)) * ((t > n) & (t < n + 50)) for n in range(100, 1100, 100)], axis=0)
    i3 = np.sum([(22. - (n * 1 / 100)) * ((t > n) & (t < n + 50)) for n in range(120, 1100, 100)], axis=0)
    i4 = np.sum([(10. + (n * 2 / 100)) * ((t > n) & (t < n + 20)) for n in range(100, 1100, 80)], axis=0)
    is_3 = np.stack([i1, i2, i3, i4], axis=1)
    i_fin = np.stack([is_, is_3], axis=1)
    if(nb_neuron_zero is not None):
        i_zeros = np.zeros((len(t), i_fin.shape[1], 6))
        i_fin = np.append(i_fin, i_zeros, axis=2)
    return t, i_fin




if __name__ == '__main__':
    t, i = full4()
    print(i.shape)
    i_1 = np.zeros((len(t), i.shape[1], 6))
    print(i_1.shape)
    i = np.append(i, i_1, axis=2)
    print(i.shape)

t_len = 5000.
t = np.array(sp.arange(0.0, t_len, DT))
i_inj = 10. * ((t > 100) & (t < 750)) + 20. * ((t > 1500) & (t < 2500)) + 40. * ((t > 3000) & (t < 4000))
v_inj = 115 * (t / t_len) - np.full(t.shape, 65)
v_inj_rev = np.full(t.shape, 50) - v_inj
i_inj = np.array(i_inj, dtype=np.float32)

t_test = np.array(sp.arange(0.0, 2000, DT))
i_test = 10. * ((t_test > 100) & (t_test < 300)) + 20. * ((t_test > 400) & (t_test < 600)) + 40. * (
            (t_test > 800) & (t_test < 950)) + \
         (t_test - 1200) * (50. / 500) * ((t_test > 1200) & (t_test < 1700))
