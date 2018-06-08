import scipy as sp
import numpy as np
from scipy.stats import norm
from collections import OrderedDict
import random

#'[k|c]a?_[^_]*__(.*)': ['"](.*) .*["']


REST_CA = 0.
"""init states for neurons : current, voltage, rates and [Ca2+]"""
INITS = {
        'i' : 0,
        'V' : -50.,
        'p' : 0.,
        'q' : 0.95,
        'n' : 0.,
        'e' : 0.,
        'f' : 1.,
        'h' : 0.,
        'cac' : 1.e-7
}
INIT_STATE = [INITS[p] for p in ['V', 'p', 'q', 'n', 'e', 'f', 'cac']]
INIT_STATE_ica = [INITS[p] for p in ['i', 'e', 'f', 'h', 'cac']]
INIT_STATE_ik = [INITS[p] for p in ['i', 'p', 'q', 'n']]


SYNAPSE = {
        'G' : 1.,
        'mdp' : -35.,
        'scale' : 2.,
        'E' : 20.
}

SYNAPSE_inhib = {
        'G' : 1.,
        'mdp' : -35.,
        'scale' : -2.,
        'E' : 20.
}

CONSTRAINTS_syn = {
        'G' : [1e-5,np.infty]
}

"""constraints for neuron parameters"""
CONSTRAINTS = {
        'decay_ca' : [1e-3,np.infty],
        'rho_ca' : [1e-3,np.infty],
        'C_m' : [5e-1,np.infty],
        'e__scale' : [1e-3,np.infty],
        'e__tau' : [1e-3,np.infty],
        'f__scale' : [-np.infty, 1e-3],
        'f__tau' : [1e-3,np.infty],
        'g_Ca' : [1e-5,np.infty],
        'g_Kf' : [1e-5,np.infty],
        'g_Ks' : [1e-5,np.infty],
        'g_L' : [1e-5,np.infty],
        'h__alpha' : [0,1],
        'h__scale' : [-np.infty, 1e-3],
        'n__scale' : [1e-3,np.infty],
        'n__tau' : [1e-3,np.infty],
        'p__scale' : [1e-3,np.infty],
        'p__tau' : [1e-3,np.infty],
        'q__scale' : [-np.infty, 1e-3],
        'q__tau' : [1e-3,np.infty]
}

DEFAULT = {
        'decay_ca' : 110.,
        'rho_ca' : 0.23,

        'p__tau': 100.,  # ms
        'p__scale': 7.42636,  # mV
        'p__mdp': -8.05232,  # mV

        'q__tau': 149.963,
        'q__scale': -9.97468,
        'q__mdp': -15.6456,

        'n__tau': 25.0007,
        'n__scale': 15.8512,
        'n__mdp': 19.8741,

        'f__tau': 150.88,
        'f__scale': -5.03176,
        'f__mdp': 25.1815,

        'e__tau': 10.,
        'e__scale': 6.74821,
        'e__mdp': -3.3568,

        'h__alpha': 0.282473,  # None
        'h__scale': -1.00056,  # mol per m3
        'h__mdp': 6.41889,

        'C_m': 20.0,
        # membrane capacitance, in uF/cm^2

        'g_Ca': 3.0,
        # Calcium (Na) maximum conductances, in mS/cm^2

        'g_Ks': 10.0,
        'g_Kf': 0.07,
        # Postassium (K) maximum conductances, in mS/cm^2

        'g_L': 0.005,
        # Leak maximum conductances, in mS/cm^2

        'E_Ca': 20.0,
        # Sodium (Na) Nernst reversal potentials, in mV

        'E_K': -60.0,
        # Postassium (K) Nernst reversal potentials, in mV

        'E_L': -60.0
        # Leak Nernst reversal potentials, in mV
}
ALL = set(DEFAULT.keys())

MAX_TAU = 200.
MIN_SCALE = 1.
MAX_SCALE = 50.
MIN_MDP = -40.
MAX_MDP = 30.
MAX_G = 10.

"""Random parameters for a synapse"""
def get_syn_rand(exc=True):
        #scale is negative if inhibitory
        if(exc):
                scale = random.uniform(MIN_SCALE, MAX_SCALE)
        else:
                scale = random.uniform(-MAX_SCALE,-MIN_SCALE)
        return {
                'G': random.uniform(0.01, MAX_G),
                'mdp': random.uniform(MIN_MDP, MAX_MDP),
                'scale': scale,
                'E': random.uniform(-20., 50.),
        }

"""Random parameters for a neuron"""
def give_rand():

        return {
                'decay_ca': 110.,
                'rho_ca': 0.23e-2,
                'p__tau': random.uniform(0.1,MAX_TAU),
                'p__scale':random.uniform(MIN_SCALE,MAX_SCALE),
                'p__mdp': random.uniform(MIN_MDP, MAX_MDP),

                'q__tau': random.uniform(0.1,MAX_TAU),
                'q__scale': random.uniform(-MAX_SCALE,-MIN_SCALE),
                'q__mdp': random.uniform(MIN_MDP, MAX_MDP),

                'n__tau': random.uniform(0.1,MAX_TAU),
                'n__scale': random.uniform(MIN_SCALE,MAX_SCALE),
                'n__mdp': random.uniform(MIN_MDP, MAX_MDP),

                'f__tau': random.uniform(0.1,MAX_TAU),
                'f__scale': random.uniform(-MAX_SCALE,-MIN_SCALE),
                'f__mdp': random.uniform(MIN_MDP, MAX_MDP),

                'e__tau': random.uniform(0.1,MAX_TAU),
                'e__scale': random.uniform(MIN_SCALE,MAX_SCALE),
                'e__mdp': random.uniform(-30., 0.),

                'h__alpha': random.uniform(0.1,0.9),
                'h__scale': random.uniform(-MAX_SCALE,-MIN_SCALE),
                'h__mdp': random.uniform(1,100),

                'C_m': random.uniform(0.1, 40.),
                'g_Ca': random.uniform(0.1,MAX_G),
                'g_Ks': random.uniform(0.1,MAX_G),
                'g_Kf': random.uniform(0.1,MAX_G),
                'g_L': random.uniform(0.001, 0.5),
                'E_Ca': random.uniform(0., 40),
                'E_K': random.uniform(-80, -40.),
                'E_L': random.uniform(-80, -40.),
        }

params = DEFAULT
params = OrderedDict(sorted(params.items(), key=lambda t: t[0]))



DT = 0.1
t_train = np.array(sp.arange(0.0, 1200., DT))
i_inj_train = 10.*((t_train>100)&(t_train<300)) + 20.*((t_train>400)&(t_train<600)) + 40.*((t_train>800)&(t_train<950))
i_inj_train = np.array(i_inj_train, dtype=np.float32)

def give_train(dt=DT):
        t_train = np.array(sp.arange(0.0, 1200., dt))
        i_inj_train = 10.*((t_train>100)&(t_train<300)) + 20.*((t_train>400)&(t_train<600)) + 40.*((t_train>800)&(t_train<950))
        i_inj_train = np.array(i_inj_train, dtype=np.float32)
        return t_train, i_inj_train

t_len = 5000.
t = np.array(sp.arange(0.0, t_len, DT))
i_inj = 10.*((t>100)&(t<750)) + 20.*((t>1500)&(t<2500)) + 40.*((t>3000)&(t<4000))
v_inj = 115*(t/t_len) - np.full(t.shape,65)
v_inj_rev = np.full(t.shape, 50) - v_inj
i_inj = np.array(i_inj, dtype=np.float32)


t_test = np.array(sp.arange(0.0, 2000, DT))
i_test = 10.*((t_test>100)&(t_test<300)) + 20.*((t_test>400)&(t_test<600)) + 40.*((t_test>800)&(t_test<950)) + \
         (t_test-1200)*(50./500)*((t_test>1200)&(t_test<1700))



