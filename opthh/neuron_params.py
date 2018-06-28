import numpy as np
import random


REST_CA = 0.
"""init states for neurons : current, voltage, rates and [Ca2+]"""
INITS = {
    'i': 0,
    'V': -50.,
    'p': 0.,
    'q': 0.95,
    'n': 0.,
    'e': 0.,
    'f': 1.,
    'h': 0.,
    'cac': 1.e-7
}
INIT_STATE = np.array([INITS[p] for p in ['V', 'p', 'q', 'n', 'e', 'f', 'cac']])
INIT_STATE_ica = [INITS[p] for p in ['i', 'e', 'f', 'h', 'cac']]
INIT_STATE_ik = [INITS[p] for p in ['i', 'p', 'q', 'n']]


"""constraints for neuron parameters"""
CONSTRAINTS = {
    'decay_ca': [1e-3, np.infty],
    'rho_ca': [1e-3, np.infty],
    'C_m': [5e-1, np.infty],
    'e__scale': [1e-3, np.infty],
    'e__tau': [1e-3, np.infty],
    'f__scale': [-np.infty, 1e-3],
    'f__tau': [1e-3, np.infty],
    'g_Ca': [1e-5, np.infty],
    'g_Kf': [1e-5, np.infty],
    'g_Ks': [1e-5, np.infty],
    'g_L': [1e-5, np.infty],
    'h__alpha': [0, 1],
    'h__scale': [-np.infty, 1e-3],
    'n__scale': [1e-3, np.infty],
    'n__tau': [1e-3, np.infty],
    'p__scale': [1e-3, np.infty],
    'p__tau': [1e-3, np.infty],
    'q__scale': [-np.infty, 1e-3],
    'q__tau': [1e-3, np.infty]
}

DEFAULT_2 = {
    'decay_ca': 110.,
    'rho_ca': 0.23,
    'p__tau': 100.,  # ms
    'p__scale': 7.43,  # mV
    'p__mdp': -8.05,  # mV
    'q__tau': 100.,
    'q__scale': -9.97,
    'q__mdp': -15.6,
    'n__tau': 1050.,
    'n__scale': 20.,
    'n__mdp': 2.,
    'f__tau': 301.,
    'f__scale': -20.03,
    'f__mdp': 5.2,
    'e__tau': 20.,
    'e__scale': 15.,
    'e__mdp': -6.,
    'h__alpha': 0.282,  # None
    'h__scale': -1.,  # mol per m3
    'h__mdp': 302.,
    'C_m': 20.0,
    'g_Ca': 3.,
    'g_Ks': 6.0,
    'g_Kf': 0.07,
    'g_L': 0.005,
    'E_Ca': 20.0,
    'E_K': -60.0,
    'E_L': -60.0
}
DEFAULT = {
    'decay_ca': 110.,
    'rho_ca': 0.23,

    'p__tau': 100.,  # ms
    'p__scale': 7.43,  # mV
    'p__mdp': -8.05,  # mV

    'q__tau': 150.,
    'q__scale': -9.97,
    'q__mdp': -15.6,

    'n__tau': 25.,
    'n__scale': 15.9,
    'n__mdp': 19.9,

    'f__tau': 151.,
    'f__scale': -5.03,
    'f__mdp': 25.2,

    'e__tau': 10.,
    'e__scale': 6.75,
    'e__mdp': -3.36,

    'h__alpha': 0.282,  # None
    'h__scale': -1.,  # mol per m3
    'h__mdp': 6.42,

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

def give_rand():
    """Random parameters for a neuron"""
    return {
        'decay_ca': 110.,
        'rho_ca': 0.23e-2,
        'p__tau': random.uniform(0.1, MAX_TAU),
        'p__scale': random.uniform(MIN_SCALE, MAX_SCALE),
        'p__mdp': random.uniform(MIN_MDP, MAX_MDP),

        'q__tau': random.uniform(0.1, MAX_TAU),
        'q__scale': random.uniform(-MAX_SCALE, -MIN_SCALE),
        'q__mdp': random.uniform(MIN_MDP, MAX_MDP),

        'n__tau': random.uniform(MAX_TAU, 5*MAX_TAU),
        'n__scale': random.uniform(MIN_SCALE, MAX_SCALE),
        'n__mdp': random.uniform(MIN_MDP, MAX_MDP),

        'f__tau': random.uniform(0.1, MAX_TAU),
        'f__scale': random.uniform(-MAX_SCALE, -MIN_SCALE),
        'f__mdp': random.uniform(MIN_MDP, MAX_MDP),

        'e__tau': random.uniform(0.1, MAX_TAU),
        'e__scale': random.uniform(MIN_SCALE, MAX_SCALE),
        'e__mdp': random.uniform(-30., 0.),

        'h__alpha': random.uniform(0.1, 0.9),
        'h__scale': random.uniform(-MAX_SCALE, -MIN_SCALE),
        'h__mdp': random.uniform(1, 100),

        'C_m': random.uniform(0.1, 40.),
        'g_Ca': random.uniform(0.1, MAX_G),
        'g_Ks': random.uniform(0.1, MAX_G),
        'g_Kf': random.uniform(0.1, MAX_G),
        'g_L': random.uniform(0.001, 0.5),
        'E_Ca': random.uniform(0., 40),
        'E_K': random.uniform(-80, -40.),
        'E_L': random.uniform(-80, -40.),
    }
