import scipy as sp
import numpy as np
from scipy.stats import norm
from collections import OrderedDict
import random

#'[k|c]a?_[^_]*__(.*)': ['"](.*) .*["']

DECAY_CA = 110.
RHO_CA = 0.23
REST_CA = 0.
INIT_STATE = [-50., 0., 0.95, 0., 0., 1., 1.e-7]
INIT_STATE_ica = [0., 0, 1, 1, 0]

RES = {
        'Cm' : 21.79517,
        'C_m' : 26.057444,
        'E_Ca' : 14.850545,
        'E_K' : -40.62314,
        'E_L' : -25.78042,
        'e__mdp' : 4.935627,
        'e__scale' : 24.047474,
        'e__tau' : 31.966341,
        'f__mdp' : -61.386616,
        'f__scale' : -34.79839,
        'f__tau' : 58.10077,
        'g_Ca' : 5.9096923,
        'g_Kf' : -2.9872065,
        'g_Ks' : 2.9289305,
        'g_L' : -0.11360436,
        'h__alpha' : 4.8876266,
        'h__mdp' : 9.4649,
        'h__scale' : -44.16387,
        'n__mdp' : -17.819798,
        'n__scale' : 26.955765,
        'n__tau' : 66.007095,
        'p__mdp' : -17.149368,
        'p__scale' : 22.812742,
        'p__tau' : 78.51109,
        'q__mdp' : -23.938337,
        'q__scale' : -27.434593,
        'q__tau' : 124.33194
}

DEFAULT = {
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

MAX_TAU = 200.
MIN_SCALE = 0.5
MAX_SCALE = 50.
MIN_MDP = -65.
MAX_MDP = 20.
MAX_G = 10.

PARAMS_RAND = {
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
        'e__mdp': random.uniform(MIN_MDP, MAX_MDP),

        'h__alpha': random.uniform(0.1,0.9),
        'h__scale': random.uniform(-MAX_SCALE,-MIN_SCALE),
        'h__mdp': random.uniform(1,100),

        'C_m': random.uniform(0.1, 50.),
        # membrane capacitance, in uF/cm^2

        'g_Ca': random.uniform(0.1,MAX_G),
        # Calcium (Na) maximum conductances, in mS/cm^2

        'g_Ks': random.uniform(0.1,MAX_G),
        'g_Kf': random.uniform(0.1,MAX_G),
        # Postassium (K) maximum conductances, in mS/cm^2

        'g_L': random.uniform(0.1,MAX_G),
        # Leak maximum conductances, in mS/cm^2

        'E_Ca': random.uniform(0., MAX_MDP),
        # Sodium (Na) Nernst reversal potentials, in mV

        'E_K': random.uniform(MIN_MDP, 0.),
        # Postassium (K) Nernst reversal potentials, in mV

        'E_L': random.uniform(MIN_MDP, 0.),
        # Leak Nernst reversal potentials, in mV
}

params = DEFAULT
params = OrderedDict(sorted(params.items(), key=lambda t: t[0]))



DT = 0.1

# i_inj = 0.01*(t)*((t>1000)&(t<4000))
t_train = np.array(sp.arange(0.0, 1200., DT))
i_inj_train = 10.*((t_train>100)&(t_train<300)) + 20.*((t_train>400)&(t_train<600)) + 40.*((t_train>800)&(t_train<950))
i_inj_train = np.array(i_inj_train, dtype=np.float32)

t = np.array(sp.arange(0.0, 2000., DT))
i_inj = 10.*((t>100)&(t<750)) + 20.*((t>1500)&(t<2500)) + 40.*((t>3000)&(t<4000))
v_inj = 100*(t/2000) - np.full(t.shape,50)
i_inj = np.array(i_inj, dtype=np.float32)

