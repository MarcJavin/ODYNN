import scipy as sp
import numpy as np
from scipy.stats import norm
from collections import OrderedDict
import random

#'[k|c]a?_[^_]*__(.*)': ['"](.*) .*["']

DECAY_CA = 110.
RHO_CA = 0.23
REST_CA = 0.

PARAMS = {
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

MAX_TAU = 300.
MIN_SCALE = 0.5
MAX_SCALE = 100.
MIN_MDP = -65.
MAX_MDP = 40.
MAX_G = 12.

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

params = PARAMS_RAND

params = OrderedDict(sorted(params.items(), key=lambda t: t[0]))



DT = 0.1
t = np.array(sp.arange(0.0, 5000., DT))
i_inj = {}
sigma = 500
mu = 5000
n = 0
# i_inj = 0.01*(t)*((t>1000)&(t<4000))
i_inj = 10.*((t>000)&(t<1000)) + 20.*((t>1500)&(t<2500)) + 40.*((t>3000)&(t<4000))
        # + 20*((t>3000)&(t<3500))\
        #    + 70*norm(mu, sigma).pdf(t)*sigma
i_inj = np.array(i_inj, dtype=np.float32)