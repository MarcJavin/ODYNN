import scipy as sp
import numpy as np
from scipy.stats import norm
from collections import OrderedDict

#'[k|c]a?_[^_]*__(.*)': ['"](.*) .*["']

PARAMS = {
        'p__tau': 100, #ms
        'p__scale': 7.42636, #mV
        'p__mdp': -8.05232, #mV

        'q__tau': 149.963,
        'q__scale': -9.97468,
        'q__mdp': -15.6456,

        'n__tau': 25.0007,
        'n__scale': 15.8512,
        'n__mdp': 19.8741,

        'f__tau': 150.88,
        'f__scale': -5.03176,
        'f__mdp': 25.1815,

        'e__tau': 10,
        'e__scale': 6.74821,
        'e__mdp': -3.3568,

        'h__alpha' : 0.282473, #None
        'h__scale' : -1.00056, #mol per m3
        'h__mdp' : 6.41889,


        'C_m' : 20.0,
        #membrane capacitance, in uF/cm^2
        
        'g_Ca' : 3.0,
        #Calcium (Na) maximum conductances, in mS/cm^2
        
        'g_Ks' : 10.0,
        'g_Kf' : 0.07,
        #Postassium (K) maximum conductances, in mS/cm^2
        
        'g_L' : 0.005,
        #Leak maximum conductances, in mS/cm^2
        
        'E_Ca' : 20.0,
        #Sodium (Na) Nernst reversal potentials, in mV
        
        'E_K' : -60.0,
        #Postassium (K) Nernst reversal potentials, in mV
        
        'E_L' : -60.0
        #Leak Nernst reversal potentials, in mV
}

PARAM_GATES2 = {
        'p__tau' : 3,
        'p__scale' : 5,
        'p__mdp' : 7.4,

        'q__tau' : 50,
        'q__scale' : -200,
        'q__mdp' : -0.65,

        'n__tau' : 4000,
        'n__scale' : 16,
        'n__mdp' : 15,

        'e__tau' : 70,
        'e__scale' : 10.7,
        'e__mdp' : 0,

        'f__tau' : 1000,
        'f__scale' : -5,
        'f__mdp' : 25,

        'h__alpha': 0.1,
        'h__scale' : -1e-8,
        'h__mdp' : 6.4e-8,
        
}

PARAM_MEMB2 = {

        "g_L": 0.1,
        "g_Kf": 3,
        "g_Ks": 0.07,
        "g_Ca": 3,

        'E_Ca' : 40.0,
        'E_K': -60,
        'E_L': -50
    }

params = PARAMS

params = OrderedDict(sorted(params.items(), key=lambda t: t[0]))



DT = 0.1
t = np.array(sp.arange(0.0, 5200., DT))
i_inj = {}
sigma = 500
mu = 5000
n = 0
# i_inj = 0.01*(t)*((t>1000)&(t<4000))
i_inj = 5*((t>000)&(t<500)) + 10*((t>1000)&(t<1500)) + 15*((t>2000)&(t<2500)) + 20*((t>3000)&(t<3500))\
           + 70*norm(mu, sigma).pdf(t)*sigma
i_inj = np.array(i_inj*2, dtype=np.float32)