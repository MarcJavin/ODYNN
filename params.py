import scipy as sp
import numpy as np
from scipy.stats import norm

PARAM_GATES = {
        'p__tau': 2.25518, #ms
        'p__scale': 7.42636, #mV
        'p__mdp': -8.05232, #mV

        'q__tau': 149.963,
        'q__scale': -9.97468,
        'q__mdp': -15.6456,

        'n__tau': 25.0007,
        'n__scale': 15.8512,
        'n__mdp': 19.8741,

        'e__tau': 0.100027,
        'e__scale': 6.74821,
        'e__mdp': -3.3568,

        'f__tau': 150.88,
        'f__scale': -5.03176,
        'f__mdp': 25.1815,

        'e__tau': 0.100027,
        'e__scale': 6.74821,
        'e__mdp': -3.3568,

        'h__alpha' : 0.282473, #None
        'h__scale' : -1.00056, #mol per m3
        'h__mdp' : 6.41889
}

PARAM_MEMB = {

        'C_m' : 1.0,
        #membrane capacitance, in uF/cm^2
        
        'g_Ca' : 3.0,
        #Calcium (Na) maximum conductances, in mS/cm^2
        
        'g_Ks' : 3.0,
        'g_Kf' : 0.0712,
        #Postassium (K) maximum conductances, in mS/cm^2
        
        'g_L' : 0.005,
        #Leak maximum conductances, in mS/cm^2
        
        'E_Ca' : 40.0,
        #Sodium (Na) Nernst reversal potentials, in mV
        
        'E_K' : -60.0,
        #Postassium (K) Nernst reversal potentials, in mV
        
        'E_L' : -50.0
        #Leak Nernst reversal potentials, in mV
}


DT = 0.1
t = np.array(sp.arange(0.0, 5200., DT))
i_inj = {}
sigma = 500
mu = 5000
n = 0
i_inj = 0.01*(5200-t)*((t>100)&(t<5000))
# i_inj = 5*((t>000)&(t<500)) + 10*((t>1000)&(t<1500)) + 15*((t>2000)&(t<2500)) + 20*((t>3000)&(t<3500))\
#            + 70*norm(mu, sigma).pdf(t)*sigma
i_inj = np.array(i_inj, dtype=np.float32)