import scipy as sp
from scipy.stats import norm
from scipy.integrate import odeint
import time
from utils import plots_results, get_data
from params import PARAM_GATES, PARAM_MEMB
import params
import numpy as np

T, X, Y = get_data()
DT = T[1] - T[0]

class HodgkinHuxley():
    """Full Hodgkin-Huxley Model implemented in Python"""

    C_m  =   1.0
    """membrane capacitance, in uF/cm^2"""

    DECAY_CA = 11.6  # ms
    RHO_CA = 0.000239e3  # mol_per_cm_per_uA_per_ms
    REST_CA = 0  # M
    
    param_gates = PARAM_GATES
    memb = PARAM_MEMB

    dt = 0.1
    t = params.t
    i_inj = params.i_inj
    """ The time to integrate over """

    def inf(self, V, rate, p=param_gates):
        mdp = p['%s__mdp'%rate]
        scale = p['%s__scale'%rate]
        return 1 / (1 + sp.exp((mdp - V)/scale))

    def h(self, cac, p=param_gates):
        """Channel gating kinetics. Functions of membrane voltage"""
        q = self.inf(cac, 'h')
        return 1 + (q-1)*p['h__alpha']

    h_notensor = h

    def I_Ca(self, V, e, f, h):
        """
        Membrane current (in uA/cm^2)
        Sodium (Na = element name)
        """
        return self.memb['g_Ca'] * e ** 2 * f * h * (V - self.memb['E_Ca'])

    def I_Kf(self, V, p, q):
        """
        Membrane current (in uA/cm^2)
        Potassium (K = element name)
        """
        return self.memb['g_Kf'] * p ** 4 * q * (V - self.memb['E_K'])

    def I_Ks(self, V, n):
        """
        Membrane current (in uA/cm^2)
        Potassium (K = element name)
        """
        return self.memb['g_Ks'] * n * (V - self.memb['E_K'])

    #  Leak
    def I_L(self, V):
        """
        Membrane current (in uA/cm^2)
        Leak
        """
        return self.memb['g_L'] * (V - self.memb['E_L'])

    def I_inj(self, t):
        """
        External Current
        """
        sigma = 250
        mu = 5500
        n = 0
        if(t>5000):
            n = norm(mu, sigma).pdf(t)*sigma*40
        return 5*((t>000) and (t<500)) + 10*((t>1000) and (t<1500)) + 15*((t>2000) and (t<2500)) + 20*((t>3000) and (t<3500))\
               + 25*((t>4000) and (t<4500)) + n

    @staticmethod
    def dALLdt(X, t, self):
        """
        Integrate

        |  :param X:
        |  :param t:
        |  :return: calculate membrane potential & activation variables
        """
        V, p, q, n, e, f, cac = X

        h = self.h(cac)
        dVdt = (self.I_inj(t) - self.I_Ca(V, e, f, h) - self.I_Ks(V, n) - self.I_Kf(V, p, q) - self.I_L(V)) / self.C_m
        dpdt = (self.inf(V, 'p') - p) / self.param_gates['p__tau']
        dqdt = (self.inf(V, 'q') - q) / self.param_gates['q__tau']
        dedt = (self.inf(V, 'e') - e) / self.param_gates['e__tau']
        dfdt = (self.inf(V, 'f') - f) / self.param_gates['f__tau']
        dndt = (self.inf(V, 'n') - n) / self.param_gates['n__tau']
        dcacdt = - self.I_Ca(V, e, f, h) * self.RHO_CA - ((cac - self.REST_CA) / self.DECAY_CA)
        return dVdt, dpdt, dqdt, dndt, dedt, dfdt, dcacdt
    
    
    def integ_comp(self, X, i_inj, dt):
        """
        Integrate
        """
        V = X[0]
        p = X[1]
        q = X[2]
        n = X[3]
        e = X[4]
        f = X[5]
        cac = X[6]
        h = self.h(cac)
        V += ((i_inj - self.I_Ca(V, e, f, h) - self.I_Ks(V, n) - self.I_Kf(V, p, q) - self.I_L(V)) / self.C_m) * dt
        cac += (-self.I_Ca(V, e, f, h) * self.RHO_CA - ((cac - self.REST_CA) / self.DECAY_CA)) * dt
        cac = (self.DECAY_CA/(dt+self.DECAY_CA)) * (cac - self.I_Ca(V, e, f, h)*self.RHO_CA*dt + self.REST_CA*self.DECAY_CA/dt)
        tau = self.param_gates['p__tau']
        p = ((tau*dt) / (tau+dt)) * ((p/dt) + (self.inf(V, 'p')/tau))
        tau = self.param_gates['q__tau']
        q = ((tau*dt) / (tau+dt)) * ((q/dt) + (self.inf(V, 'q')/tau))
        tau = self.param_gates['e__tau']
        e = ((tau*dt) / (tau+dt)) * ((e/dt) + (self.inf(V, 'e')/tau))
        tau = self.param_gates['f__tau']
        f = ((tau*dt) / (tau+dt)) * ((f/dt) + (self.inf(V, 'f')/tau))
        tau = self.param_gates['n__tau']
        n = ((tau*dt) / (tau+dt)) * ((n/dt) + (self.inf(V, 'n')/tau))
        return [V, p, q, n, e, f, cac]

    def no_tau(self, X, i_inj, dt):
        """
        Integrate
        """
        V = X[0]
        p = X[1]
        q = X[2]
        n = X[3]
        e = X[4]
        f = X[5]
        cac = X[6]
        h = self.h(cac)
        V += ((i_inj - self.I_Ca(V, e, f, h) - self.I_Ks(V, n) - self.I_Kf(V, p, q) - self.I_L(
            V)) / self.C_m) * dt
        cac += (-self.I_Ca(V, e, f, h) * self.RHO_CA - ((cac - self.REST_CA) / self.DECAY_CA)) * dt
        cac = (self.DECAY_CA / (dt + self.DECAY_CA)) * (
                    cac - self.I_Ca(V, e, f, h) * self.RHO_CA * dt + self.REST_CA * self.DECAY_CA / dt)
        p = self.inf(V, 'p')
        q = self.inf(V, 'q')
        e = self.inf(V, 'e')
        f = self.inf(V, 'f')
        n = self.inf(V, 'n')
        return [V, p, q, n, e, f, cac]





    def Main(self):
        """
        Main demo for the Hodgkin Huxley neuron model
        """
        start = time.time()
        # X = odeint(self.dALLdt, [-65, 0., 0.95, 0, 0, 1, 0], self.t, args=(self,))

        X =  [[-65, 0., 0.95, 0, 0, 1, 0]]
        for i in self.i_inj:
            X.append(self.integ_comp(X[-1], i, self.dt))
        X = X[1:]

        print(time.time() - start)
        plots_results(self, self.t, self.i_inj, np.array(X))



if __name__ == '__main__':
    runner = HodgkinHuxley()
    runner.Main()