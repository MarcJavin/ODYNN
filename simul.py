import scipy as sp
from scipy.integrate import odeint
import time
from utils import plots_results, get_data
import utils
import params
import numpy as np
import pickle

T, X, Y = get_data()
Y = (Y-50)*60
X = X*10
DT = T[1] - T[0]

class HodgkinHuxley():
    """Full Hodgkin-Huxley Model implemented in Python"""


    DECAY_CA = params.DECAY_CA
    RHO_CA = params.RHO_CA
    REST_CA = params.REST_CA
    
    param = params.params

    dt = 0.1
    t = params.t
    i_inj = params.i_inj
    """ The time to integrate over """

    def inf(self, V, rate, p=param):
        mdp = p['%s__mdp'%rate]
        scale = p['%s__scale'%rate]
        return 1 / (1 + sp.exp((mdp - V)/scale))

    def h(self, cac, p=param):
        """Channel gating kinetics. Functions of membrane voltage"""
        q = self.inf(cac, 'h')
        return 1 + (q-1)*p['h__alpha']

    h_notensor = h

    def I_Ca(self, V, e, f, h):
        """
        Membrane current (in uA/cm^2)
        Sodium (Na = element name)
        """
        return self.param['g_Ca'] * e ** 2 * f * h * (V - self.param['E_Ca'])

    def I_Kf(self, V, p, q):
        """
        Membrane current (in uA/cm^2)
        Potassium (K = element name)
        """
        return self.param['g_Kf'] * p ** 4 * q * (V - self.param['E_K'])

    def I_Ks(self, V, n):
        """
        Membrane current (in uA/cm^2)
        Potassium (K = element name)
        """
        return self.param['g_Ks'] * n * (V - self.param['E_K'])

    #  Leak
    def I_L(self, V):
        """
        Membrane current (in uA/cm^2)
        Leak
        """
        return self.param['g_L'] * (V - self.param['E_L'])

    def I_inj(self, t):
        """
        External Current
        """
        return 5*((t>000) and (t<500))

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
        dVdt = (self.I_inj(t) - self.I_Ca(V, e, f, h) - self.I_Ks(V, n) - self.I_Kf(V, p, q) - self.I_L(V)) / self.param['C_m']
        dpdt = (self.inf(V, 'p') - p) / self.param['p__tau']
        dqdt = (self.inf(V, 'q') - q) / self.param['q__tau']
        dedt = (self.inf(V, 'e') - e) / self.param['e__tau']
        dfdt = (self.inf(V, 'f') - f) / self.param['f__tau']
        dndt = (self.inf(V, 'n') - n) / self.param['n__tau']
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
        V += ((i_inj - self.I_Ca(V, e, f, h) - self.I_Ks(V, n) - self.I_Kf(V, p, q) - self.I_L(V)) / self.param['C_m']) * dt
        cac += (-self.I_Ca(V, e, f, h) * self.RHO_CA - ((cac - self.REST_CA) / self.DECAY_CA)) * dt
        # cac = (self.DECAY_CA/(dt+self.DECAY_CA)) * (cac - self.I_Ca(V, e, f, h)*self.RHO_CA*dt + self.REST_CA*self.DECAY_CA/dt)
        tau = self.param['p__tau']
        p = ((tau*dt) / (tau+dt)) * ((p/dt) + (self.inf(V, 'p')/tau))
        tau = self.param['q__tau']
        q = ((tau*dt) / (tau+dt)) * ((q/dt) + (self.inf(V, 'q')/tau))
        tau = self.param['e__tau']
        e = ((tau*dt) / (tau+dt)) * ((e/dt) + (self.inf(V, 'e')/tau))
        tau = self.param['f__tau']
        f = ((tau*dt) / (tau+dt)) * ((f/dt) + (self.inf(V, 'f')/tau))
        tau = self.param['n__tau']
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
            V)) / self.param['C_m']) * dt
        cac += (-self.I_Ca(V, e, f, h) * self.RHO_CA - ((cac - self.REST_CA) / self.DECAY_CA)) * dt
        cac = (self.DECAY_CA / (dt + self.DECAY_CA)) * (
                    cac - self.I_Ca(V, e, f, h) * self.RHO_CA * dt + self.REST_CA * self.DECAY_CA / dt)
        p = self.inf(V, 'p')
        q = self.inf(V, 'q')
        e = self.inf(V, 'e')
        f = self.inf(V, 'f')
        n = self.inf(V, 'n')
        return [V, p, q, n, e, f, cac]


    def fitness(self, params):
        idx = 0
        for k, v in self.param.items():
            self.param[v] = params[idx]
            idx += 1
        print(self.param)
        S = [[-50, 0., 0.95, 0, 0, 1, 0]]
        div = DT/self.dt
        for i in X:
            S.append(self.integ_comp(S[-1], i, self.dt))
            s = S[-1]
            for d in range(div-1):
                s = self.integ_comp(s, i, self.dt)
        S = np.array(S[1:])
        V = S[:,0]
        mse = ((Y - V)**2).mean()
        return mse

    def get_bounds(self):
        m = -100
        M = 1000
        low = np.tile([0, m, m], 7), [0, 0, 0, 0, 0, m, m, m]
        up = np.full((29), M)
        return (low, up)



    def Main(self):
        """
        Main demo for the Hodgkin Huxley neuron model
        """
        start = time.time()
        # X = odeint(self.dALLdt, [-65, 0., 0.95, 0, 0, 1, 0], self.t, args=(self,))
        #
        X =  [[-50, 0., 0.95, 0, 0, 1, 0]]
        for i in self.i_inj:
            X.append(self.integ_comp(X[-1], i, self.dt))
        X = np.array(X[1:])

        todump = np.vstack((self.t, self.i_inj, X[:,0], X[:,-1]))

        with open(utils.DUMP_FILE, 'wb') as f:
            pickle.dump(todump, f)


        print(time.time() - start)
        plots_results(self, self.t, self.i_inj, np.array(X))



if __name__ == '__main__':
    runner = HodgkinHuxley()
    runner.Main()