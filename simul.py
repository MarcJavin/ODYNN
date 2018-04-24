import scipy as sp
import pylab as plt
from scipy.integrate import odeint

PARAM_CHANNELS = {
        'p__tau': 2.25518,
        'p__scale': 7.42636,
        'p__midpoint': -8.05232,

        'q__tau': 149.963,
        'q__scale': -9.97468,
        'q__midpoint': -15.6456,

        'n__tau': 25.0007,
        'n__scale': 15.8512,
        'n__midpoint': 19.8741,

        'e__tau': 0.100027,
        'e__scale': 6.74821,
        'e__midpoint': -3.3568,

        'f__tau': 150.88,
        'f__scale': -5.03176,
        'f__midpoint': 25.1815,

        'e__tau': 0.100027,
        'e__scale': 6.74821,
        'e__midpoint': -3.3568,

        'h__alpha' : 0.282473,
        'h__k' : -1.00056e-11,
        'h__ca_half' : 6.41889e-11
}

class HodgkinHuxley():
    """Full Hodgkin-Huxley Model implemented in Python"""

    C_m  =   1.0
    """membrane capacitance, in uF/cm^2"""

    g_Ca = .0
    """Calcium (Na) maximum conductances, in mS/cm^2"""

    g_Ks = 3.0
    g_Kf = 0.07
    """Postassium (K) maximum conductances, in mS/cm^2"""

    g_L  =   0.005
    """Leak maximum conductances, in mS/cm^2"""

    E_Ca =  40.0
    """Sodium (Na) Nernst reversal potentials, in mV"""

    E_K  = -60.0
    """Postassium (K) Nernst reversal potentials, in mV"""

    E_L  = -50.0
    """Leak Nernst reversal potentials, in mV"""

    DECAY_CA = 11.6  # s
    RHO_CA = 0.000239e-2  # mol_per_cm_per_A_per_s
    REST_CA = 0  # M


    t = sp.arange(0.0, 450.0, 0.01)
    """ The time to integrate over """

    def inf_e(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 1 / (1 + sp.exp((PARAM_CHANNELS['e__midpoint'] - V)/PARAM_CHANNELS['e__scale']))

    def inf_f(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 1 / (1 + sp.exp((PARAM_CHANNELS['f__midpoint'] - V)/PARAM_CHANNELS['f__scale']))

    def inf_p(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 1 / (1 + sp.exp((PARAM_CHANNELS['p__midpoint'] - V)/PARAM_CHANNELS['p__scale']))

    def inf_q(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 1 / (1 + sp.exp((PARAM_CHANNELS['q__midpoint'] - V)/PARAM_CHANNELS['q__scale']))

    def inf_n(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 1 / (1 + sp.exp((PARAM_CHANNELS['n__midpoint'] - V)/PARAM_CHANNELS['n__scale']))

    def h(self, cac):
        """Channel gating kinetics. Functions of membrane voltage"""
        q = 1 / (1 + sp.exp((PARAM_CHANNELS['h__ca_half'] - cac)/PARAM_CHANNELS['h__k']))
        return 1 + (q-1)*PARAM_CHANNELS['h__alpha']

    def I_Ca(self, V, e, f, h):
        """
        Membrane current (in uA/cm^2)
        Sodium (Na = element name)

        |  :param V:
        |  :param m:
        |  :param h:
        |  :return:
        """
        return self.g_Ca * e**2 * f * h * (V - self.E_Ca)

    def I_Kf(self, V, p, q):
        """
        Membrane current (in uA/cm^2)
        Potassium (K = element name)

        |  :param V:
        |  :param h:
        |  :return:
        """
        return self.g_Kf  * p**4 * q * (V - self.E_K)
    
    def I_Ks(self, V, n):
        """
        Membrane current (in uA/cm^2)
        Potassium (K = element name)

        |  :param V:
        |  :param h:
        |  :return:
        """
        return self.g_Ks * n * (V - self.E_K)
    
    #  Leak
    def I_L(self, V):
        """
        Membrane current (in uA/cm^2)
        Leak

        |  :param V:
        |  :param h:
        |  :return:
        """
        return self.g_L * (V - self.E_L)

    def I_inj(self, t):
        """
        External Current

        |  :param t: time
        |  :return: step up to 10 uA/cm^2 at t>100
        |           step down to 0 uA/cm^2 at t>200
        |           step up to 35 uA/cm^2 at t>300
        |           step down to 0 uA/cm^2 at t>400
        """
        return 10*(t>100) - 10*(t>200) + 35*(t>300) - 35*(t>400)

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
        dpdt = (self.inf_p(V) - p) / PARAM_CHANNELS['p__tau']
        dqdt = (self.inf_q(V) - q) / PARAM_CHANNELS['q__tau']
        dedt = (self.inf_e(V) - e) / PARAM_CHANNELS['e__tau']
        dfdt = (self.inf_f(V) - f) / PARAM_CHANNELS['f__tau']
        dndt = (self.inf_n(V) - n) / PARAM_CHANNELS['n__tau']
        dcacdt = self.I_Ca(V, e, f, h) * self.RHO_CA - ((cac - self.REST_CA) / self.DECAY_CA)
        return dVdt, dpdt, dqdt, dndt, dedt, dfdt, dcacdt

    def Main(self):
        """
        Main demo for the Hodgkin Huxley neuron model
        """

        X = odeint(self.dALLdt, [-65, 0.5, 0.5, 0.5, 0.5, 0.5, 1e-10], self.t, args=(self,))
        V = X[:,0]
        p = X[:,1]
        q = X[:,2]
        n = X[:,3]
        e = X[:,4]
        f = X[:,5]
        cac = X[:,6]
        h = self.h(cac)
        ica = self.I_Ca(V, e, f, h)
        ik = self.I_Ks(V, n) + self.I_Kf(V, p, q)
        il = self.I_L(V)

        plt.figure()

        plt.subplot(5,1,1)
        plt.title('Hodgkin-Huxley Neuron')
        plt.plot(self.t, V, 'k')
        plt.ylabel('V (mV)')

        plt.subplot(5,1,2)
        plt.plot(self.t, ica, 'c', label='$I_{Na}$')
        plt.plot(self.t, ik, 'y', label='$I_{K}$')
        plt.plot(self.t, il, 'm', label='$I_{L}$')
        plt.ylabel('Current')
        plt.legend()

        plt.subplot(5,1,3)
        plt.plot(self.t, p, 'r', label='p')
        plt.plot(self.t, q, 'g', label='q')
        plt.plot(self.t, n, 'b', label='n')
        plt.ylabel('Gating Value')
        plt.legend()

        plt.subplot(5, 1, 4)
        plt.plot(self.t, e, 'r', label='e')
        plt.plot(self.t, f, 'g', label='f')
        plt.plot(self.t, h, 'b', label='h')
        plt.ylabel('Gating Value')
        plt.legend()

        plt.subplot(5,1,5)
        i_inj_values = [self.I_inj(t) for t in self.t]
        plt.plot(self.t, i_inj_values, 'k')
        plt.xlabel('t (ms)')
        plt.ylabel('$I_{inj}$ ($\\mu{A}/cm^2$)')
        plt.ylim(-1, 40)

        plt.show()

if __name__ == '__main__':
    runner = HodgkinHuxley()
    runner.Main()