"""
.. module:: cls
    :synopsis: Module containing basic cls abstract class

.. moduleauthor:: Marc Javin
"""

import pylab as plt
from cycler import cycler
from abc import ABC, abstractmethod
import numpy as np
from odynn import utils
import torch


class Model():
    """Abstract class to implement for using a new biological model
    All methods and class variables have to be implemented in order to have the expected behavior

    """
    default_init_state = []
    """array, Initial values for the vector of state variables"""
    default_params = {}
    """dict, Default set of parameters for the model, of the form {<param_name> : value}"""
    _parameter_names = []
    """names of parameters from the model"""
    _constraints = {}
    """dict, Constraints to be applied during optimization
        Should be of the form : {<variable_name> : [lower_bound, upper_bound]}
    """
    _random_bounds = {}


    def __new__(cls, *args, **kwargs):
        obj = ABC.__new__(cls)
        obj._init_names()
        return obj

    def __init__(self, init_p=None, tensors=False, dt=0.1):
        """
        Reshape the initial state and parameters for parallelization in case init_p is a list

        Args:
            init_p(dict): initial parameters of the neuron(s)
            tensors(bool): used in the step function in order to use PyTorch or numpy
            dt(float): time step

        """
        self.dt = dt
        self._init_state = self.default_init_state
        if(init_p is None):
            init_p = {k: np.array([[v]]) for k,v in self.default_params.items()}
            self._num = 1
            self._parallel = 1
        else:
            self._num, self._parallel = init_p[self.parameter_names[0]].shape
        self._tensors = tensors
        self._init_p = init_p
        self._param = self._init_p.copy()
        if tensors:
            self._lib = torch
            self.to_tensor()
        else:
            self._lib = np
            def sigmoid(x):
                return 1 / (1 + np.exp(-x))
            np.sigmoid = sigmoid
        self.dt = dt

    def to_tensor(self):
        # try:
        self._param = {k: torch.tensor(v, dtype=torch.float, requires_grad=True) for k,v in self._param.items()}
        # except:
        #     self._param = {k: torch.Tensor([v]) for k, v in self._param.items()}

    @property
    def num(self):
        """int, Number of neurons being modeled in this object"""
        return self._num

    @property
    def parallel(self):
        """int, Number of neurons being modeled in this object"""
        return self._parallel

    @property
    def parameters(self):
        return self._param

    @property
    def parameter_names(self):
        """int, Number of neurons being modeled in this object"""
        return self._parameter_names

    @property
    def init_state(self):
        """ndarray, Initial state vector"""
        return self._init_state

    @abstractmethod
    def step(self, X, i):
        """
        Integrate and update state variable (voltage and possibly others) after one time step

        Args:
          X(ndarray): State variables
          i(float): Input current

        Returns:
            ndarray: updated state vector

        """
        pass

    @classmethod
    def _init_names(cls):
        cls._parameter_names = list(cls.default_params.keys())

    @classmethod
    def get_random(cls, num=1, parallel=1):
        """Return a dictionnary with the same keys as default_params and random values"""
        return {k: np.random.uniform(v[0], v[1], size=(num,parallel)) for k,v in cls._random_bounds.items()}

    @classmethod
    def create_random(cls, num=1, parallel=1, dt=0.1, tensors=True):
        return cls(cls.get_random(num, parallel), tensors, dt)

    @classmethod
    def create_default(cls, num=1, parallel=1, dt=0.1, tensors=False):
        params = {k: np.full((num, parallel), v) for k, v in cls.default_params.items()}
        return cls(params, tensors, dt)

    def apply_constraints(self):
        # with torch.no_grad():
        for k,c in self._constraints.items():
            self._param[k].data = self._param[k].data.clamp(c[0], c[1])
            # self._param[k].requires_grad = True


class NeuronModel(Model):

    _ions = {}
    V_pos = 0

    def __init__(self, init_p=None, tensors=False, dt=0.1):
        Model.__init__(self, init_p, tensors, dt)

    def calculate(self, i_inj, init=None):
        """
        Simulate the neuron with input current `i_inj` and return the state vectors

        Args:
            i_inj: input currents of shape [time, batch]

        Returns:
            ndarray: series of state vectors of shape [time, state, batch]

        """
        if init is None:
            init = self._init_state
            init = np.repeat(init[:, None], self._num, axis=-1)
            init = init[:, None]
        # init = np.repeat(init[..., None], self._parallel, axis=-1)
        # i_inj = np.repeat(i_inj[..., None], self._parallel, axis=-1)
        if self._tensors:
            init = torch.Tensor(init)
        X = [init]

        # print('Initial states shape : ', init.shape, 'Input current shape : ', i_inj.shape)
        for i in i_inj:
            X.append(self.step(X[-1], i))
        if self._tensors:
            return torch.stack(X[1:])
        else:
            return np.array(X[1:])

    def _inf(self, V, rate):
        """Compute the steady state value of a gate activation rate"""
        mdp = self._param['%s__mdp' % rate]
        scale = self._param['%s__scale' % rate]
        return self._lib.sigmoid((V - mdp) / scale)

    def _update_gate(self, rate, name, V):
        tau = self._param['%s__tau'%name]
        return ((tau * self.dt) / (tau + self.dt)) * ((rate / self.dt) + (self._inf(V, name) / tau))

    @staticmethod
    def plot_results(*args, **kwargs):
        """Function for plotting detailed results of some experiment"""
        pass

    @classmethod
    def plot_output(cls, ts, i_inj, states, y_states=None, suffix="", show=True, save=False, l=1, lt=1,
                    targstyle='-'):
        """
        Plot voltage and ion concentrations, potentially compared to a target model

        Args:
          ts(ndarray of dimension [time]): time steps of the measurements
          i_inj(ndarray of dimension [time]): input current
          states(ndarray of dimension [time, state_var, nb_neuron]):
          y_states(list of ndarray [time, nb_neuron], optional): list of values for the target model, each element is an
            ndarray containing the recordings of one state variable (Default value = None)
          suffix(str): suffix for the name of the saved file (Default value = "")
          show(bool): If True, show the figure (Default value = True)
          save(bool): If True, save the figure (Default value = False)
          l(float): width of the main lines (Default value = 1)
          lt(float): width of the target lines (Default value = 1)
          targstyle(str): style of the target lines (Default value = '-')

        """

        plt.figure()
        nb_plots = len(cls._ions) + 2
        custom_cycler = None
        if (states.ndim > 3):  # circuit in parallel
            states = np.reshape(np.swapaxes(states, -2, -1), (states.shape[0], states.shape[1], -1))
            custom_cycler = cycler('color', utils.COLORS.repeat(y_states[cls.V_pos].shape[1]))
            y_states = [np.reshape(y, (y.shape[0], -1)) if y is not None else None for y in y_states]

        # Plot voltage
        p = plt.subplot(nb_plots, 1, 1)
        if custom_cycler is not None:
            p.set_prop_cycle(custom_cycler)
        plt.plot(ts, states[:, cls.V_pos], linewidth=l)
        if y_states is not None:
            if y_states[cls.V_pos] is not None:
                plt.plot(ts, y_states[cls.V_pos], 'r', linestyle=targstyle, linewidth=lt, label='target model')
                plt.legend()
        plt.ylabel('Voltage (mV)')

        for i, (ion, pos) in enumerate(cls._ions.items()):
            p = plt.subplot(nb_plots, 1, 2 + i)
            if custom_cycler is not None:
                p.set_prop_cycle(custom_cycler)
            plt.plot(ts, states[:, pos], linewidth=l)
            if y_states is not None:
                if y_states[pos] is not None:
                    plt.plot(ts, y_states[pos], 'r', linestyle=targstyle, linewidth=lt, label='target model')
                    plt.legend()
            plt.ylabel('[{}]'.format(ion))

        plt.subplot(nb_plots, 1, nb_plots)
        plt.plot(ts, i_inj, 'b')
        plt.xlabel('t (ms)')
        plt.ylabel('$I_{inj}$ ($\\mu{A}/cm^2$)')

        utils.save_show(show, save, utils.IMG_DIR + 'output_%s' % suffix)