"""
.. module:: cls
    :synopsis: Module containing basic cls abstract class

.. moduleauthor:: Marc Javin
"""

import pylab as plt
from abc import ABC, abstractmethod
import numpy as np
from . import utils
from utils import classproperty




class NeuronModel(ABC):
    """Abstract class to implement for using a new cls
    All methods and class variables have to be implemented in order to have the expected behavior

    Args:

    Returns:

    """
    V_pos = 0
    """int, Default position of the voltage in state vectors"""
    _ions = {}
    """dictionnary, name of ions in the vector states and their positions"""
    default_params = None
    """dict, Default set of parameters for the cls"""
    _constraints_dic = None
    """dict, Constraints to be applied during optimization
        Should be of the form : {<variable_name> : [lower_bound, upper_bound]}
    """
    _init_state = None
    """array, Initial values for the vector of state variables"""

    def __init__(self, init_p=None, tensors=False, dt=0.1):
        """Initialize the attributes
        Reshape the initial state and parameters for parallelization in case init_p is a list

        Args:
          init_p:  (Default value = None)
          tensors:  (Default value = False)
          dt:  (Default value = 0.1)

        Returns:

        
        """
        if(init_p is None):
            init_p = self.default_params
        self._tensors = tensors
        if isinstance(init_p, list):
            self._num = len(init_p)
            init_p = dict([(var, np.array([p[var] for p in init_p], dtype=np.float32)) for var in init_p[0].keys()])
            self._init_state = np.stack([self._init_state for _ in range(self._num)], axis=1)
        else:
            self._num = 1
        self._param = init_p
        self.dt = dt

    @property
    def num(self):
        return self._num

    @property
    def init_state(self):
        return self._init_state

    @classproperty
    def ions(self):
        return self._ions

    @abstractmethod
    def step(self, X, i):
        """Integrate and update voltage after one time step

        Args:
          X(vector): State variables
          i(float): Input current

        Returns:

        
        """
        pass

    @staticmethod
    def get_random():
        """ """
        pass

    @staticmethod
    def plot_results(*args, **kwargs):
        pass

    @classmethod
    def plot_output(cls, ts, i_inj, states, y_states=None, suffix="", show=True, save=False, l=1, lt=1,
                            targstyle='-'):
        """plot voltage and ion concentrations, potentially compared to a target cls

        Args:
          ts(array of dimension [time]): time steps of the measurements
          i_inj(array of dimension [time]): 
          states(array of dimension [time, state_var, nb_neuron]): 
          y_states(list of arrays [time, nb_neuron], optional):  (Default value = None)
          suffix:  (Default value = "")
          show(bool): If True, show the figure (Default value = True)
          save(bool): If True, save the figure (Default value = False)
          l:  (Default value = 1)
          lt:  (Default value = 1)
          targstyle:  (Default value = '-')

        Returns:


        """
        plt.figure()
        nb_plots = len(cls._ions) + 2

        if (states.ndim > 3):
            states = np.reshape(states, (states.shape[0], states.shape[1], -1))
            y_states = [np.reshape(y, (y.shape[0], -1)) if y is not None else None for y in y_states]

        # Plot voltage
        plt.subplot(nb_plots, 1, 1)
        plt.plot(ts, states[:, cls.V_pos], linewidth=l)
        if y_states is not None:
            if y_states[cls.V_pos] is not None:
                plt.plot(ts, y_states[cls.V_pos], 'r', linestyle=targstyle, linewidth=lt, label='target cls')
                plt.legend()
        plt.ylabel('Voltage (mV)')

        for ion, pos in cls._ions.items():
            plt.subplot(nb_plots, 1, 2)
            plt.plot(ts, states[:, pos], linewidth=l)
            if y_states is not None:
                if y_states[pos] is not None:
                    plt.plot(ts, y_states[pos], 'r', linestyle=targstyle, linewidth=lt, label='target cls')
                    plt.legend()
            plt.ylabel('[{}]'.format(ion))

        plt.subplot(nb_plots, 1, nb_plots)
        plt.plot(ts, i_inj, 'b')
        plt.xlabel('t (ms)')
        plt.ylabel('$I_{inj}$ ($\\mu{A}/cm^2$)')

        utils.save_show(show, save, utils.IMG_DIR + 'output_%s' % suffix)
        plt.close()

    @abstractmethod
    def calculate(self, i):
        """Iterate over i (current) and return the state variables obtained after each step

        Args:
          i(ndarray):

        Returns:

        
        """
        pass
