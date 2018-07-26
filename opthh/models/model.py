"""
.. module:: cls
    :synopsis: Module containing basic cls abstract class

.. moduleauthor:: Marc Javin
"""

import pylab as plt
from cycler import cycler
from abc import ABC, abstractmethod
import numpy as np
from opthh import utils
from opthh.utils import classproperty


class Neuron(ABC):
    V_pos = 0
    """int, Default position of the voltage in state vectors"""
    _ions = {}
    """dictionnary, name of ions in the vector states and their positions"""
    default_init_state = None
    """array, Initial values for the vector of state variables"""

    def __init__(self, dt=0.1):
        self.dt = dt
        self._init_state = self.default_init_state

    @property
    def num(self):
        """int, Number of neurons being modeled in this object"""
        return self._num

    @property
    def init_state(self):
        """ndarray, Initial state vector"""
        return self._init_state

    @classproperty
    def ions(self):
        """dict, contains the names of modeled ion concentrations as keys and their position in init_state as values"""
        return self._ions

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
        if (states.ndim > 3): # circuit in parallel
            states = np.reshape(np.swapaxes(states,-2,-1), (states.shape[0], states.shape[1], -1))
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

        for ion, pos in cls._ions.items():
            p = plt.subplot(nb_plots, 1, 2)
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

    @abstractmethod
    def calculate(self, i):
        """Iterate over i (current) and return the state variables obtained after each step

        Args:
          i(ndarray): input current, dimension [time, (batch, (self.num))]

        Returns:
            ndarray: state vectors concatenated [i.shape[0], len(self.init_state)(, i.shape[1], (i.shape[2]))]
        """
        pass



class BioNeuron(Neuron):
    """Abstract class to implement for using a new biological model
    All methods and class variables have to be implemented in order to have the expected behavior

    """
    default_params = None
    """dict, Default set of parameters for the model, of the form {<param_name> : value}"""
    parameter_names = None
    _constraints_dic = None
    """dict, Constraints to be applied during optimization
        Should be of the form : {<variable_name> : [lower_bound, upper_bound]}
    """


    def __init__(self, init_p=None, tensors=False, dt=0.1):
        """
        Reshape the initial state and parameters for parallelization in case init_p is a list

        Args:
            init_p(dict or list of dict): initial parameters of the neuron(s). If init_p is a list, then this object
                will model n = len(init_p) neurons
            tensors(bool): used in the step function in order to use tensorflow or numpy
            dt(float): time step

        """
        Neuron.__init__(self, dt=dt)
        if(init_p is None):
            init_p = self.default_params
        elif(init_p == 'random'):
            init_p = self.get_random()
        elif isinstance(init_p, list):
            self._num = len(init_p)
            init_p = dict([(var, np.array([p[var] for p in init_p], dtype=np.float32)) for var in init_p[0].keys()])
            self._init_state = np.stack([self._init_state for _ in range(self._num)], axis=-1)
        elif hasattr(init_p['C_m'], '__len__'):
            self._num = len(init_p['C_m'])
            if isinstance(init_p['C_m'], list):
                init_p = {var: np.array(val, dtype=np.float32) for var, val in init_p.items()}
            self._init_state = np.stack([self._init_state for _ in range(self._num)], axis=-1)
        else:
            self._num = 1
        self._tensors = tensors
        self._init_p = init_p
        self.dt = dt
        self.init_names()

    @classmethod
    def init_names(cls):
        cls.parameter_names = list(cls.default_params.keys())

    @staticmethod
    def get_random():
        """Return a dictionnary with the same keys as default_params and random values"""
        pass

    @staticmethod
    def plot_results(*args, **kwargs):
        """Function for plotting detailed results of some experiment"""
        pass
