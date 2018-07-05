"""
.. module:: model
    :synopsis: Module containing basic model abstract class

.. moduleauthor:: Marc Javin
"""


from abc import ABC, abstractmethod
import numpy as np




class Model(ABC):
    """
    Abstract class to implement for using a new model
    All methods and class variables have to be implemented in order to have the expected behavior
    """
    V_pos = 0
    """int, Default position of the voltage in state vectors"""
    Ca_pos = -1
    """int, Default position of the calcium concentration in state vectors"""
    ions_in_state = {}
    """dictionnary, name of ions in the vector states and their positions"""
    default_params = None
    """dict, Default set of parameters for the model"""
    _constraints_dic = None
    """dict, Constraints to be applied during optimization
        Should be of the form : {<variable_name> : [lower_bound, upper_bound]}
    """
    _init_state = None
    """array, Initial values for the vector of state variables"""

    def __init__(self, init_p=None, tensors=False, dt=0.1):
        """
        Initialize the attributes
        Reshape the initial state and parameters for parallelization in case init_p is a list

        Parameters
        ----------
        init_p : dict or list
            Values of the parameters in a dictionnary
            In case of a list, the instance will behave as a set of neurons
        tensors : bool
            In order to define functions in pure python or tensorflow
        dt : float
            timestep of the system
        """
        if(init_p is None):
            init_p = self.default_params
        self._tensors = tensors
        if isinstance(init_p, list):
            self.num = len(init_p)
            init_p = dict([(var, np.array([p[var] for p in init_p], dtype=np.float32)) for var in init_p[0].keys()])
            self._init_state = np.stack([self._init_state for _ in range(self.num)], axis=1)
        else:
            self.num = 1
        self._param = init_p
        self.dt = dt

    @staticmethod
    @abstractmethod
    def step_model(X, i, self):
        """
        Integrate and update voltage after one time step

        Parameters
        ----------
        X : vector
            State variables
        i : float
            Input current

        Returns
        ----------
        Vector with the same size as X containing the updated state variables
        """
        pass

    @staticmethod
    def get_random():
        """Return a dictionnary with random parameters"""
        pass

    @staticmethod
    def plot_results(*args, **kwargs):
        pass

    @staticmethod
    def plot_output(t, i_inj, states, states_y, suffix='', show=False, save=False, l=1, lt=1, targstyle='-'):
        pass

    @abstractmethod
    def calculate(self, i):
        """
        Iterate over i (current) and return the state variables obtained after each step
        Parameters
        ---------
        i : array
            array of successive input current to the neuron
        Returns
        ---------
        Array of dimension [len(i), len(init_state)]
        """
        pass
