"""
.. module:: config
    :synopsis: Module for configuration of the project, mainly which neuron model is used

.. moduleauthor:: Marc Javin
"""

from .hhmodel import CElegansNeuron
from .leakint import Custom

NEURON_MODEL = Custom
"""Class used for biological neuron models"""
