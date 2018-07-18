"""
.. module:: config
    :synopsis: Module for configuration of the project, mainly which neuron model is used

.. moduleauthor:: Marc Javin
"""

from .hhmodel import CElegansNeuron

NEURON_MODEL = CElegansNeuron
"""Class used for biological neuron models"""
