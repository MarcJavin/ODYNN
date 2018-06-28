"""
.. module:: config
    :synopsis: Module for configuration of the project, mainly which neuron model is used

.. moduleauthor:: Marc Javin
"""

from opthh.hhmodel import HodgkinHuxley

NEURON_MODEL = HodgkinHuxley
"""Class used for neuron models"""