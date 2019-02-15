"""
.. module:: 
    :synopsis: Module doing stuff...

.. moduleauthor:: Marc Javin
"""

from .leakint import LeakyIntegrate
from .hhsimple import HodgHuxSimple
from .celeg import CElegansNeuron
from .chemsyn import ChemSyn

model_classes = [CElegansNeuron, LeakyIntegrate, HodgHuxSimple]
model_names = ['celeg', 'leakint', 'hhsimp']
neur_models = {'celeg': CElegansNeuron,
               'leakint': LeakyIntegrate,
               'hhsimp': HodgHuxSimple}

syn_models = {'classic': ChemSyn}