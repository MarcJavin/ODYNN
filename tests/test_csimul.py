"""
.. module::
    :synopsis: Module doing stuff...

.. moduleauthor:: Marc Javin
"""

from unittest import TestCase
from odynn import circuit, csimul, neuron
import numpy as np


class Testcsimul(TestCase):

    def test_simul(self):
        syns = {(0,1):circuit.SYNAPSE, (1,2):circuit.SYNAPSE}
        gaps  ={(2,1): circuit.GAP}
        t = [0., 0.1, 0.2, 0.3]
        i = np.ones((4,3))
        p = [neuron.PyBioNeuron.default_params for _ in range(3)]
        csimul.simul(t, i, p, syns, gaps)