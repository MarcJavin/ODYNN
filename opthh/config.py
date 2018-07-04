"""
.. module:: config
    :synopsis: Module for configuration of the project, mainly which neuron model is used

.. moduleauthor:: Marc Javin
"""

from .hhmodel import HodgkinHuxley

NEURON_MODEL = HodgkinHuxley
"""Class used for neuron models"""


# Use on my server
import matplotlib as mpl
import socket
if (socket.gethostname()=='1080'):
    mpl.use("Agg")


# Tune the plots appearance
COLORS = [ 'k', 'c', 'Gold', 'Darkred', 'b', 'Orange', 'm', 'Lime', 'Salmon', 'Indigo', 'DarkGrey', 'Crimson', 'Olive']
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=COLORS)
import pylab as plt
SMALL_SIZE = 8
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels