"""
.. module:: utils
    :synopsis: Module for plots, paths and saving files

.. moduleauthor:: Marc Javin
"""
import numpy as np
import os
import pylab as plt
import seaborn as sns

COLORS = np.array([ 'k', 'c', 'Gold', 'Darkred', 'b', 'Orange', 'm', 'Lime', 'Salmon', 'Indigo', 'DarkGrey', 'Crimson', 'Olive'])

RES_DIR = 'results/'
IMG_DIR = 'img/'
NEUR_DIR = 'neurons/'
SYN_DIR = 'synapses/'
TMP_DIR = 'tmp/'
subdirs = [IMG_DIR, NEUR_DIR, SYN_DIR, TMP_DIR]
_current_dir = RES_DIR
OUT_PARAMS = 'params'
OUT_SETTINGS = 'settings'
REGEX_VARS = '(.*) : (.*)'


class classproperty(object):

    def __init__(self, fget):
        self.fget = fget

    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_cls)


def set_dir(subdir):
    """Set directory for saving files to `utils.RES_DIR`+`subdir` and create subfolders

    Args:
      subdir(str): name of the directory
    Returns:
        str: complete path to the new directory

    """
    global _current_dir
    _current_dir = RES_DIR + subdir + '/'
    if not os.path.exists(_current_dir):
        os.makedirs(_current_dir)
    for sd in subdirs:
        if not os.path.exists(_current_dir + sd):
            os.makedirs(_current_dir + sd)
    return _current_dir


def save_show(show, save, name='', dpi=500):
    """
    Show and/or save the current plot in `utils.current_dir`/`name`
    Args:
        show(bool): If True, show the plot
        save(bool): If True, save the plot
        name(str): Name for the saved file
        dpi(int): quality

    """
    global _current_dir
    if (save):
        plt.savefig('{}{}.png'.format(_current_dir,name), format='png', dpi=dpi)
    if (show):
        plt.show()
    plt.close()


def bar(ax, var, good_val=None):
    sns.barplot(x=np.arange(len(var)), y=var, ax=ax)
    if good_val is not None:
        ax.axhline(y=good_val, color='r', label='target value')
    ax.set_xticks([])
def plot(ax, var, good_val=None):
    ax.plot(var, linewidth=0.5)
    if good_val is not None:
        ax.axhline(y=good_val, color='r', label='target value')
def boxplot(ax, var):
    ax.boxplot(var, vert=True, showmeans=True)


def box(df, cols, labels):
    lighter = [colorscale(c, 0.6) for c in cols]
    sns.boxplot(data=df[labels], palette = lighter)
    sns.swarmplot(data=df[labels], palette=cols, size=2)
    # from odynn.models.cfg_model import NEURON_MODEL
    # import pandas as pd
    # dd = pd.DataFrame(NEURON_MODEL.default_params, index=[0])
    # sns.swarmplot(data=dd[labels], color='r', edgecolor='#ffffff', marker='*', linewidth=1, size=20)


def clamp(val, minimum=0, maximum=255):
    """
        Clamp `val` between `minimum` and `maximum`
    Args:
        val(float): value to clamp
        minimum(int): minimum
        maximum(int): maximum

    Returns:
        int: clamped value

    """
    if val < minimum:
        return minimum
    if val > maximum:
        return maximum
    return int(val)


def colorscale(hexstr, scalefactor):
    """
    Scales a hex string by ``scalefactor``. Returns scaled hex string.
    """

    hexstr = hexstr.strip('#')

    if scalefactor < 0 or len(hexstr) != 6:
        return hexstr

    r, g, b = int(hexstr[:2], 16), int(hexstr[2:4], 16), int(hexstr[4:], 16)

    r = clamp(r * scalefactor)
    g = clamp(g * scalefactor)
    b = clamp(b * scalefactor)

    return "#%02x%02x%02x" % (r, g, b)




