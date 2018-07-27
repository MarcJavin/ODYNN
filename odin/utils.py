"""
.. module:: utils
    :synopsis: Module for plots, paths and saving files

.. moduleauthor:: Marc Javin
"""
import numpy as np
import os
import re
import seaborn as sns
import pylab as plt

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
    """Set directory to save files

    Args:
      subdir(str): path to the directory
    Returns:

    """
    global _current_dir
    _current_dir = RES_DIR + subdir + '/'
    if not os.path.exists(_current_dir):
        os.makedirs(_current_dir)
    for sd in subdirs:
        if not os.path.exists(_current_dir + sd):
            os.makedirs(_current_dir + sd)
    return _current_dir


def get_dic_from_var(dir, name=""):
    """Get variables values into a dictionnary"""
    file = '{}{}/{}_{}.txt'.format(RES_DIR, dir, OUT_PARAMS, name)
    dic = {}
    with open(file, 'r') as f:
        for line in f:
            m = re.search(REGEX_VARS, line)
            if(m.group(2)[0]=='['):
                # several params
                l = re.findall('[-]?[\d]*[\.]?[\d]+[e]?[+-]?[\d]+', m.group(2))
                l = map(float, l)
                dic[m.group(1)] = l
            else:
                dic[m.group(1)] = float(m.group(2))
    return dic


def save_show(show, save, name='', dpi=100):
    global _current_dir
    if (save):
        plt.savefig('{}{}.png'.format(_current_dir,name), dpi=dpi)
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


def clamp(val, minimum=0, maximum=255):
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




