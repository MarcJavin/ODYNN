"""
.. module:: utils
    :synopsis: Module for plots, paths and saving files

.. moduleauthor:: Marc Javin
"""
import numpy as np
import os
import re
import seaborn
import pylab as plt

COLORS = np.array([ 'k', 'c', 'Gold', 'Darkred', 'b', 'Orange', 'm', 'Lime', 'Salmon', 'Indigo', 'DarkGrey', 'Crimson', 'Olive'])


RES_DIR = 'results/'
IMG_DIR = 'img/'
NEUR_DIR = 'neurons/'
SYN_DIR = 'synapses/'
current_dir = RES_DIR
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
    global current_dir
    current_dir = RES_DIR + subdir + '/'
    if not os.path.exists(current_dir):
        os.makedirs(current_dir)
    if not os.path.exists(current_dir + IMG_DIR):
        os.makedirs(current_dir + IMG_DIR)
        os.makedirs(current_dir + 'tmp/')
    return current_dir


def get_dic_from_var(dir, name=""):
    """Get variables values into a dictionnary

    Args:
      dir(str): path to the directory      name:  (Default value = "")

    Returns:

    """
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
    if (save):
        plt.savefig('{}{}.png'.format(current_dir,name), dpi=dpi)
    if (show):
        plt.show()
    plt.close()

def bar(ax, var):
    ax.bar(x=range(len(var)), height=var, color=COLORS)
def plot(ax, var):
    ax.plot(var, linewidth=0.5)
def boxplot(ax, var):
    ax.boxplot(var, vert=True, showmeans=True)

def box(var_dic, cols, labels):
    bp = plt.boxplot([var_dic[k][~np.isnan(var_dic[k])] for k in labels], vert=True, patch_artist=True, showmeans=True, labels=labels)
    for b, color in zip(bp["boxes"], cols):
        b.set_facecolor(color)




def plots_output_mult(ts, i_inj, Vs, Cacs, i_syn=None, labels=None, suffix="", show=True, save=False, l=1):
    """plot multiple voltages and Ca2+ concentration

    Args:
      ts: 
      i_inj: 
      Vs: 
      Cacs: 
      i_syn:  (Default value = None)
      labels:  (Default value = None)
      suffix:  (Default value = "")
      show(bool): If True, show the figure (Default value = True)
      save(bool): If True, save the figure (Default value = False)

    Returns:

    """
    plt.figure()

    if (Vs.ndim > 2):
        Vs = np.reshape(Vs, (Vs.shape[0], -1))
        Cacs = np.reshape(Cacs, (Cacs.shape[0], -1))

    if(labels is None):
        labels = range(len(Vs))
    if (i_syn is not None):
        n_plots = 4
        plt.subplot(n_plots, 1, 3)
        plt.plot(ts, i_syn, linewidth=l)
        plt.xlabel('t (ms)')
        plt.ylabel('$I_{syn}$ ($\\mu{A}/cm^2$)')
    else:
        n_plots = 3

    plt.subplot(n_plots, 1, 1)
    plt.plot(ts, Vs, linewidth=l)
    plt.ylabel('Voltage (mV)')
    leg = plt.legend(labels, bbox_to_anchor=(1, 0.5), handlelength=0.2)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(3.0)

    plt.subplot(n_plots, 1, 2)
    plt.plot(ts, Cacs, linewidth=l)
    plt.ylabel('[$Ca^{2+}$]')

    plt.subplot(n_plots, 1, n_plots)
    if(i_inj.ndim<2):
        plt.plot(ts, i_inj, 'b')
    else:
        plt.plot(ts, i_inj)
    plt.xlabel('t (ms)')
    plt.ylabel('$I_{inj}$ ($\\mu{A}/cm^2$)')

    save_show(show, save, 'Output_%s'%suffix, dpi=250)

    h = seaborn.heatmap(Vs.transpose(), yticklabels=labels, cmap='RdYlBu_r', xticklabels=False)
    save_show(show, save, 'Voltage_%s' % suffix, dpi=250)

    seaborn.heatmap(Cacs.transpose(), yticklabels=labels, cmap='RdYlBu_r', xticklabels=False)
    save_show(show, save, 'Calcium_%s' % suffix, dpi=250)




