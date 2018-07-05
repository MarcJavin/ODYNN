"""
.. module:: utils
    :synopsis: Module for plots, paths and saving files

.. moduleauthor:: Marc Javin
"""
import matplotlib
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter
import pylab as plt
import numpy as np
import os
import re

# Use on my server
import matplotlib as mpl
import socket
if (socket.gethostname()=='1080'):
    mpl.use("Agg")


RES_DIR = 'results/'
IMG_DIR = 'img/'
current_dir = RES_DIR
OUT_PARAMS = 'params'
OUT_SETTINGS = 'settings'
REGEX_VARS = '(.*) : (.*)'

RATE_COLORS = {'p' : '#00ccff',
               'q' : '#0000ff',
               'n' : '#cc00ff',
               'e' : '#b30000',
               'f' : '#ff9900',
               'h' : '#ff1a1a'
                }
GATES = ['e', 'f', 'n', 'p', 'q']
CONDS = ['g_Ks', 'g_Kf', 'g_Ca', 'g_L']
MEMB = ['C_m', 'E_K', 'E_Ca', 'E_L']



def set_dir(subdir):
    """Set directory to save files"""
    global current_dir
    current_dir = RES_DIR + subdir + '/'
    if not os.path.exists(current_dir):
        os.makedirs(current_dir)
    if not os.path.exists(current_dir + IMG_DIR):
        os.makedirs(current_dir + IMG_DIR)
        os.makedirs(current_dir + 'tmp/')
    return current_dir


def get_dic_from_var(dir, suffix=""):
    """Get variables values into a dictionnary"""
    file = '{}{}/{}_{}.txt'.format(RES_DIR, dir, OUT_PARAMS, suffix)
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


def bar(ax, var):
    ax.bar(x=range(len(var)), height=var, color=matplotlib.rcParams['axes.prop_cycle'])
def plot(ax, var):
    ax.plot(var, linewidth=0.5)
def boxplot(ax, var):
    ax.boxplot(var, vert=True, showmeans=True)

def box(var_dic, cols, labels):
    bp = plt.boxplot([var_dic[k][~np.isnan(var_dic[k])] for k in labels], vert=True, patch_artist=True, showmeans=True, labels=labels)
    for b, color in zip(bp["boxes"], cols):
        b.set_facecolor(color)

def boxplot_vars(var_dic, suffix="", show=True, save=False):
    plt.figure()
    plt.subplot(121)
    cols=[RATE_COLORS['n'], RATE_COLORS['n'], RATE_COLORS['f'], 'k']
    labels = CONDS
    box(var_dic, cols, labels)
    plt.title('Conductances')
    plt.subplot(122)
    cols = ['b', RATE_COLORS['n'], RATE_COLORS['f'], 'k']
    labels = MEMB
    box(var_dic, cols, labels)
    plt.title('Membrane')
    if (save):
        plt.savefig('{}var_{}_{}.png'.format(current_dir, 'Membrane', suffix), dpi=300)
    if (show):
        plt.show()

    plt.figure()
    plt.subplot(211)
    box(var_dic, ['k'], ['rho_ca'])
    plt.title('Rho_ca')
    plt.subplot(212)
    box(var_dic, ['b'], ['decay_ca'])  # , 'b')
    plt.title('Decay_ca')
    plt.tight_layout()
    if (save):
        plt.savefig('{}var_{}_{}.png'.format(current_dir, 'CalciumPump', suffix), dpi=300)
    if (show):
        plt.show()

    plt.figure()
    for i, type in enumerate(['mdp', 'scale', 'tau']):
        plt.subplot(3,1,i+1)
        plt.title(type)
        labels = ['{}__{}'.format(rate, type) for rate in RATE_COLORS.keys()]
        cols = RATE_COLORS.values()
        if(type=='tau'):
            labels = ['h__alpha' if x=='h__tau' else x for x in labels]
        box(var_dic, cols, labels)
    if (save):
        plt.savefig('{}var_{}_{}.png'.format(current_dir, 'Rates', suffix), dpi=300)
    if (show):
        plt.show()
    plt.close()

def plot_vars_syn(var_dic, suffix="", show=True, save=False, func=plot):
    """plot variation/comparison/boxplots of synaptic variables"""
    plt.figure()
    if(list(var_dic.values())[0].ndim > 2):
        var_dic = dict([(var, np.reshape(val, (val.shape[0], -1))) for var, val in var_dic.items()])
    for i,var in enumerate(['G', 'mdp', 'E', 'scale']):
        plt.subplot(2,2,i+1)
        func(plt, var_dic[var])
        plt.ylabel(var)
    plt.tight_layout()
    if (save):
        plt.savefig('{}var_{}_{}.png'.format(current_dir, 'Synapses', suffix), dpi=300)
    if (show):
        plt.show()
    plt.close()



def plot_vars(var_dic, suffix="", show=True, save=False, func=plot):
    """plot variation/comparison/boxplots of all variables organized by categories"""
    fig = plt.figure()
    grid = plt.GridSpec(2, 3)
    for nb in range(len(GATES)):
        gate = GATES[nb]
        plot_vars_gate(gate, var_dic['{}__mdp'.format(gate)], var_dic['{}__scale'.format(gate)],
                       var_dic['{}__tau'.format(gate)], fig, grid[nb], (nb % 3 == 0), func=func)
    plot_vars_gate('h', var_dic['h__mdp'], var_dic['h__scale'],
                   var_dic['h__alpha'], fig, grid[5], False, func=func)
    plt.tight_layout()
    if save:
        plt.savefig('{}var_{}_{}.png'.format(current_dir, 'Rates', suffix), dpi=300)
    if show:
        plt.show()

    fig = plt.figure()
    grid = plt.GridSpec(1, 2)
    subgrid = gridspec.GridSpecFromSubplotSpec(4, 1, grid[0], hspace=0.1)
    for i, var in enumerate(CONDS):
        ax = plt.Subplot(fig, subgrid[i])
        func(ax, var_dic[var])#)
        ax.set_ylabel(var)
        if(i==0):
            ax.set_title('Conductances')
        fig.add_subplot(ax)
    subgrid = gridspec.GridSpecFromSubplotSpec(4, 1, grid[1], hspace=0.1)
    for i, var in enumerate(MEMB):
        ax = plt.Subplot(fig, subgrid[i])
        func(ax, var_dic[var])#)
        ax.set_ylabel(var)
        if(i==0):
            ax.set_title('Membrane')
        ax.yaxis.tick_right()
        fig.add_subplot(ax)
    plt.tight_layout()
    if save:
        plt.savefig('{}var_{}_{}.png'.format(current_dir, 'Membrane', suffix), dpi=300)
    if show:
        plt.show()

    plt.figure()
    ax = plt.subplot(211)
    func(ax, var_dic['rho_ca'])#, 'r')
    plt.ylabel('Rho_ca')
    ax = plt.subplot(212)
    func(ax, var_dic['decay_ca'])#, 'b')
    plt.ylabel('Decay_ca')
    if (save):
        plt.savefig('{}var_{}_{}.png'.format(current_dir, 'CalciumPump', suffix), dpi=300)
    if (show):
        plt.show()

    plt.close('all')

def plot_vars_gate(name, mdp, scale, tau, fig, pos, labs, func=plot):
    """plot the gates variables"""
    subgrid = gridspec.GridSpecFromSubplotSpec(3,1,pos, hspace=0.1)
    vars = [('Midpoint',mdp), ('Scale',scale), ('Tau',tau)]
    for i, var in enumerate(vars):
        ax = plt.Subplot(fig, subgrid[i])
        func(ax, var[1])#, 'r')
        ax.set_xlabel([])
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        if(labs):
            ax.set_ylabel(var[0])
        if(i==0):
            ax.set_title(name)
        fig.add_subplot(ax)

def plots_output_mult(ts, i_inj, Vs, Cacs, i_syn=None, labels=None, suffix="", show=True, save=False):
    """plot multiple voltages and Ca2+ concentration"""
    plt.figure()

    if (Vs.ndim > 2):
        Vs = np.reshape(Vs, (Vs.shape[0], -1))
        Cacs = np.reshape(Cacs, (Cacs.shape[0], -1))

    if(labels is None):
        labels = range(len(Vs))
    if (i_syn is not None):
        n_plots = 4
        plt.subplot(n_plots, 1, 3)
        plt.plot(ts, i_syn)
        plt.xlabel('t (ms)')
        plt.ylabel('$I_{syn}$ ($\\mu{A}/cm^2$)')
        plt.legend(labels, ncol=int(Vs.shape[1]/4)+1)
    else:
        n_plots = 3

    plt.subplot(n_plots, 1, 1)
    plt.plot(ts, Vs)
    plt.ylabel('Voltage (mV)')
    plt.legend(labels, ncol=int(Vs.shape[1]/4)+1)

    plt.subplot(n_plots, 1, 2)
    plt.plot(ts, Cacs)
    plt.ylabel('[$Ca^{2+}$]')

    plt.subplot(n_plots, 1, n_plots)
    if(i_inj.ndim<2):
        plt.plot(ts, i_inj, 'b')
    else:
        plt.plot(ts, i_inj)
    plt.xlabel('t (ms)')
    plt.ylabel('$I_{inj}$ ($\\mu{A}/cm^2$)')

    if (save):
        plt.savefig('{}{}output_{}.png'.format(current_dir, IMG_DIR, suffix))
    if (show):
        plt.show()
    plt.close()

def plots_output_double(model, ts, i_inj, states, y_states=None, suffix="", show=True, save=False, l=1, lt=1, targstyle='-'):
    """
    plot voltage and ion concentrations, potentially compared to a target model
    Parameters
    -----------
    ts : array of dimension [time]
        time steps of the measurements
    i_inj : array of dimension [time]
    states : array of dimension [time, state_var, nb_neuron]
    y_states : array of dimension [time, state_var]
        values for the target model
    Returns
    -----------
    Nothing
    """
    plt.figure()
    nb_plots = len(model.ions_in_state) + 2

    if(states.ndim > 3):
        states = np.reshape(states, (states.shape[0], states.shape[1], -1))
        y_states = np.reshape(y_states, (y_states.shape[0], y_states.shape[1], -1))

    # Plot voltage
    plt.subplot(nb_plots, 1, 1)
    plt.plot(ts, states[:,model.V_pos], linewidth=l)
    if y_states is not None:
        if y_states[:,model.V_pos] is not None:
            plt.plot(ts, y_states[:,model.V_pos], 'r', linestyle=targstyle, linewidth=lt, label='target model')
    plt.ylabel('Voltage (mV)')
    plt.legend()

    for ion, pos in model.ions_in_state.items():
        plt.subplot(nb_plots, 1, 2)
        plt.plot(ts, states[:,pos], linewidth=l)
        if y_states is not None:
            if y_states[:, pos] is not None:
                plt.plot(ts, y_states[:, pos], 'r', linestyle=targstyle, linewidth=lt, label='target model')
        plt.ylabel('[{}]'.format(ion))
        plt.legend()

    plt.subplot(nb_plots, 1, nb_plots)
    plt.plot(ts, i_inj, 'b')
    plt.xlabel('t (ms)')
    plt.ylabel('$I_{inj}$ ($\\mu{A}/cm^2$)')

    if save:
        plt.savefig('{}{}output_{}.png'.format(current_dir, IMG_DIR, suffix))
    if show:
        plt.show()
    plt.close()

def plots_ica_from_v(ts, V, results, suffix="", show=True, save=False):
    """plot i_ca and Ca conc depending on the voltage"""
    ica = results[:, 0]
    e = results[:, 1]
    f = results[:, 2]
    h = results[:, 3]
    cac = results[:, -1]

    plt.figure()

    plt.subplot(4, 1, 1)
    plt.title('Hodgkin-Huxley Neuron : I_ca from a fixed V')
    plt.plot(ts, ica, 'b')
    plt.ylabel('I_ca')

    plt.subplot(4, 1, 2)
    plt.plot(ts, cac, 'r')
    plt.ylabel('$Ca^{2+}$ concentration')

    plt.subplot(4, 1, 3)
    plt.plot(ts, e, RATE_COLORS['e'], label='e')
    plt.plot(ts, f, RATE_COLORS['f'], label='f')
    plt.plot(ts, h, RATE_COLORS['h'], label='h')
    plt.ylabel('Gating Value')
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(ts, V, 'k')
    plt.ylabel('V (input) (mV)')
    plt.xlabel('t (ms)')

    if save:
        plt.savefig('{}results_ica_{}.png'.format(current_dir, suffix))
    if show:
        plt.show()
    plt.close()

def plots_ik_from_v(ts, V, results, suffix="", show=True, save=False):
    """plot i_k depending on the voltage"""
    ik = results[:, 0]
    p = results[:, 1]
    q = results[:, 2]
    n = results[:, 3]

    plt.figure()

    plt.subplot(3, 1, 1)
    plt.title('Hodgkin-Huxley Neuron : I_ca from a fixed V')
    plt.plot(ts, ik, 'b')
    plt.ylabel('I_k')

    plt.subplot(3, 1, 2)
    plt.plot(ts, p, RATE_COLORS['p'], label='p')
    plt.plot(ts, q, RATE_COLORS['q'], label='q')
    plt.plot(ts, n, RATE_COLORS['n'], label='n')
    plt.ylabel('Gating Value')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(ts, V, 'k')
    plt.ylabel('V (input) (mV)')
    plt.xlabel('t (ms)')

    if (save):
        plt.savefig('{}results_ik_{}.png'.format(current_dir, suffix))
    if (show):
        plt.show()
    plt.close()

