import socket
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter
if (socket.gethostname()=='1080'):
    mpl.use("Agg")
import pylab as plt
import numpy as np
import pandas as pd
import os
import re

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels

RES_DIR = 'results/'
IMG_DIR = 'img/'
DIR = RES_DIR
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

COLORS = ['k', 'c', 'Gold', 'Darkred', 'b', 'Orange', 'm', 'Lime', 'Salmon', 'Indigo', 'DarkGrey', 'Crimson', 'Olive']
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=COLORS)


if __name__ == '__main__':
    plt.figure()
    plt.bar(x=range(len(COLORS)), height=[1 for _ in range(len(COLORS))], color=COLORS)
    plt.show()

"""Set directory to save files"""
def set_dir(subdir):
    global DIR
    DIR = RES_DIR + subdir + '/'
    if not os.path.exists(DIR):
        os.makedirs(DIR)
    if not os.path.exists(DIR+IMG_DIR):
        os.makedirs(DIR+IMG_DIR)
        os.makedirs(DIR+'tmp')
    return DIR


"""Get variables values into a dictionnary"""
def get_dic_from_var(dir, suffix=""):
    file = '%s%s/%s_%s.txt' % (RES_DIR, dir, OUT_PARAMS, suffix)
    dic = {}
    with open(file, 'r') as f:
        for line in f:
            print('line', line)
            m = re.search(REGEX_VARS, line)
            print(m)
            print(m.group(2))
            if(m.group(2)[0]=='['):
                #several params
                l = re.findall('[-]?[\d]*[\.]?[\d]+[e]?[+-]?[\d]+', m.group(2))
                l = map(float, l)
                dic[m.group(1)] = l
            else:
                dic[m.group(1)] = float(m.group(2))
    return dic


def bar(ax, var):
    ax.bar(x=range(len(var)), height=var, color=COLORS)
def plot(ax, var):
    ax.plot(var, linewidth=0.5)
def boxplot(ax, var):
    ax.boxplot(var, vert=True, showmeans=True)

def box(var_dic, cols, labels):
    bp = plt.boxplot([var_dic[k] for k in labels], vert=True, patch_artist=True, showmeans=True, labels=labels)
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
        plt.savefig('%svar_%s_%s.png' % (DIR, 'Membrane', suffix), dpi=300)
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
        plt.savefig('%svar_%s_%s.png' % (DIR, 'CalciumPump', suffix), dpi=300)
    if (show):
        plt.show()

    plt.figure()
    for i, type in enumerate(['mdp', 'scale', 'tau']):
        plt.subplot(3,1,i+1)
        plt.title(type)
        labels = ['%s__%s'%(rate,type) for rate in RATE_COLORS.keys()]
        cols = RATE_COLORS.values()
        if(type=='tau'):
            labels[2] = 'h__alpha'
        box(var_dic, cols, labels)
    if (save):
        plt.savefig('%svar_%s_%s.png' % (DIR, 'Rates', suffix), dpi=300)
    if (show):
        plt.show()
    plt.close()

"""plot variation/comparison/boxplots of synaptic variables"""
def plot_vars_syn(var_dic, suffix="", show=True, save=False, func=plot):
    plt.figure()
    for i,var in enumerate(['G', 'mdp', 'E', 'scale']):
        plt.subplot(2,2,i+1)
        func(plt, var_dic[var])
        plt.ylabel(var)
    plt.tight_layout()
    if (save):
        plt.savefig('%svar_%s_%s.png' % (DIR, 'Synapses', suffix), dpi=300)
    if (show):
        plt.show()
    plt.close()



"""plot variation/comparison/boxplots of all variables organized by categories"""
def plot_vars(var_dic, suffix="", show=True, save=False, func=plot):
    fig = plt.figure()
    grid = plt.GridSpec(2, 3)
    for nb in range(len(GATES)):
        gate = GATES[nb]
        plot_vars_gate(gate, var_dic['%s__mdp' % gate], var_dic['%s__scale' % gate],
                       var_dic['%s__tau' % gate], fig, grid[nb], (nb%3==0), func=func)
    plot_vars_gate('h', var_dic['h__mdp'], var_dic['h__scale'],
                   var_dic['h__alpha'], fig, grid[5], False, func=func)
    plt.tight_layout()
    if(save):
        plt.savefig('%svar_%s_%s.png' % (DIR, 'Rates', suffix), dpi=300)
    if(show):
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
    if(save):
        plt.savefig('%svar_%s_%s.png' % (DIR, 'Membrane', suffix), dpi=300)
    if(show):
        plt.show()

    plt.figure()
    ax = plt.subplot(211)
    func(ax, var_dic['rho_ca'])#, 'r')
    plt.ylabel('Rho_ca')
    ax = plt.subplot(212)
    func(ax, var_dic['decay_ca'])#, 'b')
    plt.ylabel('Decay_ca')
    if (save):
        plt.savefig('%svar_%s_%s.png' % (DIR, 'CalciumPump', suffix), dpi=300)
    if (show):
        plt.show()

    plt.close('all')

"""plot the gates variables"""
def plot_vars_gate(name, mdp, scale, tau, fig, pos, labs, func=plot):
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


"""plot loss (log10) and learning rate"""
def plot_loss_rate(losses, rates, suffix="", show=True, save=False):
    plt.figure()

    plt.subplot(2,1,1)
    if(losses.ndim == 1):
        plt.plot(losses, 'r')
    else:
        plt.plot(losses, linewidth=0.8)
    plt.ylabel('Loss')
    plt.yscale('log')

    plt.subplot(2,1,2)
    plt.plot(rates)
    plt.ylabel('Learning rate')

    if(save):
        plt.savefig('%slosses_%s.png' % (DIR,suffix))
    if(show):
        plt.show()
    plt.close()

"""plot multiple voltages and Ca2+ concentration"""
def plots_output_mult(ts, i_inj, Vs, Cacs, i_syn=None, labels=None, suffix="", show=True, save=False):
    plt.figure()

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
        plt.savefig('%s%soutput_%s.png' % (DIR, IMG_DIR, suffix))
    if (show):
        plt.show()
    plt.close()

"""plot voltage and Ca2+ conc compared to the target model"""
def plots_output_double(ts, i_inj, v, y_v, cac, y_cac, suffix="", show=True, save=False, l=1, lt=1):
    plt.figure()

    plt.subplot(3, 1, 2)
    plt.plot(ts, cac, linewidth=l)
    plt.plot(ts, y_cac, 'r', linewidth=lt, label='target model')
    plt.ylabel('[$Ca^{2+}$]')
    plt.legend()

    plt.subplot(3, 1, 1)
    plt.plot(ts, v, linewidth=l)
    plt.plot(ts, y_v, 'r', linewidth=lt, label='target model')
    plt.ylabel('Voltage (mV)')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(ts, i_inj, 'b')
    plt.xlabel('t (ms)')
    plt.ylabel('$I_{inj}$ ($\\mu{A}/cm^2$)')

    if(save):
        plt.savefig('%s%soutput_%s.png' % (DIR,IMG_DIR,suffix))
    if(show):
        plt.show()
    plt.close()

"""plot i_ca and Ca conc depending on the voltage"""
def plots_ica_from_v(ts, V, results, suffix="", show=True, save=False):
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

    if(save):
        plt.savefig('%sresults_ica_%s.png' % (DIR, suffix))
    if(show):
        plt.show()
    plt.close()

"""plot i_k depending on the voltage"""
def plots_ik_from_v(ts, V, results, suffix="", show=True, save=False):
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
        plt.savefig('%sresults_ik_%s.png' % (DIR, suffix))
    if (show):
        plt.show()
    plt.close()


"""plot all dynamics"""
def plots_results(model, ts, i_inj_values, results, suffix="", show=True, save=False):
    V = results[:, 0]
    p = results[:, 1]
    q = results[:, 2]
    n = results[:, 3]
    e = results[:, 4]
    f = results[:, 5]
    cac = results[:, 6]

    h = model.h(cac)
    ica = model.I_Ca(V, e, f, h)
    ik = model.I_Ks(V, n) + model.I_Kf(V, p, q)
    il = model.I_L(V)

    plt.figure()

    plt.subplot(5, 1, 1)
    plt.title('Hodgkin-Huxley Neuron')
    if(V.ndim==1):
        plt.plot(ts, V, 'k')
    else:
        plt.plot(ts, V)
    plt.ylabel('V (mV)')

    plt.subplot(5, 1, 2)
    if (V.ndim == 1):
        plt.plot(ts, cac, 'r')
    else:
        plt.plot(ts, cac)
    plt.ylabel('$Ca^{2+}$ concentration')

    plt.subplot(5, 1, 3)
    plt.plot(ts, ica, 'c', label='$I_{Ca}$')
    plt.plot(ts, ik, 'y', label='$I_{K}$')
    plt.plot(ts, il, 'm', label='$I_{L}$')
    plt.ylabel('Current')
    plt.legend()

    plt.subplot(5, 1, 4)
    plt.plot(ts, p, RATE_COLORS['p'], label='p')
    plt.plot(ts, q, RATE_COLORS['q'], label='q')
    plt.plot(ts, n, RATE_COLORS['n'], label='n')
    plt.plot(ts, e, RATE_COLORS['e'], label='e')
    plt.plot(ts, f, RATE_COLORS['f'], label='f')
    plt.plot(ts, h, RATE_COLORS['h'], label='h')
    plt.ylabel('Gating Value')
    plt.legend()

    plt.subplot(5, 1, 5)
    plt.plot(ts, i_inj_values, 'b')
    plt.xlabel('t (ms)')
    plt.ylabel('$I_{inj}$ ($\\mu{A}/cm^2$)')
    # plt.ylim(-1, 40)

    if(save):
        plt.savefig('%sresults_%s.png' % (DIR,suffix))
    if(show):
        plt.show()
    plt.close()