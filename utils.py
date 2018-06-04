import socket
import matplotlib
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter
if (socket.gethostname()=='1080'):
    matplotlib.use("Agg")
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

COLS_MULT = ['r', 'g', 'b', 'y', 'k']

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
def get_dic_from_var(dir, sufix=""):
    file = '%s%s/%s_%s.txt' % (RES_DIR, dir, OUT_PARAMS, sufix)
    dic = {}
    with open(file, 'r') as f:
        for line in f:
            m = re.search(REGEX_VARS, line)
            dic[m.group(1)] = float(m.group(2))
    return dic




def get_data(file='AVAL_test.csv'):
    df = pd.read_csv(file)
    Y = np.array(df['trace'])
    X = np.array(df['inputCurrent']) * 10 + np.full(Y.shape, 0.001)
    T = np.array(df['timeVector']) * 1000
    return T, X, Y

"""plot variation of all variables organized by categories"""
def plot_vars(var_dic, lim, suffix="", show=True, save=False):
    fig = plt.figure()
    grid = plt.GridSpec(2, 3)
    for nb in range(len(GATES)):
        gate = GATES[nb]
        plot_vars_gate(gate, var_dic['%s__mdp' % gate][:lim + 1], var_dic['%s__scale' % gate][:lim + 1],
                       var_dic['%s__tau' % gate][:lim + 1], fig, grid[nb], (nb%3==0))
    plot_vars_gate('h', var_dic['h__mdp'][:lim + 1], var_dic['h__scale'][:lim + 1],
                   var_dic['h__alpha'][:lim + 1], fig, grid[5], False)
    plt.tight_layout()
    if(save):
        plt.savefig('%svar_%s_%s.png' % (DIR, 'Rates', suffix), dpi=300)
    if(show):
        plt.show()

    fig = plt.figure()
    grid = plt.GridSpec(1, 2)
    subgrid = gridspec.GridSpecFromSubplotSpec(4, 1, grid[0], hspace=0.1)
    ax = plt.Subplot(fig, subgrid[0])
    ax.plot(var_dic['g_Ks'][:lim + 1], RATE_COLORS['n'])
    ax.set_ylabel('KS cond.')
    ax.set_title('Conductances')
    fig.add_subplot(ax)
    ax = plt.Subplot(fig, subgrid[1])
    ax.plot(var_dic['g_Kf'][:lim + 1], RATE_COLORS['n'])
    ax.set_ylabel('KF cond.')
    fig.add_subplot(ax)
    ax = plt.Subplot(fig, subgrid[2])
    ax.plot(var_dic['g_Ca'][:lim + 1], RATE_COLORS['e'])
    ax.set_ylabel('Ca cond.')
    fig.add_subplot(ax)
    ax = plt.Subplot(fig, subgrid[3])
    ax.plot(var_dic['g_L'][:lim + 1], 'k')
    ax.set_ylabel('Leak cond.')
    fig.add_subplot(ax)

    subgrid = gridspec.GridSpecFromSubplotSpec(4, 1, grid[1], hspace=0.1)
    ax = plt.Subplot(fig, subgrid[0])
    ax.plot(var_dic['C_m'][:lim + 1])
    ax.set_ylabel('Capacitance')
    ax.yaxis.tick_right()
    ax.set_title('Membrane')
    fig.add_subplot(ax)
    ax = plt.Subplot(fig, subgrid[1])
    ax.plot(var_dic['E_K'][:lim + 1], RATE_COLORS['n'])
    ax.set_ylabel('K E_rev')
    ax.yaxis.tick_right()
    fig.add_subplot(ax)
    ax = plt.Subplot(fig, subgrid[2])
    ax.plot(var_dic['E_Ca'][:lim + 1], RATE_COLORS['f'])
    ax.set_ylabel('Ca E_rev')
    ax.yaxis.tick_right()
    fig.add_subplot(ax)
    ax = plt.Subplot(fig, subgrid[3])
    ax.plot(var_dic['E_L'][:lim + 1], 'k')
    ax.set_ylabel('Leak E_rev')
    ax.yaxis.tick_right()
    fig.add_subplot(ax)
    plt.tight_layout()
    if(save):
        plt.savefig('%svar_%s_%s.png' % (DIR, 'Membrane', suffix), dpi=300)
    if(show):
        plt.show()

    plt.close('all')


def plot_vars_gate(name, mdp, scale, tau, fig, pos, labs):
    subgrid = gridspec.GridSpecFromSubplotSpec(3,1,pos, hspace=0.1)
    ax = plt.Subplot(fig, subgrid[0])
    ax.plot(mdp, 'r')
    ax.set_xlabel([])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    if(labs):
        ax.set_ylabel('Midpoint')
    ax.set_title(name)
    fig.add_subplot(ax)
    ax = plt.Subplot(fig, subgrid[1])
    ax.plot(scale, 'g')
    ax.set_xlabel([])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    if (labs):
        ax.set_ylabel('Scale')
    fig.add_subplot(ax)
    ax = plt.Subplot(fig, subgrid[2])
    ax.plot(tau)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    if (labs):
        ax.set_ylabel('Tau')
    fig.add_subplot(ax)



def plot_loss_rate(losses, rates, lim, suffix="", show=True, save=False):
    plt.figure()

    plt.subplot(2,1,1)
    plt.plot(losses[:lim+1], 'r')
    plt.ylabel('Loss')

    plt.subplot(2,1,2)
    plt.plot(rates[:lim+1])
    plt.ylabel('Learning rate')

    if(save):
        plt.savefig('%slosses_%s.png' % (DIR,suffix))
    if(show):
        plt.show()
    plt.close()


def plots_output_mult(ts, i_inj, Vs, Cacs, suffix="", show=True, save=False):
    plt.figure()

    plt.subplot(3, 1, 1)
    for idx in range(len(Vs)):
        print(len(ts), len(Vs[idx]))
        plt.plot(ts, Vs[idx], COLS_MULT[idx])
    plt.ylabel('Voltage (mV)')

    plt.subplot(3, 1, 2)
    for idx in range(len(Cacs)):
        plt.plot(ts, Cacs[idx], COLS_MULT[idx])
    plt.ylabel('[Ca2+]')

    plt.subplot(3, 1, 3)
    plt.plot(ts, i_inj, 'k')
    plt.xlabel('t (ms)')
    plt.ylabel('$I_{inj}$ ($\\mu{A}/cm^2$)')

    if (save):
        plt.savefig('%s%soutput_%s.png' % (DIR, IMG_DIR, suffix))
    if (show):
        plt.show()
    plt.close()


def plots_output_double(ts, i_inj, v, y_v, cac, y_cac, suffix="", show=True, save=False):
    plt.figure()

    plt.subplot(3, 1, 2)
    plt.plot(ts, y_cac, 'g', label='target model')
    plt.plot(ts, cac, 'r', label='current model')
    plt.ylabel('[Ca2+]')
    plt.legend()

    plt.subplot(3, 1, 1)
    plt.plot(ts, y_v, 'g', label='target model')
    plt.plot(ts, v, 'r', label='current model')
    plt.ylabel('Voltage (mV)')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(ts, i_inj, 'k')
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
    plt.ylabel('Ca2+ concentration')

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
    plt.plot(ts, V, 'k')
    plt.ylabel('V (mV)')

    plt.subplot(5, 1, 2)
    plt.plot(ts, cac, 'r')
    plt.ylabel('Ca2+ concentration')

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
    plt.plot(ts, i_inj_values, 'k')
    plt.xlabel('t (ms)')
    plt.ylabel('$I_{inj}$ ($\\mu{A}/cm^2$)')
    # plt.ylim(-1, 40)

    if(save):
        plt.savefig('%sresults_%s.png' % (DIR,suffix))
    if(show):
        plt.show()
    plt.close()