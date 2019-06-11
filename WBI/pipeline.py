import sys
import data
import pylab as plt
import pandas as pd
import networkx as nx
import numpy as np
import torch
sys.path.insert(0, '..')
from odynn.circuit import Circuit
from odynn.models import LeakyIntegrate, ChemSyn, GapJunction
import os
from tqdm import tqdm
import seaborn as sns
from odynn import optim
import pickle
from scipy import stats
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import multiprocessing
from multiprocessing import Process


"""Constants"""
N_CPU = multiprocessing.cpu_count()
N_THREADS = 4
torch.set_num_threads(N_THREADS)

T_BASE = 2000
N_PARALLEL = 1000


DATASETS = data.get_all_data()
ALL_NAMES = data.get_all_names()
print('All identified neurons :\n', ALL_NAMES)


def optim_neuron(neuron0='RID', n_epochs=501):

    dirName = neuron0
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        os.mkdir(dirName+'/imgs')
        print("Directory ", dirName, " Created ")
    else:
        print("Directory ", dirName, " already exists")

    '''Get presynaptic neurons'''
    dfs = data.get_synapses()
    dfg = data.get_gaps()

    gaps = dfg.index[dfg[neuron0] > 0].tolist()
    syns = dfs.index[dfs[neuron0] > 0].tolist()
    inputs = np.unique(gaps + syns).tolist()
    print('Gaps :', gaps)
    print('Synapses :', syns)
    name_to_nb = {name: nb for nb, name in enumerate(inputs)}
    print(name_to_nb)
    with open(dirName + '/conns.txt', 'w') as f:
        f.write('Gaps : ' + str(gaps) + '\nSynapses : ' + str(syns) + '\n' + 'name to nb : ' + str(name_to_nb))

    '''Plot graph'''
    G = nx.DiGraph()
    G.add_nodes_from(gaps + syns)
    pos = nx.circular_layout(G)
    pos[neuron0] = [0,0]
    G.add_edges_from([(g, neuron0) for g in gaps])
    G.add_edges_from([(g, neuron0) for g in syns])
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, [(g, neuron0) for g in gaps], edge_color='b')
    nx.draw_networkx_edges(G, pos, [(g, neuron0) for g in syns], edge_color='r')
    plt.axis('off')
    plt.savefig(dirName + '/graph_connectome')
    plt.close()

    '''Get datasets with neuron labelled'''
    recs = []
    max_len = 0
    sets = []
    print('Datasets with %s identified : ' % neuron0)
    for i, d in enumerate(DATASETS):
        if neuron0 in d.columns:
            recs.append(d)
            sets.append(i)
            if len(d) > max_len:
                max_len = len(d)
    with open(dirName + '/datasets.txt', 'w') as f:
        f.write(str(sets))

    '''Collect input traces and initialize dt'''
    neurons = inputs + [neuron0]
    traces = np.zeros((max_len, len(recs), len(neurons)))
    dt = torch.zeros(len(recs),1,1)

    plt.figure(figsize = (20,20))
    for i, rec in enumerate(recs):
        for inp in inputs:
            if inp not in rec.columns:
                rec[inp] = np.zeros(rec.shape[0])
        traces[:rec.shape[0],i] = rec[neurons]
        dt[i] = T_BASE / rec.shape[0]
        plt.subplot(len(recs),1,i+1)
        plt.plot(traces[:,i] + [1.1*j for j in range(len(neurons))])
        plt.yticks([1.1*j for j in range(len(neurons))], neurons)

    plt.savefig(dirName + '/traces')
    plt.close()
    with open(dirName + '/dt.txt', 'w') as f:
        f.write(str(dt.numpy()))
    print('dt : ', dt)

    '''Define connections'''
    conn_g = [(name_to_nb[g], len(inputs)) for g in gaps]
    conn_s = [(name_to_nb[s], len(inputs)) for s in syns]

    with open(dirName + '/conns_defined.txt', 'w') as f:
        f.write('Gaps : %s \n Synapses : %s' % (str(conn_g), str(conn_s)))

    '''Create circuit'''
    def get_circ(N_parallel = 500, dt=dt):
        n_par = LeakyIntegrate.get_default(len(inputs), N_parallel)
        pout = LeakyIntegrate.get_random(1, N_parallel)
        pn = {k: np.concatenate((n_par[k], pout[k]), 0) for k,v in pout.items()}
        n = LeakyIntegrate(init_p=pn, tensors=True, dt=dt)
        ps = ChemSyn.get_random(len(conn_s),N_parallel)
        ps['E'] = np.repeat([[-1],[0],[0],[1]], N_parallel, axis=-1) + np.random.rand(4,N_parallel) * 0.2
        s = ChemSyn([c[0] for c in conn_s], [c[1] for c in conn_s], 
                                  init_p=ps, tensors=True, dt=dt)
        pg = GapJunction.get_random(len(conn_g),N_parallel)
        g = GapJunction([c[0] for c in conn_g], [c[1] for c in conn_g], 
                                  init_p=pg, tensors=True, dt=dt)
        return Circuit(n, s, g)

    get_circ().plot(labels={i: n for i,n in enumerate(inputs + [neuron0])}, img_size=5, save=dirName + '/graph_defined')

    '''Correlation to initialize E'''
    correlations = np.zeros((len(syns),1))

    for r in range(len(recs)):
        for i in range(len(syns)):
            corr = np.corrcoef(traces[:,r,i], traces[:,r,-1])[0,1]
            if np.isnan(corr):
                corr = 0
            correlations[i] += corr
    correlations /= len(recs)
    sns.heatmap(correlations, vmin=-1, vmax=1, cmap='seismic')
    plt.savefig(dirName + '/correlations')
    plt.close()
    init_E = (correlations + 1) / 2

    '''Optimize'''
    target = torch.Tensor(traces[:, None, :, :, None])
    init = target[0]
    vmask = torch.zeros((1,1,target.shape[-2],1))
    vmask[:,:,-1] = 1
    vadd = target.clone()
    vadd[:,:,:,-1] = 0
    print(init.shape, target.shape)

    """Optimize out neuron"""
    circuit = get_circ(N_PARALLEL)
    circuit._synapses._param['E'] = torch.Tensor(np.repeat(init_E, N_PARALLEL, axis=-1))
    circuit._synapses._param['E'].requires_grad = True

    def load_param(name='params%s' % neuron0):
        with open(name, 'rb') as f:
            p = pickle.load(f)
        for sub in [circuit._neurons, circuit._synapses, circuit._gaps]:
            for n in sub._parameter_names:
                sub._param[n] = torch.Tensor(p[n])
                sub._param[n].requires_grad = True
    # load_param()

    ALIGN = [1.1*n for n in range(target.shape[-2])]
    def plots(y, traces, loss, it):
        for i in range(len(recs)):
            plt.figure(figsize=(15,15))
            best = loss.argmin()
            plt.subplot(211)
            plt.plot(traces[:,0,i,:-1,0].detach().numpy() + ALIGN[:-1], linewidth=1)
            plt.plot(2*traces[:,0,i,-1,0].detach().numpy() + ALIGN[-1], linewidth=1.2, color='r')
            plt.plot(2*y[:,0,i,-1,best].detach().numpy() + ALIGN[-1], linewidth=1.1, linestyle='--', color='k')
            plt.yticks(ALIGN, inputs+[neuron0])
            plt.subplot(212)
            best_cat = torch.cat( (traces[:,0,i,:,0],y[:,0,i,-1:,best]), dim=1 ).detach().numpy().T
            sns.heatmap(best_cat, cmap='jet', vmin=0, vmax=1)
            plt.yticks(ALIGN, inputs+[neuron0]+[neuron0])
            plt.savefig(dirName + '/imgs/result_%s_%s' % (it, i))
            plt.close()

    losses = []
    params = [v for v in circuit.parameters.values()]
    optimizer = torch.optim.Adam(params, lr=0.001)

    for t in tqdm(range(n_epochs)):
        plt.figure()
        y = circuit.calculate(torch.zeros(traces.shape[0]), init, vmask=vmask, vadd=vadd)

        loss = optim.loss_mse(y, target)

        losses.append(loss.detach().numpy())
        # Upgrade variables
        optimizer.zero_grad()
        loss.mean().backward()
        for v in circuit._neurons.parameters.values():
            v.grad.data[:-1].zero_()

        optimizer.step()

        circuit.apply_constraints()

        print(loss.mean().detach().numpy(), loss.min().detach().numpy())

        if t%10 == 0:
            plots(y, target, loss, t)
            if loss.min() <= losses[-1].min():
                with open(dirName + '/params', 'wb') as f:
                    p = {k: v.detach().numpy() for k,v in circuit.parameters.items()}
                    pickle.dump(p, f)
        with open(dirName + '/losses', 'wb') as f:
            pickle.dump(losses, f)

    plt.figure(figsize=(10,10))
    plt.plot([l for l in losses], linewidth=0.2)
    plt.yscale('log')
    plt.savefig(dirName + '/losses')
    plt.close()

    plot_params(circuit, losses, gaps, syns, dirName)

def plot_params(circuit, losses, gaps, syns, dirName):

    best = losses[-1].min()
    bests100 = []
    for n in range(N_PARALLEL):
        if losses[-1][n] < best + 100:
            bests100.append(n)

    def get_par(par):
        return circuit.parameters[par][:, bests100].detach().numpy()

    NEURS_SYN = syns
    NEURS_GAP = gaps
    neur_p = np.concatenate([get_par(p)[-1:] for p in circuit._neurons.parameter_names])
    neur_c = [p for p in circuit._neurons.parameter_names]

    gap_p = np.concatenate([get_par(p) for p in circuit._gaps.parameter_names])
    gap_c = ['%s_%s' % (p, n) for p in circuit._gaps.parameter_names for n in NEURS_GAP]

    syn_p = np.concatenate([get_par(p) for p in circuit._synapses.parameter_names])
    syn_c = ['%s_%s' % (p, n) for p in circuit._synapses.parameter_names for n in NEURS_SYN]

    res = pd.DataFrame(np.concatenate((neur_p, gap_p, syn_p)).T, columns=neur_c + gap_c + syn_c)

    cmap = cm.bwr
    norm = Normalize(vmin=-1, vmax=1)

    def corrfunc(x, y, **kws):
        r, _ = stats.pearsonr(x, y)
        ax = plt.gca()
        ax.set_facecolor(cmap(norm(r)))

    def corrscat(x, y, **kws):
        r, _ = stats.pearsonr(x, y)
        plt.scatter(x, y, color=cmap(norm(r)), linewidths=0.7, edgecolors='k')

    g = sns.PairGrid(res)
    g.fig.set_size_inches(30, 30)
    g.map_lower(corrfunc)
    g.map_lower(sns.kdeplot, cut=0, cmap='Greys', shade=True)
    g.map_upper(corrscat)
    g.map_diag(sns.distplot, kde=False, color='g')
    plt.savefig(dirName + '/paramscorr')

if __name__ == '__main__':
    n_process = N_CPU // N_THREADS
    for n in ALL_NAMES[1:]:
        optim_neuron(n)
