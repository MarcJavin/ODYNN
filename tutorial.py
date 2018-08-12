from odynn import optim, utils
import pandas as pd
import seaborn as sns
import pylab as plt
import numpy as np
from odynn.models import cfg_model
from odynn import neuron as nr
from odynn import nsimul as ns
from sklearn.decomposition import PCA



def corr(df):

    corr = df.corr()

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, cmap=cmap, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    plt.show()

def scatt(df):

    f, ax = plt.subplots(figsize=(6.5, 6.5))
    sns.despine(f, left=True, bottom=True)
    sns.scatterplot(x="loss", y="n__tau",
                    hue="rho_ca",
                    palette="autumn", linewidth=0,
                    data=df, ax=ax)
    plt.show()

def violin(df):
    # Use cubehelix to get a custom sequential palette
    # pal = sns.cubehelix_palette(p, rot=-.5, dark=.3)

    # Show each distribution with both violins and points
    sns.violinplot(data=df, inner="points")
    plt.show()

def get_df(dir):
    dic = optim.get_vars(dir)
    return pd.DataFrame.from_dict(dic)

def real_std(df):
    df = df.copy()
    mdps = [col for col in df.columns if 'mdp' in col or 'E' in col]
    df = df.drop(columns=mdps)
    variation = df.std() / df.mean()
    d = {'Variation': abs(variation.values),
         'Parameter': df.columns.values}
    df2 = pd.DataFrame(d)
    df2 = df2.sort_values(['Variation']).reset_index(drop=True)
    mx = np.max(d['Variation'])
    r = np.array([1., 0., 0.])
    g = np.array([0., 1., 0.])
    colors = [r * (1. - v / mx) + g * (v / mx) for v in df2['Variation']]
    df2.plot.bar(x='Parameter', y='Variation', colors=colors, title='Relative standard deviation')
    # ax = sns.barplot(x='Parameter', y='Variation', data=df2, palette=colors)
    # plt.yscale('log')
    plt.show()

def sigm():
    def plot_sigm(pts, scale, col='k'):
        plt.plot(pts, 1 / (1 + sp.exp((-30. - pts) / scale)), col, label='scale=%s'%scale)
    import scipy as sp
    pts = sp.arange(-12000, 20, 0.5)
    # plot_sigm(pts, -1, col='#000000')
    # plot_sigm(pts, -3, col='#440000')
    # plot_sigm(pts, -10, col='#880000')
    # plot_sigm(pts, -30, col='#bb0000')
    # plot_sigm(pts, -100, col='#ff0000')
    plot_sigm(pts, 1, col='#000000')
    plot_sigm(pts, 3, col='#004400')
    plot_sigm(pts, 10, col='#008800')
    plot_sigm(pts, 30, col='#00bb00')
    plot_sigm(pts, 1000, col='#00ff00')
    plt.legend()
    plt.title('Influence of $V_{scale}$ on the rate dynamics')
    plt.show()
    exit(0)

def table():
    import re
    neur = cfg_model.NEURON_MODEL
    from odynn.models import celeg
    dir = utils.set_dir('Integcomp_both_mod1noiNtau')
    best = optim.get_best_result(dir)
    for k, v in neur.default_params.items():
        v = neur._constraints_dic.get(k, ['-inf', 'inf'])
        u = ''
        if 'tau' in k:
            u = 'ms'
        elif 'scale' in k or 'mdp' in k or 'E' in k:
            u = 'mV'
        elif 'g' in k:
            u = 'mS/cm$^2$'
        elif k == 'C_m':
            u = '$\mu$F/cm$^2$'
        else:
            u = 'none'
        tp = '%s &&& %s & %s&%s&%s&%s \\\\\n \\hline' % (k, v[0], v[1], u, cfg_model.NEURON_MODEL.default_params[k], best[k])
        tp = re.sub('(.)__(.*) (&.*&.*&.*&.*&)', '\g<2>_\g<1> \g<3>', tp)
        tp = tp.replace('inf', '$\\infty$')

        tp = re.sub('scale_(.)', '$V_{scale}^\g<1>$', tp)
        tp = re.sub('mdp_(.)', '$V_{mdp}^\g<1>$', tp)
        tp = re.sub('tau_(.)', '$\\ tau^\g<1>$', tp)
        tp = re.sub('E_(..?)', '$E_{\g<1>}$', tp)
        tp = tp.replace('\\ tau', '\\tau')
        tp = re.sub('g_([^ ]*) +', '$g_{\g<1>}$ ', tp)
        tp = tp.replace('rho_ca', '$\\rho_{Ca}$')
        tp = tp.replace('decay_ca', '$\\tau_{Ca}$')
        tp = tp.replace('C_m', '$C_m$')
        tp = tp.replace('alpha_h', '$\\alpha^h$')

        tp = re.sub('(.*tau.*)&&&', '\g<1>&%s&%s&' % (celeg.MIN_TAU, celeg.MAX_TAU), tp)
        tp = re.sub('(.*scale.*)&&&', '\g<1>&%s&%s&' % (celeg.MIN_SCALE, celeg.MAX_SCALE), tp)
        print(tp)
    exit(0)

def hhsimp_box(df):
    utils.box(df, ['b', 'g', 'm', 'g', 'm'], ['C_m', 'g_L', 'g_K', 'E_L', 'E_K'])
    plt.title('Membrane')
    utils.save_show(True, True, 'boxmemb', dpi=300)
    plt.subplot(3, 1, 1)
    utils.box(df, ['m', '#610395'], ['a__mdp', 'b__mdp'])
    plt.title('Midpoint')
    plt.subplot(3, 1, 2)
    utils.box(df, ['m', '#610395'], ['a__scale', 'b__scale'])
    plt.title('Scale')
    plt.subplot(3, 1, 3)
    utils.box(df, ['m', '#610395'], ['a__tau', 'b__tau'])
    plt.yscale('log')
    plt.title('Time constant')
    plt.tight_layout()
    utils.save_show(True, True, 'boxrates', dpi=300)

if __name__ == '__main__':


    dir = utils.set_dir('Integcomp_volt_hhsimpnoise')

    # dic = optim.get_vars(dir, loss=False)
    # df = pd.DataFrame.from_dict(dic)
    # dfdisp = (df - df.mean()) / df.std()
    # plt.plot(dfdisp.transpose())
    # utils.save_show(True, True, 'disp', dpi=300)


    dic = optim.get_vars(dir, loss=True)

    train, test = optim.get_data(dir)
    df = pd.DataFrame.from_dict(dic)#.head(4)
    corr(df)
    exit(0)
    df = df.sort_values('loss').reset_index(drop=True)
    # df = df.dropna()
    sns.barplot(x=df.index, y='loss', data=df)
    # df.plot.bar(y='loss')
    utils.save_show(True, True, 'lossfin_virt', dpi=300)
    # df = df[df['loss'] <= np.min(df['loss'] + 0.2)]

    corr(df)
    cfg_model.NEURON_MODEL.boxplot_vars(dic, show=True, save=True)

    # dic = collections.OrderedDict(sorted(dic.items(), key=lambda t: t[0]))
    # obj = circuit.CircuitTf.create_random(n_neuron=9, syn_keys={(i,i+1):True for i in range(8)}, gap_keys={}, n_rand=50, dt=0.1)
    p = optim.get_best_result(dir)

    for i in range(train[1].shape[-1]):
        ns.comp_pars_targ(p, cfg_model.NEURON_MODEL.default_params, dt=train[0][1] - train[0][0], i_inj=train[1][:,i], suffix='virtrain%s'%i, show=True, save=True)
    for i in range(test[1].shape[-1]):
        ns.comp_pars_targ(p, cfg_model.NEURON_MODEL.default_params, dt=test[0][1] - test[0][0], i_inj=test[1][:,i], suffix='virtest%s'%i, show=True, save=True)


    # for i in range(X.shape[2]):
    #     plt.subplot(2, 1, 1)
    #     plt.plot(train[-1][-1], 'r', label='train data')
    #     plt.plot(X[:, -1, i])
    #     plt.legend()
    #     plt.subplot(2, 1, 2)
    #     plt.plot(test[-1][-1], 'r', label='test data')
    #     plt.plot(Xt[:, -1, i])
    #     plt.legend()
    #     plt.show()
    #     utils.save_show(True,True,'best_result%s'%i, dpi=250)
    # for i in range(9):
    #     dicn = {k: v[:,i] for k,v in dic.items()}
    #     hhmodel.CElegansNeuron.plot_vars(dicn, show=True, save=False)



    # scatt(df)

    # pca = PCA()
    # pca.fit(df)
    # for c in pca.components_:
    #     for i, name in enumerate(df):
    #         print(name, '%.2f'%c[i])
    # plt.plot(pca.explained_variance_ratio_)
    # plt.show()


    # sns.FacetGrid(data=df, row='C_m')
    # plt.show()
    # violin(df)