from odynn import optim, utils
import pandas as pd
import seaborn as sns
import pylab as plt
import numpy as np
from odynn.models import cfg_model
from odynn import neuron as nr
from odynn import nsimul as ns



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

if __name__ == '__main__':

    plt.subplot(5,1,1)
    plt.plot([1,2])
    plt.subplot(5, 1, 2)
    plt.plot([1, 2])
    plt.show()

    dir = utils.set_dir('Tuto')
    t = np.arange(0., 1200., 0.1)
    i = 10. * ((t > 200) & (t < 600)) + 20. * ((t > 800) & (t < 1000))
    ns.simul(dt=0.1, i_inj=i, show=True)
    exit(0)


    plt.plot([1, 4])
    plt.show()
    plt.close()

    dir = utils.set_dir('Integcomp_both_500rate-YAY')
    dic = optim.get_vars(dir, loss=True)
    df = pd.DataFrame.from_dict(dic).head(4)
    # df = df.dropna()
    print(df)

    scatt(df)
    # dir = utils.set_dir('Integcomp_both_500-YE')
    # dic2 = optim.get_vars(dir)
    # df = pd.DataFrame.from_dict(dic2)
    # df.merge(df1)
    # dic = collections.OrderedDict(sorted(dic.items(), key=lambda t: t[0]))
    # obj = circuit.CircuitTf.create_random(n_neuron=9, syn_keys={(i,i+1):True for i in range(8)}, gap_keys={}, n_rand=50, dt=0.1)
    # cfg_model.NEURON_MODEL.study_vars(dic, show=True, save=False)
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