from opthh import optim, utils
import pandas as pd
import seaborn as sns
import pylab as plt
import collections


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
    sns.scatterplot(x="loss", y="e__scale",
                    hue="g_Ca",
                    palette="autumn", linewidth=0,
                    data=df, ax=ax)
    plt.show()

def violin(df):
    # Use cubehelix to get a custom sequential palette
    # pal = sns.cubehelix_palette(p, rot=-.5, dark=.3)

    # Show each distribution with both violins and points
    sns.violinplot(data=df, inner="points")
    plt.show()

if __name__ == '__main__':
    dir = utils.set_dir('Integcomp_both_500rate-YAY')
    dic1 = optim.get_vars(dir)
    dir = utils.set_dir('Integcomp_both_500-YE')
    dic2 = optim.get_vars(dir)
    dic = {**dic1, **dic2}
    dic = collections.OrderedDict(sorted(dic.items(), key=lambda t: t[0]))
    from opthh import hhmodel
    # obj = circuit.CircuitTf.create_random(n_neuron=9, syn_keys={(i,i+1):True for i in range(8)}, gap_keys={}, n_rand=50, dt=0.1)
    # hhmodel.CElegansNeuron.study_vars(dic, show=True, save=False)
    # for i in range(9):
    #     dicn = {k: v[:,i] for k,v in dic.items()}
    #     hhmodel.CElegansNeuron.plot_vars(dicn, show=True, save=False)

    df = pd.DataFrame.from_dict(dic)
    scatt(df)
    # sns.FacetGrid(data=df, row='C_m')
    # plt.show()
    # violin(df)