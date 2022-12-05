#%%
import matplotlib.pyplot as plt
import scvelo as scv
from IPython import display
import seaborn as sns
import numpy as np
import os
from .individual_gene import Validation

def plot_zero_gene_example(gene_name, adata, nonzero_para, zero_para):
    import math
    nrows = math.ceil(len(gene_name[:8]) / 2)

    fig, axes = plt.subplots(nrows, 2, figsize=(12, 5 * nrows))
    for i in range(nrows):
        sns.scatterplot(np.squeeze(adata[:, gene_name[i * 2]].layers['Ms']), 
                        np.squeeze(adata[:, gene_name[i * 2]].layers['Mu']), 
                        sizes=1, ax=axes[i][0], 
                        hue=adata.obs[adata.uns['label']], palette='tab20')
        axes[i][0].set_title(f'{gene_name[i * 2]}')

        nonzero_coef = np.round(nonzero_para.loc[gene_name[i * 2]]['coef'], 2)
        zero_coef = np.round(zero_para.loc[gene_name[i * 2]]['coef'], 2)
        nonzero_r2 = np.round(nonzero_para.loc[gene_name[i * 2]]['r2'], 2)
        zero_r2 = np.round(zero_para.loc[gene_name[i * 2]]['r2'], 2)

        axes[i][0].set_xlabel(f'({nonzero_coef}, {zero_coef})')
        axes[i][0].set_ylabel(f'({nonzero_r2}, {zero_r2})')
        
        sns.scatterplot(np.squeeze(adata[:, gene_name[i * 2 + 1]].layers['Ms']), 
                        np.squeeze(adata[:, gene_name[i * 2 + 1]].layers['Mu']), 
                        sizes=1, ax=axes[i][1], 
                        hue=adata.obs[adata.uns['label']], palette='tab20')
        axes[i][1].set_title(f'{gene_name[i * 2 + 1]}')

        nonzero_coef = np.round(nonzero_para.loc[gene_name[i * 2 + 1]]['coef'], 2)
        zero_coef = np.round(zero_para.loc[gene_name[i * 2 + 1]]['coef'], 2)
        nonzero_r2 = np.round(nonzero_para.loc[gene_name[i * 2 + 1]]['r2'], 2)
        zero_r2 = np.round(zero_para.loc[gene_name[i * 2 + 1]]['r2'], 2)

        axes[i][1].set_xlabel(f'({nonzero_coef}, {zero_coef})')
        axes[i][1].set_ylabel(f'({nonzero_r2}, {zero_r2})')

    plt.show()
    fig.savefig(os.path.join(adata.uns['temp'], 'Gene_Filter_ByZero.png'), dpi=300)
    from .pl import plot_zero_gene_distribution
    plot_zero_gene_distribution(nonzero_para, zero_para)

def plot_zero_gene_distribution(nonzero_para, zero_para, adata=None):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    sns.distplot(nonzero_para['coef'].values, ax=axes[0][0], kde=False)
    axes[0][0].set_title('Nonzero genes coefficient')

    sns.distplot(zero_para['coef'].values, ax=axes[0][1], kde=False)
    axes[0][1].set_title('Zero genes coefficient')

    sns.distplot(nonzero_para['r2'].values, ax=axes[1][0], kde=False)
    axes[1][0].set_title('Nonzero genes r2')

    sns.distplot(zero_para['r2'].values, ax=axes[1][1], kde=False)
    axes[1][1].set_title('Zero genes r2')
    plt.show()
    fig.savefig(os.path.join(adata.uns['temp'], 'Gene_Filter_ByZero_Distribution.png'), dpi=300)

def rbf(x, height, sigma, tau, offset_rbf):
    return height * np.exp(-sigma * (x - tau) * (x - tau)) + offset_rbf

def rbf_deri(x, height, sigma, tau, offset_rbf):
    return (rbf(x, height, sigma, tau, offset_rbf)  - offset_rbf) * (-sigma * 2 * (x - tau))

def rbf_u(x, height, sigma, tau, offset_rbf, beta, gamma, intercept):
    return (rbf_deri(x, height, sigma, tau, offset_rbf) + gamma * rbf(x, height, sigma, tau, offset_rbf)) / beta + intercept

def plot_range(
    gene_name, 
    adata, 
    config_file=None, 
    save_fig=False, 
    show_ax=False,
    show_legend=True,
    show_details=False,
    time_metric='latent_time',
    palette='tab20',
    size=20,
    ncols=1
):
    """
    Plotting function of phase portraits of individual genes.
    
    Args:
        gene_name (str): name of that gene to be illusrated, would extend to list of genes in next release
        adata (AnnData)
        config_file (.Config class): configuration file used for velocity estimation
        save_fig (bool): if True, save fig, default False
        
        show_ax (bool)
        show_legend (bool)
        show_details (bool): if True, plot detailed regression results together with estimated temporal change
        time_metric (str): inferred cell time, default 'latent_time'

        show_temporal (bool, experimental): whether plot temporal changes
            show_positive (bool, experimental): related to self.ASSIGN_POS_U
            t_left (float, experimental): starting time of phase portraits
            t_right (float, experimental): ending time of phase portraits
    """

    if config_file == None:
        raise ValueError('Please set attribute `config_file`')

    if time_metric == 'latent_time':
        if 'latent_time' not in adata.obs.columns:
            scv.tl.latent_time(adata, min_likelihood=None)

    if show_details:
        from .individual_gene import exam_genes        
        exam_genes(adata, gene_name, time_metric=time_metric)

    else:
        gene_name = gene_name if type(gene_name) == list else [gene_name]
        figs = []    

        for gn in gene_name:
            fig, axes = plt.subplots(
                    nrows=1,
                    ncols=3, 
                    figsize=(18, 4)
            )
            gdata = adata[:, gn]

            boundary = (gdata.var.fit_t.values - 3 * (1 / np.sqrt(2 * np.exp(gdata.var.fit_a.values))), 
                        gdata.var.fit_t.values + 3 * (1 / np.sqrt(2 * np.exp(gdata.var.fit_a.values))))
            
            t_one = np.linspace(0, 1, 1000)
            t_boundary = np.linspace(boundary[0], boundary[1], 2000)

            spre = np.squeeze(rbf(t_boundary, gdata.var.fit_h.values, gdata.var.fit_a.values, gdata.var.fit_t.values, gdata.var.fit_offset.values))
            sone = np.squeeze(rbf(t_one, gdata.var.fit_h.values, gdata.var.fit_a.values, gdata.var.fit_t.values, gdata.var.fit_offset.values))

            upre = np.squeeze(rbf_u(t_boundary, gdata.var.fit_h.values, gdata.var.fit_a.values, gdata.var.fit_t.values, gdata.var.fit_offset.values, gdata.var.fit_beta.values, gdata.var.fit_gamma.values, gdata.var.fit_intercept.values))
            uone = np.squeeze(rbf_u(t_one, gdata.var.fit_h.values, gdata.var.fit_a.values, gdata.var.fit_t.values, gdata.var.fit_offset.values, gdata.var.fit_beta.values, gdata.var.fit_gamma.values, gdata.var.fit_intercept.values))

            g1 = sns.scatterplot(x=np.squeeze(gdata.layers['Ms']), 
                                y=np.squeeze(gdata.layers['Mu']), 
                                s=size, hue=adata.obs[adata.uns['label']], 
                                palette=palette, ax=axes[0])
            axes[0].plot(spre, upre, color='lightgrey', linewidth=2, label='Predicted Curve')
            axes[0].plot(sone, uone, color='black', linewidth=2, label='Predicted Curve Time 0-1')
            axes[0].set_xlabel('Spliced Reads')
            axes[0].set_ylabel('Unspliced Reads')

            axes[0].set_xlim([-0.005 if gdata.layers['Ms'].min() < 1
                    else gdata.layers['Ms'].min() * 0.95, 
                    gdata.layers['Ms'].max() * 1.05])
            axes[0].set_ylim([-0.005 if gdata.layers['Mu'].min() < 1
                    else gdata.layers['Mu'].min() * 0.95, 
                    gdata.layers['Mu'].max() * 1.05])

            g2 = sns.scatterplot(x=np.squeeze(adata.obs[time_metric].values), 
                                y=np.squeeze(gdata.layers['Ms']), 
                                s=size, hue=adata.obs[adata.uns['label']], 
                                palette=palette, ax=axes[1])
            sns.lineplot(x=t_one, y=sone, color='black', linewidth=2, ax=axes[1])

            axes[1].set_xlabel('Inferred Cell Time')
            axes[1].set_ylabel('Spliced')

            g3 = sns.scatterplot(x=np.squeeze(adata.obs[time_metric].values), 
                                y=np.squeeze(gdata.layers['Mu']), 
                                s=size, hue=adata.obs[adata.uns['label']], 
                                palette=palette, ax=axes[2])
            sns.lineplot(x=t_one, y=uone, color='black', linewidth=2, ax=axes[2])

            axes[2].set_xlabel('Inferred Cell Time')
            axes[2].set_ylabel('Unspliced')

            # if not show_ax:
            #     axes.axis("off")

            if not show_legend:
                g1.get_legend().remove()
                g2.get_legend().remove()
                g3.get_legend().remove()

            axes[1].set_title(gn, fontsize=12)
            plt.show()

            if save_fig:
                plt.savefig(os.path.join(adata.uns['temp'], f'GM_{gn}.png'), dpi=300, bbox_inches='tight')

def plot_phase_portrait(adata, args, sobs, uobs, spre, upre):
    if 'examine_genes' in adata.uns.keys():
        display.clear_output(wait=True)
        examine = Validation(adata)
        fig, axes = plt.subplots(1, 1, figsize=(6, 4))

        examine.plot_mf(adata, sobs, uobs, axes[0])
        examine.plot_mf(adata, spre, upre, axes[0])
        plt.show()
    
    else:
        pass

def plot_cell_time(adata):
    if 'examine_genes' in adata.uns.keys():
        raise ValueError(
            f'self.VGENES in configuration file should not be a specified gene name.\n'
            f'Please re-run the model use alternative setting.'
        )
    
    else:
        scv.pl.scatter(
            adata, color='latent_time', cmap='gnuplot', 
            size=25, title='Assigned cell time', dpi=300
        )

def plot_loss(iter, loss, thres=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x = range(iter + 1)

    subiter, subloss = x[800:thres - 1], loss[800:thres - 1]
    axes[0].plot(subiter, subloss)
    axes[0].set_title('Iter # from 800 to cutoff')
    axes[0].set_ylabel('Euclidean Loss')

    # subiter, subloss = x[int(iter / 2):], loss[int(iter / 2):]
    # axes[1].plot(subiter, subloss)
    # axes[1].set_title('Iter # from 1/2 of maximum')

    subiter, subloss = x[int(thres * 1.01):], loss[int(thres * 1.01):]
    axes[1].plot(subiter, subloss)
    axes[1].set_title('Iter # from cutoff to terminated state')

    plt.show()
    plt.close()

def plot_compare_loss(adata):
    var = adata.var.loc[adata.var['velocity_genes'] == True]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    x, y = np.log(var['li_loss']), np.log(var['fit_loss'])
    sns.scatterplot(x, y, ax=axes[0])
    axes[0].plot([np.min(x), np.max(x)], [np.min(x), np.max(x)], linestyle='--', c='r')
    axes[0].set_title('Log Loss')

    x, y = var['li_loss'], var['fit_loss']
    sns.scatterplot(x, y, ax=axes[1])
    axes[1].plot([np.min(x), np.max(x)], [np.min(x), np.max(x)], linestyle='--', c='r')
    axes[1].set_title('Normal Loss')

    plt.show()
    plt.close()

    print (f'---> # of genes which linear regression loss is smaller: ', end='')
    print (var.loc[var['fit_loss'] >= var['li_loss']].shape[0])

def plot_compare_bic(adata):
    var = adata.var.loc[adata.var['velocity_genes'] == True]

    fig, axes = plt.subplots(1, 1, figsize=(6, 4))
    x, y = var['li_bic'], var['fit_bic']
    sns.scatterplot(x, y, ax=axes)
    axes.plot([np.min(x), np.max(x)], [np.min(x), np.max(x)], linestyle='--', c='r')
    axes.set_title('Normal BIC')

    plt.show()
    plt.close()

    print (f'---> # of genes which linear regression BIC is smaller: ', end='')
    print (var.loc[var['fit_bic'] >= var['li_bic']].shape[0])
    print (var.loc[var['fit_bic'] >= var['li_bic']].shape[0] / var.shape[0])

def plot_compare_llf(adata):
    var = adata.var.loc[adata.var['velocity_genes'] == True]

    fig, axes = plt.subplots(1, 1, figsize=(6, 4))
    ratio = -2 * (var['li_llf'] - var['fit_llf'])
    sns.distplot(ratio, ax=axes, bins=200, kde=True)
    axes.set_title('Likelihood Ratio')
    
    plt.show()
    plt.close()

def plot_reverse_tran_scatter(adata):
    sns.scatterplot(x='rbf_r2', y='qua_r2', 
        data=adata.var.loc[adata.var['velocity_genes'] == True])
    plt.axline((0, 0), (0.5, 0.5), color='r')
        
    plt.title(f'$R^2$ comparison of RBF and Quadratic model')
    plt.show()
    plt.close()