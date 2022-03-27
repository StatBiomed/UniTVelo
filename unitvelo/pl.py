#%%
import matplotlib.pyplot as plt
import scvelo as scv
from IPython import display
import seaborn as sns
import numpy as np
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
    fig.savefig(f'./figures/Gene_Filter_ByZero.png', dpi=300)
    from .pl import plot_zero_gene_distribution
    plot_zero_gene_distribution(nonzero_para, zero_para)

def plot_zero_gene_distribution(nonzero_para, zero_para):
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
    fig.savefig(f'./figures/Gene_Filter_ByZero_Distribution.png', dpi=300)

def plot_range(
    gene_name, 
    adata, 
    config_file=None, 
    save_fig=False, 
    show_ax=False,
    show_legend=False,
    show_details=False,
    time_metric='latent_time',
    show_temporal=False,
    show_positive=False,
    t_left=None,
    t_right=None
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
        scv.tl.latent_time(adata, min_likelihood=None)
        from .utils import min_max
        adata.obs['latent_time'] = min_max(adata.obs['latent_time'].values)

    if show_details:
        from .individual_gene import exam_genes        
        exam_genes(adata, gene_name, time_metric=time_metric)

    else:
        from .optimize_utils import exp_args, Model_Utils
        f = Validation(adata, time_metric=time_metric)
        f.init_data(adata)
        f.concat_data()
        f.vars_trends(gene_name, adata)

        columns = exp_args(adata)
        for col in columns:
            f.para[col] = np.log(f.para[col])

        boundary = (f.para['t0'] - 3 * (1 / np.sqrt(2 * np.exp(f.para['a0']))), 
                    f.para['t0'] + 3 * (1 / np.sqrt(2 * np.exp(f.para['a0']))))
        
        validate = Model_Utils(config=config_file)
        spre, upre = f.func(validate, validate.init_time(boundary, (3000, 1)))
        sone, uone = f.func(validate, validate.init_time((0, 1), (3000, 1)))

        fig, ax = plt.subplots()
        if not show_ax:
            ax.axis("off")

        g = sns.scatterplot(np.squeeze(adata[:, gene_name].layers['Ms']), 
                            np.squeeze(adata[:, gene_name].layers['Mu_scale']), 
                            s=20, hue=adata.obs[adata.uns['label']], 
                            palette=f.palette)
                            
        plt.plot(np.squeeze(spre), np.squeeze(upre), color='lightgrey', linewidth=2)
        plt.plot(np.squeeze(sone), np.squeeze(uone), color='black', linewidth=2)
        plt.xlim([
            -0.05 if adata[:, gene_name].layers['Ms'].min() < 1
                else adata[:, gene_name].layers['Ms'].min() * 0.95, 
            adata[:, gene_name].layers['Ms'].max() * 1.05])
        plt.ylim([
            -0.05 if adata[:, gene_name].layers['Mu_scale'].min() < 1
                else adata[:, gene_name].layers['Mu_scale'].min() * 0.95, 
            adata[:, gene_name].layers['Mu_scale'].max() * 1.05])

        if not show_legend:
            g.get_legend().remove()

        plt.xlabel('Spliced')
        plt.ylabel('Unspliced')
        # plt.title(gene_name, fontsize=12)
        plt.show()

        if save_fig:
            plt.savefig(f'./figures/GM_{gene_name}.png', dpi=300, bbox_inches='tight')
        
        if show_temporal:
            f.vars_trends(gene_name, adata)
            fit_left = f.fs[f'{gene_name}_fits'][np.argwhere(f.lt == 0)[0][0]]
            fit_right = f.fs[f'{gene_name}_fits'][np.argwhere(f.lt == 1)[0][0]]
            
            def get_reversed_data(fit_left, fit_right, t_left=None, t_right=None):
                t_left = np.sqrt(np.log((fit_left - f.para.offset0) / f.para.h0) / (-f.para.a0)) + f.para.t0 \
                    if t_left == None else t_left
                t_right = np.sqrt(np.log((fit_right - f.para.offset0) / f.para.h0) / (-f.para.a0)) + f.para.t0 \
                    if t_right == None else t_right
                print (t_left, t_right, f.para.t0)

                if f.para.t0 > 0 and f.para.t0 <= 0.5 and t_left == None:
                        t_left = -np.sqrt(np.log((fit_left - f.para.offset0) / f.para.h0) / (-f.para.a0)) + f.para.t0

                t_left = 0 if np.isnan(t_left) else t_left
                t_right = 1 if np.isnan(t_right) else t_right
                print (t_left, t_right, f.para.t0)

                t_left = max(t_left, 0) if t_left <= 1 else 0
                t_right = min(t_right, 1)
                print (t_left, t_right, f.para.t0)
                
                t = np.linspace(t_left, t_right, 1000)
                s = f.para.h0 * np.exp(-f.para.a0 * ((t - f.para.t0) ** 2)) + f.para.offset0
                sde = s * (-2 * f.para.a0 * (t - f.para.t0))
                u = (sde + f.para.gamma * s) / f.para.beta + f.para.intercept
                t_reverse = (t - t.min()) / (t.max() - t.min())

                return s, u, t_reverse, t_left, t_right
            
            s, u, t_re, t_left, t_right = get_reversed_data(fit_left, fit_right, t_left, t_right)
            
            if show_positive:
                s = np.where(s < 0, 0, s)
                u = np.where(u < 0, 0, u)

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            if not show_ax:
                axes[0].axis("off")
                axes[1].axis("off")

            g1 = sns.scatterplot(np.squeeze(adata.obs[time_metric].values), 
                                np.squeeze(adata[:, gene_name].layers['Ms']), 
                                s=20, hue=adata.obs[adata.uns['label']], 
                                palette=f.palette, ax=axes[0])
            sns.lineplot(t_re, s, color='black', linewidth=2, ax=axes[0])

            g2 = sns.scatterplot(np.squeeze(adata.obs[time_metric].values), 
                                np.squeeze(adata[:, gene_name].layers['Mu_scale']), 
                                s=20, hue=adata.obs[adata.uns['label']], 
                                palette=f.palette, ax=axes[1])
            sns.lineplot(t_re, u, color='black', linewidth=2, ax=axes[1])

            # if time_metric == 'latent_time':
            #     axes[0].xlim([np.round(t_left, 1), np.round(t_right, 1)])
            #     axes[1].xlim([np.round(t_left, 1), np.round(t_right, 1)])

            if not show_legend:
                g1.get_legend().remove()
                g2.get_legend().remove()

            plt.show()
            if save_fig:
                plt.savefig(f'./figures/GM_{gene_name}_temporal.png', dpi=300, bbox_inches='tight')

def plot_phase_portrait(adata, args, sobs, uobs, spre, upre):
    if adata.uns['examine_genes'] != False:
        display.clear_output(wait=True)
        examine = Validation(adata)
        fig, axes = plt.subplots(1, 1, figsize=(6, 4))

        examine.plot_mf(adata, sobs, uobs, axes[0])
        examine.plot_mf(adata, spre, upre, axes[0])

        if adata.uns['base_function'] == 'Piecewise':
            u = np.linspace(0, np.max(np.squeeze(uobs)), 200)
            s = np.squeeze(np.exp(args[1]) * u / np.exp(args[0]))
            sns.scatterplot(x=s, y=u, s=10, color='black', ax=axes[0])

        plt.show()
    
    else:
        pass

def plot_cell_time(adata):
    if adata.uns['examine_genes'] != False:
        raise ValueError(
            f'self.EXAMINE_GENE in configuration file should be False.\n'
            f'Please re-run the model use alternative setting.'
        )
    
    else:
        scv.pl.scatter(
            adata, color='latent_time', cmap='gnuplot', 
            size=25, title='Assigned cell time', dpi=300
        )

def plot_loss(iter, loss):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    x = range(iter + 1)
    subiter, subloss = x[800:], loss[800:]
    axes[0].plot(subiter, subloss)
    axes[0].set_title('Iter # from 800')
    axes[0].set_ylabel('Euclidean Loss')

    subiter, subloss = x[int(iter / 2):], loss[int(iter / 2):]
    axes[1].plot(subiter, subloss)
    axes[1].set_title('Iter # from 1/2 of maximum')

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