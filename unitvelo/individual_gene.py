#%%
import os 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython import display

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    import re
    return [atoi(c) for c in re.split(r'(\d+)', text)]

class Validation():
    def __init__(self, adata, time_metric='latent_time') -> None:
        self.adata = adata
        self.time_metric = time_metric

        if 'latent_time' not in adata.obs.columns:
            import scvelo as scv
            scv.tl.latent_time(adata, min_likelihood=None)         

        if len(set(adata.obs[adata.uns['label']])) > 20:
            self.palette = 'viridis'
        else:
            self.palette = 'tab20'

    def init_data(self, adata):
        DIR = adata.uns['temp']
        f = os.listdir(DIR)
        f.sort(key=natural_keys)

        self.mu = pd.read_csv(f'{DIR}/Mu.csv', index_col=0).add_suffix('_u')
        self.ms = pd.read_csv(f'{DIR}/Ms.csv', index_col=0).add_suffix('_s')

        self.fs = pd.DataFrame(index=self.ms.index)
        self.fu = pd.DataFrame(index=self.ms.index)
        self.var = pd.DataFrame(index=[col[:-2] for col in self.ms.columns])

        for i in f:
            if i.startswith('fits'):
                self.fs = self.fs.join(pd.read_csv(f'{DIR}/{i}', index_col=0).add_suffix(f'_fits'))

            if i.startswith('fitu'):
                self.fu = self.fu.join(pd.read_csv(f'{DIR}/{i}', index_col=0).add_suffix(f'_fitu'))

            if i.startswith('fitvar'):
                df = pd.read_csv(f'{DIR}/{i}', index_col=0)
                df.index = df.index.map(str)
                self.var = self.var.join(df)

        self.lt = adata.obs[self.time_metric].values
        self.label = adata.obs[adata.uns['label']].values

    def concat_data(self):        
        self.msmu = self.ms.join(self.mu)
        self.msmu['labels'] = self.label

        self.msfs = self.ms.join(self.fs)
        self.msfs['labels'] = self.label

        self.mufu = self.mu.join(self.fu)
        self.mufu['labels'] = self.label

        self.fsfu = self.fs.join(self.fu)
        self.fsfu['labels'] = self.label

    def inspect_genes(self, gene_name, adata):
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        self.plot_mf(adata, self.msfs[gene_name + '_s'], self.msfs[gene_name + f'_fits'], axes[0][0])
        axes[0][0].plot(
            [np.min(self.fs[gene_name + f'_fits']), np.max(self.fs[gene_name + f'_fits'])], 
            [np.min(self.fs[gene_name + f'_fits']), np.max(self.fs[gene_name + f'_fits'])], 
            ls='--', c='red')
        axes[0][0].set_title('x: ms -- y: fits')

        self.plot_mf(adata, self.mufu[gene_name + '_u'], self.mufu[gene_name + f'_fitu'], axes[0][1])
        axes[0][1].plot(
            [np.min(self.fu[gene_name + f'_fitu']), np.max(self.fu[gene_name + f'_fitu'])], 
            [np.min(self.fu[gene_name + f'_fitu']), np.max(self.fu[gene_name + f'_fitu'])], 
            ls='--', c='red')
        axes[0][1].set_title('x: mu -- y: fitu')

        self.plot_mf(adata, self.msmu[gene_name + '_s'], self.msmu[gene_name + '_u'], axes[1][0])
        axes[1][0].set_title('x: ms -- y: mu')

        self.plot_mf(adata, self.fsfu[gene_name + f'_fits'], self.fsfu[gene_name + f'_fitu'], axes[1][1])
        axes[1][1].set_title('x: fits -- y: fitu')

        plt.show()
    
    def spliced_time(self, gene_name):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        self.sns_plot(self.ms, gene_name + '_s', 'time', 'ms', axes[0])
        self.sns_plot(self.fs, gene_name + f'_fits', 'time', 'fits', axes[1])

    def unspliced_time(self, gene_name):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        self.sns_plot(self.mu, gene_name + '_u', 'time', 'mu', axes[0])
        self.sns_plot(self.fu, gene_name + f'_fitu', 'time', 'fitu', axes[1])
        plt.show()

    def sns_plot(self, data, expression, x, y, loc):
        df = pd.DataFrame(data=[self.lt, data[expression].values], index=[x, y]).T
        df['labels'] = self.label
        sns.scatterplot(x=x, y=y, data=df, hue='labels', sizes=1, palette=self.palette, ax=loc)

    def putative_trans_time(self):
        pass

    def vars_trends(self, gene_name, adata):
        par_names = adata.uns['par_names']
        para = pd.DataFrame(index=par_names, columns=['Values'])
        para['Values'] = self.var.loc[gene_name].values
        self.para = para.iloc[:, -1].T.astype(np.float32)

    def plot_mf(self, adata, s, u, ax=None, hue=None):
        data = [np.squeeze(s), np.squeeze(u)]
        df = pd.DataFrame(data=data, index=['spliced', 'unspliced']).T

        if adata.shape[0] == np.squeeze(s).shape[0]:
            df['labels'] = adata.obs[adata.uns['label']].values
            hue = 'labels'

        sns.scatterplot(x='spliced', y='unspliced', data=df, sizes=1, 
            palette=self.palette, ax=ax, hue=hue)

    def func(self, validate=None, t_cell=None):
        s, u = validate.get_s_u(self.para.values, t_cell)
        return s, u

    def plot_range(self, gene_name, adata, ctype=None):
        #! solving the scaling of unspliced problem afterwards
        from .optimize_utils import exp_args
        from .optimize_utils import Model_Utils
        validate = Model_Utils(config=adata.uns['config'])
        self.vars_trends(gene_name, adata)

        columns = exp_args(adata)
        for col in columns:
            self.para[col] = np.log(self.para[col])

        boundary = (
            np.reshape(self.para['t'] - 3 * (1 / np.sqrt(2 * np.exp(self.para['a']))), (1, 1)), 
            np.reshape(self.para['t'] + 3 * (1 / np.sqrt(2 * np.exp(self.para['a']))), (1, 1))
        )
        
        adata = adata[:, gene_name]
        if ctype != None:
            adata = adata[adata.obs[adata.obs[adata.uns['label']] == ctype].index, :]

        spre, upre = self.func(validate, validate.init_time(boundary, (3000, 1)))
        sone, uone = self.func(validate, validate.init_time((0, 1), (3000, 1)))
        sfit, ufit = self.func(validate, adata.obs['latent_time'].values)

        display.clear_output(wait=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        self.plot_mf(adata, adata.layers['Ms'], adata.layers['Mu'] / adata.var['scaling'].values, axes[0])
        self.plot_mf(adata, spre, upre, axes[0])
        self.plot_mf(adata, sone, uone, axes[0])
        self.plot_mf(adata, sfit, ufit, axes[1])

        if 'li_coef' in adata.var.columns:
            x = np.linspace(np.min(adata.layers['Ms']), np.max(adata.layers['Ms']), 1000)
            y = x * adata.var['li_coef'].values
            axes[0].plot(x, y, ls='--', c='red')

        plt.show()

    def plot_scv_fit(self, gene_name, adata):
        DIR = adata.uns['temp']
        self.mu = pd.read_csv(f'{DIR}/Mu.csv', index_col=0).add_suffix('_u')
        self.ms = pd.read_csv(f'{DIR}/Ms.csv', index_col=0).add_suffix('_s')

        self.fu = pd.read_csv(f'{DIR}/scvu.csv', index_col=0).add_suffix(f'_fitu')
        self.fs = pd.read_csv(f'{DIR}/scvs.csv', index_col=0).add_suffix(f'_fits')

        self.lt = adata.obs['latent_time'].values
        self.label = adata.obs[adata.uns['label']].values

        self.concat_data()
        self.inspect_genes(gene_name, adata)
        self.spliced_time(gene_name)
        self.unspliced_time(gene_name)

def exam_genes(adata, gene_name=None, time_metric='latent_time'):
    display.clear_output(wait=True)
    from .individual_gene import Validation
    examine = Validation(adata, time_metric=time_metric)
    examine.init_data(adata)
    examine.concat_data()
    examine.inspect_genes(gene_name, adata)
    examine.spliced_time(gene_name)
    examine.unspliced_time(gene_name)
    examine.vars_trends(gene_name, adata)

def exam_scv(data_path, gene_name, basis, label):
    try:
        import scvelo as scv
    except ModuleNotFoundError:
        print ('Install scVelo via `pip install scvelo`')
    
    adata = scv.read(data_path)
    adata.uns['datapath'] = data_path
    adata.uns['label'] = label
    adata.uns['basis'] = basis

    scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
    scv.tl.recover_dynamics(adata, n_jobs=20)
    scv.tl.velocity(adata, mode='dynamical')
    scv.tl.velocity_graph(adata)
    scv.tl.latent_time(adata)

    if basis != None:
        scv.pl.velocity_embedding_stream(
            adata, basis=basis, color=label,
            legend_loc='far right', dpi=200, 
            title='scVelo dynamical model'
        )

        scv.pl.scatter(adata, color='latent_time', color_map='gnuplot', size=50)

    if gene_name != None:    
        examine = Validation(adata, time_metric='latent_time')
        examine.plot_scv_fit(gene_name, adata)

    return adata