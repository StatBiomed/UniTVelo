import numpy as np
import os
import pandas as pd
from .model import lagrange
from .utils import make_dense, get_weight, R2, new_adata_col
import scvelo as scv
from tqdm.notebook import tqdm
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
np.random.seed(42)

class Velocity:
    def __init__(
        self,
        adata=None,
        min_ratio=0.01,
        min_r2=0.01,
        fit_offset=False,
        perc=[5, 95],
        vkey='velocity',
        config=None
    ):
        self.adata = adata
        self.vkey = vkey

        self.Ms = adata.layers["spliced"] if config.USE_RAW else adata.layers["Ms"].copy()
        self.Mu = adata.layers["unspliced"] if config.USE_RAW else adata.layers["Mu"].copy()
        self.Ms, self.Mu = make_dense(self.Ms), make_dense(self.Mu)

        self.min_r2 = min_r2
        self.min_ratio = min_ratio
        self.highly_variable = adata.var['highly_variable'] if 'highly_variable' in adata.var.keys() else None
        
        n_obs, n_vars = self.Ms.shape
        self.gamma = np.zeros(n_vars, dtype=np.float32)
        self.r2 = np.zeros(n_vars, dtype=np.float32)
        self.velocity_genes = np.ones(n_vars, dtype=np.bool)
        self.residual_scale = np.zeros([n_obs, n_vars], dtype=np.float32)
    
        self.general = config.GENERAL
        self.perc = perc
        self.fit_offset = fit_offset
        self.examine_genes = config.EXAMINE_GENE
        self.adata.uns['examine_genes'] = config.EXAMINE_GENE
        self.config = config
        self.df_linear = None

    def get_velo_genes(self, gene_ids=None):
        variable = self.adata.var
        if variable.index[0].startswith('ENSMUSG'):
            variable.index = variable['gene']
            variable.index.name = 'index' 
        
        weights = get_weight(self.Ms, self.Mu, perc=95)
        Ms, Mu = weights * self.Ms, weights * self.Mu
        self.gamma_quantile = np.sum(Mu * Ms, axis=0) / np.sum(Ms * Ms, axis=0)

        self.scaling = np.std(self.Mu, axis=0) / np.std(self.Ms, axis=0)
        if self.config.R2_ADJUST:
            Ms, Mu = self.Ms, self.Mu

        self.gene_index = variable.index
        self.gamma_ref = np.sum(Mu * Ms, axis=0) / np.sum(Ms * Ms, axis=0)
        self.residual_scale = self.Mu - self.gamma_ref * self.Ms
        self.r2 = R2(self.residual_scale, total=self.Mu - np.mean(self.Mu, axis=0))

        self.velocity_genes = np.ones(Ms.shape[1], dtype=np.bool)

        if self.examine_genes:
            self.velocity_genes = np.zeros(Ms.shape[1], dtype=np.bool)
            self.velocity_genes[
                np.argwhere(self.adata.var.index == self.examine_genes)] = True

        elif gene_ids is not None:
            temp = []
            for gene in variable.index:
                if gene in gene_ids:
                    temp.append(True)
                else:
                    temp.append(False)
            self.velocity_genes = np.array(temp)

        elif self.config.RAW_GENES:
            self.velocity_genes = np.ones(Ms.shape[1], dtype=np.bool)

        elif self.config.OFFSET_GENES:
            self.fit_linear(None, self.Ms, self.Mu, 'vgene_offset', coarse=False)

            if self.config.FILTER_CELLS:
                self.fit_linear(None, self.Ms, self.Mu, 'vgene_offset', coarse=True)

        else:
            self.velocity_genes = (
                (self.r2 > self.min_r2)
                & (self.r2 < 0.95)
                & (self.gamma_quantile > self.min_ratio)
                & (self.gamma_ref > self.min_ratio)
                & (np.max(self.Ms > 0, axis=0) > 0)
                & (np.max(self.Mu > 0, axis=0) > 0)
            )
            print (f'---> # of velocity genes used {self.velocity_genes.sum()}')
            
            if self.config.R2_ADJUST:
                lb, ub = np.nanpercentile(self.scaling, [10, 90])
                self.velocity_genes = (
                    self.velocity_genes
                    & (self.scaling > np.min([lb, 0.03]))
                    & (self.scaling < np.max([ub, 3]))
                )
            print (f'---> # of velocity genes used {self.velocity_genes.sum()}')

        if True:
            self.init_weights()
            self.velocity_genes = self.velocity_genes & (self.nobs > 0.05 * Ms.shape[1])

        self.adata.var['velocity_gamma'] = self.gamma_ref
        self.adata.var['scaling'] = self.scaling
        self.adata.var['velocity_genes'] = self.velocity_genes
        # self.adata.var['fit_r2'] = self.r2
        self.adata.uns[f"{self.vkey}_params"] = {"mode": self.general, "perc": self.perc}
        
        if np.sum(self.velocity_genes) < 2:
            print ('---> Low signal in splicing dynamics.')

    def init_weights(self):
        nonzero_s, nonzero_u = self.Ms > 0, self.Mu > 0
        weights = np.array(nonzero_s & nonzero_u, dtype=bool)
        self.nobs = np.sum(weights, axis=0)

    def fit_deterministic(self, idx, Ms, Mu, Ms_scale, Mu_scale):
        df_gamma = pd.DataFrame(index=self.gene_index, data=0, 
            dtype=np.float32, columns=['coef', 'inter'])

        if self.fit_offset:
            pass

        else:
            weights_new = get_weight(Ms, Mu, perc=self.perc)
            x, y = weights_new * Ms_scale, weights_new * Mu_scale
            df_gamma['coef'][idx] = np.sum(y * x, axis=0) / np.sum(x * x, axis=0)

        residual = self.Mu \
            - np.broadcast_to(df_gamma['coef'].values, self.Mu.shape) * self.Ms \
            - np.broadcast_to(df_gamma['inter'].values, self.Mu.shape)
        
        self.adata.var['velocity_gamma'] = df_gamma['coef'].values
        self.adata.var['intercept'] = df_gamma['inter'].values
        return residual

    def fit_linear(self, idx, Ms, Mu, method='bic', coarse=False):
        '''
        [bic] for linear BIC comparison with algorithm BIC
        [kinetic] for selection of cyclic kinetic gene and monotonic expression gene
        [vgene_offset] for determination of velocity gene alternatively using offset
        '''

        if method == 'vgene_offset':
            from sklearn import linear_model
            from sklearn.metrics import r2_score

            index = self.adata.var.index
            linear = pd.DataFrame(index=index, data=0, dtype=np.float32, 
                columns=['coef', 'inter', 'r2'])

            reg = linear_model.LinearRegression()
            for col in tqdm(range(Ms.shape[1])):
                if coarse == False:
                    if self.config.FILTER_CELLS:
                        nonzero_s = Ms[:, col] > 0
                        nonzero_u = Mu[:, col] > 0
                        valid = np.array(nonzero_s & nonzero_u, dtype=bool)
                        sobs = np.reshape(Ms[:, col][valid], (-1, 1))
                        uobs = np.reshape(Mu[:, col][valid], (-1, 1))

                    else:
                        sobs = np.reshape(Ms[:, col], (-1, 1))
                        uobs = np.reshape(Mu[:, col], (-1, 1))
                
                else:
                    sobs = np.reshape(Ms[:, col], (-1, 1))
                    uobs = np.reshape(Mu[:, col], (-1, 1))                    
                
                reg.fit(sobs, uobs)
                u_pred = reg.predict(sobs)

                linear.loc[index[col], 'coef'] = reg.coef_
                linear.loc[index[col], 'inter'] = reg.intercept_
                linear.loc[index[col], 'r2'] = r2_score(uobs, u_pred)

            self.adata.var['velocity_inter'] = np.array(linear['inter'].values)
            self.gamma_ref = np.array(linear['coef'].values)
            self.r2 = np.array(linear['r2'].values)

            if coarse:
                nonzero_genes = self.velocity_genes

            self.velocity_genes = (
                self.velocity_genes
                & (self.r2 > self.min_r2)
                & (self.r2 < 0.95)
                & (np.array(linear['coef'].values) > self.min_ratio)
                & (np.max(self.Ms > 0, axis=0) > 0)
                & (np.max(self.Mu > 0, axis=0) > 0)
            )
            print (f'---> # of velocity genes used {self.velocity_genes.sum()}')

            lb, ub = np.nanpercentile(self.scaling, [10, 90])
            self.velocity_genes = (
                self.velocity_genes
                & (self.scaling > np.min([lb, 0.03]))
                & (self.scaling < np.max([ub, 3]))
            )
            print (f'---> # of velocity genes used {self.velocity_genes.sum()}')
        
            if coarse:
                zero_genes = ~(nonzero_genes == self.velocity_genes)
                zero_genes = self.adata[:, zero_genes].var.index
                print (zero_genes)
                print (np.sum((nonzero_genes != self.velocity_genes) 
                    & (nonzero_genes == True)))
                print (np.sum((nonzero_genes != self.velocity_genes)
                    & (self.velocity_genes == True)))

                from .pl import plot_zero_gene_example
                plot_zero_gene_example(zero_genes, self.adata, 
                                self.df_linear.loc[zero_genes], linear.loc[zero_genes])
            
            self.df_linear = linear

        if method == 'bic':
            import statsmodels.api as sm
            adata_select = self.adata[:, idx]
            index = adata_select.var.index
            linear = pd.DataFrame(index=index, data=0, dtype=np.float32, 
                columns=['coef', 'inter',  'r2', 'loss', 'bic', 'llf'])

            for col in tqdm(range(Ms.shape[1])):
                if self.config.FILTER_CELLS:
                    nonzero_s = Ms[:, col] > 0
                    nonzero_u = Mu[:, col] > 0
                    valid = np.array(nonzero_s & nonzero_u, dtype=bool)
                    sobs = np.reshape(Ms[:, col][valid], (-1, 1))
                    uobs = np.reshape(Mu[:, col][valid], (-1, 1))

                else:
                    sobs = np.reshape(Ms[:, col], (-1, 1))
                    uobs = np.reshape(Mu[:, col], (-1, 1))
                
                X, Y = sm.add_constant(sobs), uobs
                model = sm.OLS(Y, X)
                results = model.fit()

                linear.loc[index[col], 'coef'] = results.params[1]
                linear.loc[index[col], 'inter'] = results.params[0]
                linear.loc[index[col], 'r2'] = results.rsquared
                linear.loc[index[col], 'loss'] = results.mse_resid
                linear.loc[index[col], 'bic'] = results.bic
                linear.loc[index[col], 'llf'] = results.llf
            
            new_adata_col(self.adata, idx,
                ['li_coef', 'li_inter', 'li_r2', 'li_loss', 'li_bic', 'li_llf'], 
                [linear['coef'].values, linear['inter'].values, linear['r2'].values, 
                linear['loss'].values, linear['bic'].values, linear['llf'].values])

    def fit_curve(self, adata, idx, Ms_scale, Mu_scale, rep=1):
        physical_devices = tf.config.list_physical_devices('GPU')

        if len(physical_devices) == 0 or self.config.GPU == -1:
            tf.config.set_visible_devices([], 'GPU')
            with tf.device('/cpu:0'):
                residual, adata = lagrange(
                    adata, idx=idx,
                    Ms=Ms_scale, Mu=Mu_scale, 
                    rep=rep, config=self.config
                )

            return residual, adata

        else:
            assert self.config.GPU < len(physical_devices), \
                'Please specify the correct GPU card.'
            tf.config.set_visible_devices(physical_devices[self.config.GPU], 'GPU')

            os.environ["CUDA_VISIBLE_DEVICES"] = f'{self.config.GPU}'
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)

            with tf.device(f'/gpu:{self.config.GPU}'):
                residual, adata = lagrange(
                    adata, idx=idx,
                    Ms=Ms_scale, Mu=Mu_scale, 
                    rep=rep, config=self.config
                )

            return residual, adata

    def fit_velo_genes(self, basis='umap', rep=1):
        idx = self.velocity_genes
        Ms, Mu = self.Ms[:, idx], self.Mu[:, idx]
        print (f'---> # of velocity genes used {idx.sum()}')

        if self.config.RESCALE_DATA:
            Ms_scale, Mu_scale = Ms, Mu / (np.std(Mu, axis=0) / np.std(Ms, axis=0))
        else:
            Ms_scale, Mu_scale = Ms, Mu

        assert self.general in ['Deterministic', 'Curve', 'Linear'], \
            'self.general not supported'

        if self.general == 'Curve':
            self.adata.layers['Mu_scale'] = \
                self.Mu / (np.std(self.Mu, axis=0) / np.std(self.Ms, axis=0))

            residual, self.adata = \
                self.fit_curve(self.adata, idx, Ms_scale, Mu_scale, rep=rep)

            if False:
                self.fit_linear(idx, Ms_scale, Mu_scale, method='bic')

                from .pl import plot_compare_bic, plot_compare_loss
                plot_compare_loss(self.adata)
                plot_compare_bic(self.adata)

        if self.general == 'Deterministic':
            residual = self.fit_deterministic(idx, Ms, Mu, Ms_scale, Mu_scale)

        if self.general == 'Linear':
            self.fit_linear(idx, Ms_scale, Mu_scale, method='kinetic')
            print (np.sum(self.adata[:, idx].var['kinetic_gene'].values) / \
                    np.sum(self.adata[:, idx].var['velocity_genes'].values))
            return self.adata

        try:
            self.adata.uns['loss'] = \
                (self.adata_dyna.uns['loss'], self.adata_mono.uns['loss'])
        except:
            self.adata.uns['loss'] = self.adata.uns['loss']
        self.adata.layers[self.vkey] = residual

        DIR = os.path.split(self.adata.uns['datapath'])[0]
        DIR = os.path.join(DIR, 'figures')
        if not os.path.exists(DIR):
            os.mkdir(DIR)

        if self.examine_genes == False and basis != None:
            scv.tl.velocity_graph(self.adata, sqrt_transform=True)
            scv.tl.velocity_embedding(self.adata, basis=basis)

        return self.adata