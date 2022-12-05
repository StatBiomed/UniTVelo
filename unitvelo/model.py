import tensorflow as tf
import numpy as np
from scvelo.tools.utils import make_unique_list
from tqdm import tqdm
import scvelo as scv
from .optimize_utils import Model_Utils, exp_args
from .utils import save_vars, new_adata_col, min_max
from .pl import plot_loss

exp = tf.math.exp
log = tf.math.log
sum = tf.math.reduce_sum
mean = tf.math.reduce_mean
sqrt = tf.math.sqrt
abs = tf.math.abs
square = tf.math.square
pow = tf.math.pow
std = tf.math.reduce_std
var = tf.math.reduce_variance

class Recover_Paras(Model_Utils):
    def __init__(
        self,
        adata,
        Ms,
        Mu,
        var_names,
        idx=None,
        rep=1,
        config=None
    ):
        super().__init__(
            adata=adata, 
            var_names=var_names,
            Ms=Ms,
            Mu=Mu,
            config = config
        )

        self.idx = idx
        self.rep = rep
        self.scaling = adata.var['scaling'].values
        self.flag = True

        self.init_pars()
        self.init_vars()
        self.init_weights()
        self.adata.uns['par_names'] = self.default_pars_names
        self.t_cell = self.compute_cell_time(args=None)
        self.pi = tf.constant(np.pi, dtype=tf.float32)

    def compute_cell_time(self, args=None, iter=None, show=False):
        if args != None:
            boundary = (args[4] - 3 * (1 / sqrt(2 * exp(args[3]))), 
                        args[4] + 3 * (1 / sqrt(2 * exp(args[3]))))
            t_range = boundary if self.config.RESCALE_TIME else (0, 1)
            x = self.init_time(t_range, (3000, self.adata.n_vars))

            s_predict, u_predict = self.get_s_u(args, x)
            s_predict = tf.expand_dims(s_predict, axis=0) # 1 3000 d
            u_predict = tf.expand_dims(u_predict, axis=0)
            Mu = tf.expand_dims(self.Mu, axis=1) # n 1 d
            Ms = tf.expand_dims(self.Ms, axis=1)

            t_cell = self.match_time(Ms, Mu, s_predict, u_predict, x.numpy(), iter)

            if self.config.AGGREGATE_T:
                t_cell = tf.reshape(t_cell, (-1, 1))
                t_cell = tf.broadcast_to(t_cell, self.adata.shape)
            
            # plot_phase_portrait(self.adata, args, Ms, Mu, s_predict, u_predict)

        else:
            boundary = (self.t - 3 * (1 / sqrt(2 * exp(self.log_a))), 
                        self.t + 3 * (1 / sqrt(2 * exp(self.log_a))))

            t_cell = self.init_time((0, 1), self.adata.shape)

            if self.rep == 1:
                if self.config.NUM_REP_TIME == 're_init':
                    t_cell = 1 - t_cell
                if self.config.NUM_REP_TIME == 're_pre':
                    t_cell = 1 - self.adata.obs['latent_time_gm'].values
                    t_cell = tf.broadcast_to(t_cell.reshape(-1, 1), self.adata.shape)
            
            if self.rep > 1:
                tf.random.set_seed(np.ceil((self.rep - 1) / 2))
                shuffle = tf.random.shuffle(t_cell)
                t_cell = shuffle if self.rep % 2 == 1 else 1 - shuffle

            if self.config.IROOT == 'gcount':
                print ('---> Use Gene Counts as initial.')
                self.adata.obs['gcount'] = np.sum(self.adata.X.todense() > 0, axis=1)
                g_time = 1 - min_max(self.adata.obs.groupby(self.adata.uns['label'])['gcount'].mean())

                for id in list(g_time.index):
                    self.adata.obs.loc[self.adata.obs[self.adata.uns['label']] == id, 'gcount'] = g_time[id]

                scv.pl.scatter(self.adata, color='gcount', cmap='gnuplot', dpi=100)
                t_cell = tf.cast(
                    tf.broadcast_to(
                        self.adata.obs['gcount'].values.reshape(-1, 1), 
                        self.adata.shape), 
                    tf.float32)

            elif type(self.config.IROOT) == list:
                t_cell, perc = [], []
                for prior in self.config.IROOT:
                    expr = np.array(self.adata[:, prior[0]].layers['Ms'])

                    perc.append(np.max(expr) * 0.75) # modify 0.75 for parameter tuning
                    t_cell.append(min_max(expr) if prior[1] == 'increase' else 1 - min_max(expr))
                
                perc_total = np.sum(perc)
                perc = [perc[i] / perc_total for i in range(len(perc))]
                print (f'assigned weights of IROOT {list(np.around(np.array(perc), 2))}')
                t_cell = [perc[i] * t_cell[i] for i in range(len(perc))]
                t_cell = tf.cast(tf.broadcast_to(np.sum(t_cell, axis=0).reshape(-1, 1), self.adata.shape), tf.float32)

            elif self.config.IROOT in self.adata.obs[self.adata.uns['label']].values:
                print ('Use diffusion pseudotime as initial.')
                import scanpy as sc
                sc.tl.diffmap(self.adata)
                self.adata.uns['iroot'] = \
                    np.flatnonzero(
                        self.adata.obs[self.adata.uns['label']] == self.config.IROOT
                    )[0]
                sc.tl.dpt(self.adata)

                if show:
                    scv.pl.scatter(self.adata, color='dpt_pseudotime', cmap='gnuplot', dpi=100)
                
                t_cell = tf.cast(
                    tf.broadcast_to(
                        self.adata.obs['dpt_pseudotime'].values.reshape(-1, 1), 
                        self.adata.shape), 
                    tf.float32)
        
            else:
                pass

        return t_cell

    def get_loglikelihood(self, distx=None, varx=None):
        n = np.clip(len(distx) - len(self.u) * 0.01, 2, None)
        loglik = -1 / 2 / n * np.sum(distx) / varx
        loglik -= 1 / 2 * np.log(2 * np.pi * varx)

    def amplify_gene(self, t_cell, iter):
        from sklearn import linear_model
        from sklearn.metrics import r2_score
        print (f'\nExaming genes which are not initially considered as velocity genes')

        r2 = np.repeat(-1., self.Ms.shape[1])
        reg = linear_model.LinearRegression()

        for col in tqdm(range(self.Ms.shape[1])):
            if not self.idx[col]:
                y = np.reshape(self.Ms[:, col], (-1, 1))
                x = np.reshape(t_cell, (-1, 1))

                reg.fit(x, y)
                y_pred = reg.predict(x)
                r2[col] = r2_score(y, y_pred)
        
        self.agenes = r2 >= self.config.AGENES_R2
        self.adata.var['amplify_r2'] = r2
        self.adata.var['amplify_genes'] = self.agenes
        self.flag = False

        _ = self.get_log(sum(self.s_r2, axis=0) + sum(self.u_r2, axis=0), True, iter=iter)
        self.used_agenes = np.array(self.adata.var['amplify_genes'].values)
        self.total_genes = self.idx | self.used_agenes

        print (f'# of amplified genes {self.agenes.sum()}, # of used {self.used_agenes.sum()}')
        print (f'# of (velocity + used) genes {self.total_genes.sum()}')

        self.infi_genes = ~np.logical_xor(~self.agenes, self.used_agenes)
        self.adata.var['amplify_infi'] = self.infi_genes
        print (f'# of infinite (or nan) genes {self.infi_genes.sum()}')

    def compute_loss(self, args, t_cell, Ms, Mu, iter, progress_bar):
        self.s_func, self.u_func = self.get_s_u(args, t_cell)
        udiff, sdiff = Mu - self.u_func, Ms - self.s_func

        if (self.config.AGENES_R2 < 1) & (iter > self.agenes_thres):
            self.u_r2 = square(udiff)
            self.s_r2 = square(sdiff) 

            if self.flag:
                self.amplify_gene(t_cell.numpy()[:, 0], iter=iter)

            if iter > int(0.9 * self.config.MAX_ITER) & self.config.REG_LOSS:
                self.s_r2 = self.s_r2 + \
                    std(Ms, axis=0) * self.config.REG_TIMES * \
                    exp(-square(args[4] - 0.5) / self.config.REG_SCALE)

            #compute variance, equivalent to np.var(np.sign(sdiff) * np.sqrt(distx))
            self.vars = mean(self.s_r2, axis=0) \
                - square(mean(tf.math.sign(sdiff) * sqrt(self.s_r2), axis=0))
            self.varu = mean(self.u_r2 * square(self.scaling), axis=0) \
                - square(mean(tf.math.sign(udiff) * sqrt(self.u_r2) * self.scaling, axis=0))

            #! edge case of mRNAs levels to be the same across all cells
            self.vars += tf.cast(self.vars == 0, tf.float32)
            self.varu += tf.cast(self.varu == 0, tf.float32)

            self.u_log_likeli = \
                - (Mu.shape[0] / 2) * log(2 * self.pi * self.varu) \
                - sum(self.u_r2 * square(self.scaling), axis=0) / (2 * self.varu) 
            self.s_log_likeli = \
                - (Ms.shape[0] / 2) * log(2 * self.pi * self.vars) \
                - sum(self.s_r2, axis=0) / (2 * self.vars) 

            error_1 = np.sum(sum(self.u_r2, axis=0).numpy()[self.total_genes]) / np.sum(self.total_genes)
            error_2 = np.sum(sum(self.s_r2, axis=0).numpy()[self.total_genes]) / np.sum(self.total_genes)
            self.se.append(error_1 + error_2)
            progress_bar.set_description(f'Loss (Total): {self.se[-1]:.3f}, (Spliced): {error_2:.3f}, (Unspliced): {error_1:.3f}')

            return self.get_loss(iter,
                                sum(self.s_r2, axis=0), 
                                sum(self.u_r2, axis=0))

        else:
            self.u_r2 = square(udiff)
            self.s_r2 = square(sdiff)

            if (self.config.FIT_OPTION == '1') & \
                (iter > int(0.9 * self.config.MAX_ITER)) & self.config.REG_LOSS:
                self.s_r2 = self.s_r2 + \
                    std(Ms, axis=0) * self.config.REG_TIMES * \
                    exp(-square(args[4] - 0.5) / self.config.REG_SCALE)

            #! convert for self.varu to account for scaling in pre-processing
            self.vars = mean(self.s_r2, axis=0) \
                - square(mean(tf.math.sign(sdiff) * sqrt(self.s_r2), axis=0))
            self.varu = mean(self.u_r2 * square(self.scaling), axis=0) \
                - square(mean(tf.math.sign(udiff) * sqrt(self.u_r2) * self.scaling, axis=0))

            self.u_log_likeli = \
                - (Mu.shape[0] / 2) * log(2 * self.pi * self.varu) \
                - sum(self.u_r2 * square(self.scaling), axis=0) / (2 * self.varu) 
            self.s_log_likeli = \
                - (Ms.shape[0] / 2) * log(2 * self.pi * self.vars) \
                - sum(self.s_r2, axis=0) / (2 * self.vars) 

            error_1 = np.sum(sum(self.u_r2, axis=0).numpy()[self.idx]) / np.sum(self.idx)
            error_2 = np.sum(sum(self.s_r2, axis=0).numpy()[self.idx]) / np.sum(self.idx)
            self.se.append(error_1 + error_2)
            progress_bar.set_description(f'Loss (Total): {self.se[-1]:.3f}, (Spliced): {error_2:.3f}, (Unspliced): {error_1:.3f}')
            
            self.vgene_loss = self.se[-1]
            return self.get_loss(iter,
                                sum(self.s_r2, axis=0), 
                                sum(self.u_r2, axis=0))

    def fit_likelihood(self):
        Ms, Mu, t_cell = self.Ms, self.Mu, self.t_cell
        log_gamma, log_beta, offset = self.log_gamma, self.log_beta, self.offset
        intercept = self.intercept
        log_a, t, log_h = self.log_a, self.t, self.log_h

        from packaging import version
        if version.parse(tf.__version__) >= version.parse('2.11.0'):
            optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.init_lr, amsgrad=True)
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.init_lr, amsgrad=True)
        
        pre = tf.repeat(1e6, Ms.shape[1]) # (2000, )
        self.se, self.m_args, self.m_ur2, self.m_sr2 = [], None, None, None
        self.m_ullf, self.m_sllf = None, None

        progress_bar = tqdm(range(self.config.MAX_ITER))
        for iter in progress_bar:
            with tf.GradientTape() as tape:                
                args = [
                    log_gamma, 
                    log_beta, 
                    offset, 
                    log_a, 
                    t, 
                    log_h, 
                    intercept
                ]
                obj = self.compute_loss(args, t_cell, Ms, Mu, iter, progress_bar)

            stop_cond = self.get_stop_cond(iter, pre, obj)

            if iter > self.agenes_thres + 1:
                self.m_args = self.get_optimal_res(args, self.m_args)
                self.m_ur2 = self.get_optimal_res(self.u_r2, self.m_ur2)
                self.m_sr2 = self.get_optimal_res(self.s_r2, self.m_sr2)
                self.m_ullf = self.get_optimal_res(self.u_log_likeli, self.m_ullf)
                self.m_sllf = self.get_optimal_res(self.s_log_likeli, self.m_sllf)

            if (iter > self.agenes_thres) & \
                (iter == self.config.MAX_ITER - 1 or \
                tf.math.reduce_all(stop_cond) == True or \
                min(self.se[self.agenes_thres + 1:]) * 1.1 < self.se[-1] 
                    if (iter > self.agenes_thres + 1) else False):

                if (iter > int(0.9 * self.config.MAX_ITER)) & self.config.REG_LOSS & \
                    (min(self.se[self.agenes_thres:]) * 1.1 >= self.se[-1]):
                    self.m_args = args
                    self.m_ur2 = self.u_r2
                    self.m_sr2 = self.s_r2
                    self.m_ullf = self.u_log_likeli
                    self.m_sllf = self.s_log_likeli

                t_cell = self.compute_cell_time(args=self.m_args, iter=iter)
                _ = self.get_fit_s(self.m_args, t_cell)
                s_derivative = self.get_s_deri(self.m_args, t_cell)
                # s_derivative = exp(args[1]) * Mu - exp(args[0]) * Ms

                self.post_utils(iter, self.m_args)
                break

            args_to_optimize = self.get_opt_args(iter, args)
            gradients = tape.gradient(target=obj, sources=args_to_optimize)

            # convert gradients of variables with unused genes to 0
            # keep other gradients by multiplying 1
            if (self.config.AGENES_R2 < 1) & (iter > self.agenes_thres):
                convert = tf.cast(self.total_genes, tf.float32)
                processed_grads = [g * convert for g in gradients]
            else:
                convert = tf.cast(self.idx, tf.float32)
                processed_grads = [g * convert for g in gradients]

            optimizer.apply_gradients(zip(processed_grads, args_to_optimize))
            pre = obj

            if iter > 0 and int(iter % 800) == 0:
                t_cell = self.compute_cell_time(args=args, iter=iter)

        self.adata.layers['fit_t'] = t_cell.numpy() if self.config.AGGREGATE_T else t_cell
        self.adata.var['velocity_genes'] = self.total_genes if not self.flag else self.idx
        self.adata.layers['fit_t'][:, ~self.adata.var['velocity_genes'].values] = np.nan

        return self.get_interim_t(t_cell, self.adata.var['velocity_genes'].values), s_derivative.numpy(), self.adata

    def get_optimal_res(self, current, opt):
        return current if min(self.se[self.agenes_thres + 1:]) == self.se[-1] else opt

    def post_utils(self, iter, args):
        # Reshape un/spliced variance to (ngenes, ) and save
        self.adata.var['fit_vars'] = np.squeeze(self.vars)
        self.adata.var['fit_varu'] = np.squeeze(self.varu)

        # Save predicted parameters of RBF kernel to adata
        self.save_pars([item.numpy() for item in args])
        self.adata.var['fit_beta'] /= self.scaling
        self.adata.var['fit_intercept'] *= self.scaling

        # Plotting function for examining model loss
        plot_loss(iter, self.se, self.agenes_thres)

        # Save observations, predictinos and variables locally
        save_vars(self.adata, args, 
                self.s_func.numpy(), self.u_func.numpy(), 
                self.K, self.scaling)

        #! Model loss, log likelihood and BIC based on unspliced counts
        gene_loss = sum(self.m_ur2, axis=0) / self.nobs \
            if self.config.FILTER_CELLS \
            else sum(self.m_ur2, axis=0) / self.Ms.shape[0]

        list_name = ['fit_loss', 'fit_llf']
        list_data = [gene_loss.numpy(), self.m_ullf.numpy()]
        new_adata_col(self.adata, list_name, list_data)

        # Mimimum loss during optimization, might not be the actual minimum
        r2_spliced = 1 - sum(self.m_sr2, axis=0) / var(self.Ms, axis=0) \
            / (self.adata.shape[0] - 1)
        r2_unspliced = 1 - sum(self.m_ur2, axis=0) / var(self.Mu, axis=0) \
            / (self.adata.shape[0] - 1)
        new_adata_col(self.adata, ['fit_sr2', 'fit_ur2'], [r2_spliced.numpy(), r2_unspliced.numpy()])

        tloss = min(self.se[self.agenes_thres + 1:])
        self.adata.uns['loss'] = self.vgene_loss
        print (f'Total loss {tloss:.3f}, vgene loss {self.vgene_loss:.3f}')

    def save_pars(self, paras):
        columns = exp_args(self.adata, 1)
        for i, name in enumerate(self.default_pars_names):
            self.adata.var[f"fit_{name}"] = np.transpose(np.squeeze(paras[i]))

            if name in columns:
                self.adata.var[f"fit_{name}"] = np.exp(self.adata.var[f"fit_{name}"])            

def lagrange(
    adata,
    idx=None,
    Ms=None,
    Mu=None,
    var_names="velocity_genes",
    rep=1,
    config=None
):
    if len(set(adata.var_names)) != len(adata.var_names):
        adata.var_names_make_unique()

    var_names = adata.var_names[idx]
    var_names = make_unique_list(var_names, allow_array=True)

    model = Recover_Paras(
        adata,
        Ms,
        Mu,
        var_names,
        idx=idx,
        rep=rep,
        config=config
    )

    latent_time_gm, s_derivative, adata = model.fit_likelihood()

    if 'latent_time' in adata.obs.columns:
        del adata.obs['latent_time']
    adata.obs['latent_time_gm'] = min_max(latent_time_gm[:, 0])

    return s_derivative, adata