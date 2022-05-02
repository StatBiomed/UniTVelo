import tensorflow as tf
import numpy as np
from scvelo.tools.utils import make_unique_list
from tqdm.notebook import tqdm
import scvelo as scv
from .optimize_utils import Model_Utils, exp_args
from .utils import save_vars, new_adata_col, min_max
from .pl import plot_phase_portrait, plot_loss

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
        adata_ori,
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

        self.adata_ori = adata_ori
        self.idx = idx
        self.pars = []
        self.rep = rep

        self.init_pars()
        self.init_vars()
        self.init_weights()
        self.adata.uns['par_names'] = self.default_pars_names
        self.adata_ori.uns['par_names'] = self.default_pars_names
        self.t_cell = self.compute_cell_time(args=None)
        self.pi = tf.constant(np.pi, dtype=tf.float32)

    def compute_cell_time(self, args=None):
        if args != None:
            boundary = (args[6] - 3 * (1 / sqrt(2 * exp(args[5]))), 
                        args[6] + 3 * (1 / sqrt(2 * exp(args[5]))))
            range = boundary if self.config.RESCALE_TIME else (0, 1)
            x = self.init_time(range, (3000, self.adata.n_vars))

            s_predict, u_predict = self.get_s_u(args, x)
            s_predict = tf.expand_dims(s_predict, axis=0) # 1 3000 d
            u_predict = tf.expand_dims(u_predict, axis=0)
            Mu = tf.expand_dims(self.Mu, axis=1) # n 1 d
            Ms = tf.expand_dims(self.Ms, axis=1)

            t_cell = self.match_time(Ms, Mu, s_predict, u_predict, x.numpy())
            if self.config.AGGREGATE_T:
                t_cell = tf.reshape(t_cell, (-1, 1))
                t_cell = tf.broadcast_to(t_cell, self.adata.shape)
            
            plot_phase_portrait(self.adata, args, Ms, Mu, s_predict, u_predict)

        else:
            boundary = (self.t - 3 * (1 / sqrt(2 * exp(self.a))), 
                        self.t + 3 * (1 / sqrt(2 * exp(self.a))))
            t_cell = self.init_time((0, 1), self.adata.shape)
            t_cell = 1 - t_cell if self.rep == 1 else t_cell
            
            if self.rep > 1:
                tf.random.set_seed(np.ceil((self.rep - 1) / 2))
                shuffle =  tf.random.shuffle(t_cell)
                t_cell = shuffle if self.rep % 2 == 1 else 1 - shuffle

            if self.config.IROOT != None:
                print ('---> Use Diffusion Pseudotime as initial.')
                import scanpy as sc
                sc.tl.diffmap(self.adata)
                self.adata.uns['iroot'] = \
                    np.flatnonzero(
                        self.adata.obs[self.adata.uns['label']] == self.config.IROOT
                    )[0]
                sc.tl.dpt(self.adata)

                # scv.pl.scatter(self.adata, color='dpt_pseudotime', cmap='gnuplot')
                t_cell = tf.cast(
                    tf.broadcast_to(
                        self.adata.obs['dpt_pseudotime'].values.reshape(-1, 1), 
                        self.adata.shape), 
                    tf.float32)
                # print (f'\n')

        return t_cell

    def get_loglikelihood(self, distx=None, varx=None):
        n = np.clip(len(distx) - len(self.u) * 0.01, 2, None)
        loglik = -1 / 2 / n * np.sum(distx) / varx
        loglik -= 1 / 2 * np.log(2 * np.pi * varx)

    def compute_loss(self, args, t_cell, Ms, Mu, iter):
        self.s_func, self.u_func = self.get_s_u(args, t_cell)
        udiff, sdiff = Mu - self.u_func, Ms - self.s_func

        if False:
            self.u_r2 = square(udiff) * self.weights
            self.s_r2 = square(sdiff) * self.weights

            #compute variance
            #equivalent to np.var(np.sign(sdiff) * np.sqrt(distx))
            self.vars = sum(self.s_r2, axis=0) / self.nobs \
                - square(sum(tf.math.sign(sdiff) * sqrt(self.s_r2), axis=0) / self.nobs)
            self.varu = sum(self.u_r2, axis=0) / self.nobs \
                - square(sum(tf.math.sign(udiff) * sqrt(self.u_r2), axis=0) / self.nobs)

            #edge case of mRNAs levels to be the same across all cells
            self.vars += tf.cast(self.vars == 0, tf.float32)
            self.varu += tf.cast(self.varu == 0, tf.float32)

            self.u_log_likeli = \
                - (self.nobs / 2) * log(2 * self.pi * self.varu) \
                - sum(self.u_r2, axis=0) / (2 * self.varu) 
            self.s_log_likeli = \
                - (self.nobs / 2) * log(2 * self.pi * self.vars) \
                - sum(self.s_r2, axis=0) / (2 * self.vars) 

        else:
            if iter <= self.config.MAX_ITER - 2000:
                self.u_r2 = square(udiff)
                self.s_r2 = square(sdiff)
                self.se.append(int(sum(self.u_r2 + self.s_r2).numpy()))

            else:
                self.u_r2 = square(udiff)
                self.s_r2 = square(sdiff) 
                self.se.append(int(sum(self.u_r2 + self.s_r2).numpy()))
                
                if self.config.REG_LOSS:
                    self.s_r2 = self.s_r2 + \
                        std(Ms, axis=0) * self.config.REG_TIMES * exp(-square(args[6] - 0.5) / self.config.REG_SCALE)

            self.vars = mean(self.s_r2, axis=0) \
                - square(mean(tf.math.sign(sdiff) * sqrt(self.s_r2), axis=0))
            self.varu = mean(self.u_r2, axis=0) \
                - square(mean(tf.math.sign(udiff) * sqrt(self.u_r2), axis=0))

            self.u_log_likeli = \
                - (Mu.shape[0] / 2) * log(2 * self.pi * self.varu) \
                - sum(self.u_r2, axis=0) / (2 * self.varu) 
            self.s_log_likeli = \
                - (Ms.shape[0] / 2) * log(2 * self.pi * self.vars) \
                - sum(self.s_r2, axis=0) / (2 * self.vars) 

        # print ("\r", f'{self.se[-1]:,}', sep=' | ', end="")

        return self.get_loss(iter, 
                            sum(self.s_r2, axis=0), 
                            sum(self.u_r2, axis=0))

    def fit_likelihood(self):
        Ms, Mu, t_cell = self.Ms, self.Mu, self.t_cell
        gamma, beta, offset = self.gamma, self.beta, self.offset
        intercept = self.intercept
        a, t, h = self.a, self.t, self.h

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.init_lr, amsgrad=True)
        pre = tf.repeat(1e6, Ms.shape[1])

        self.se, self.m_args, self.m_ur2, self.m_ullf, self.m_sr2 = \
            [], None, None, None, None

        for iter in tqdm(range(self.config.MAX_ITER)):
            with tf.GradientTape() as tape:                
                args = [
                    gamma, 
                    beta, 
                    self.vars, 
                    self.varu, 
                    offset, 
                    a, 
                    t, 
                    h, 
                    intercept
                ]
                obj = self.compute_loss(args, t_cell, Ms, Mu, iter)

            stop_cond = self.get_stop_cond(iter, pre, obj)

            if iter > 2500:
                self.m_args = self.get_optimal_res(args, self.m_args)
                self.m_ur2 = self.get_optimal_res(self.u_r2, self.m_ur2)
                self.m_sr2 = self.get_optimal_res(self.s_r2, self.m_sr2)
                self.m_ullf = self.get_optimal_res(self.u_log_likeli, self.m_ullf)

            if (iter > 5000) & \
                (iter == self.config.MAX_ITER - 1 or \
                tf.math.reduce_all(stop_cond) == True or \
                min(self.se) * 1.1 < self.se[-1]):

                if iter > self.config.MAX_ITER - 2000 & self.config.REG_LOSS:
                    self.m_args = args
                    self.m_ur2 = self.u_r2
                    self.m_sr2 = self.s_r2
                    self.m_ullf = self.u_log_likeli

                t_cell = self.compute_cell_time(args=self.m_args)
                _ = self.get_fit_s(self.m_args, t_cell)
                s_derivative = self.get_s_deri(self.m_args, t_cell)
                # s_derivative = exp(args[1]) * Mu - exp(args[0]) * Ms

                self.post_utils(iter, self.m_args)
                break

            args_to_optimize = self.get_opt_args(iter, args)
            gradients = tape.gradient(target=obj, sources=args_to_optimize)
            processed_grads = [g for g in gradients]
            optimizer.apply_gradients(zip(processed_grads, args_to_optimize))
            pre = obj

            if iter > 0 and int(iter % 800) == 0:
                t_cell = self.compute_cell_time(args=args)

        self.adata_ori.layers['fit_t'] = np.zeros(self.adata_ori.shape) * np.nan
        try:
            self.adata_ori.layers['fit_t'][:, self.idx] = t_cell.numpy()
        except:
            self.adata_ori.layers['fit_t'][:, self.idx] = t_cell
        return self.get_interim_t(t_cell), s_derivative.numpy()

    def get_optimal_res(self, current, opt):
        return current if min(self.se[2500:]) == self.se[-1] else opt

    def post_utils(self, iter, args):
        # Reshape un/spliced variance to (1, ngenes)
        args[2] = tf.reshape(args[2], (1, len(self.var_names)))
        args[3] = tf.reshape(args[3], (1, len(self.var_names)))

        # Save predicted parameters of RBF kernel to adata
        self.save_pars([item.numpy() for item in args])

        # Plotting function for examining model loss
        plot_loss(iter, self.se)

        # Save observations, predictinos and variables locally
        save_vars(self.adata, args, self.s_func.numpy(), self.u_func.numpy(), self.K)

        #! Model loss, log likelihood and BIC based on unspliced counts
        gene_loss = sum(self.m_ur2, axis=0) / self.nobs \
            if self.config.FILTER_CELLS \
            else sum(self.m_ur2, axis=0) / self.Ms.shape[0]

        gene_llf = self.m_ullf
        bic_k = 7 if self.config.FIT_OPTION == '1' else 4
        gene_bic = bic_k * log(tf.cast(self.Ms.shape[0], tf.float32)) - 2 * gene_llf

        list_name = ['fit_loss', 'fit_bic', 'fit_llf']
        list_data = [gene_loss.numpy(), gene_bic.numpy(), gene_llf.numpy()]
        new_adata_col(self.adata_ori, self.idx, list_name, list_data)

        # Mimimum loss during optimization, might not be the actual minimum
        r2 = 1 - sum(self.m_sr2, axis=0) / var(self.Ms, axis=0) \
            / (self.adata_ori.shape[0] - 1)
        new_adata_col(self.adata_ori, self.idx, ['fit_r2'], [r2.numpy()])
        self.adata_ori.uns['loss'] = min(self.se)

    def write_pars(self, pars, add_key="fit"):
        for i, name in enumerate(self.default_pars_names):
            self.adata_ori.var[f"{add_key}_{name}"] = np.transpose(pars[i])
        
        columns = exp_args(self.adata, self.K)
        for col in columns:
            self.adata_ori.var[f"{add_key}_{col}"] = \
                np.exp(self.adata_ori.var[f"{add_key}_{col}"])

    def read_pars(self, parameter):
        if parameter.shape[0] > 1:
            for i in range(parameter.shape[0]):
                par = np.zeros(self.adata_ori.n_vars) * np.nan
                par[self.idx] = np.squeeze(parameter[i, :])
                self.pars.append(par)
        else:
            par = np.zeros(self.adata_ori.n_vars) * np.nan
            par[self.idx] = np.squeeze(parameter)
            self.pars.append(par)

    def save_pars(self, paras):
        for i in range(len(paras)):
            self.read_pars(paras[i])
        self.write_pars(self.pars)

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

    residual = np.zeros(adata.shape) * np.nan
    adata_selected = adata[:, idx]

    model = Recover_Paras(
        adata_selected,
        adata,
        Ms,
        Mu,
        var_names,
        idx=idx,
        rep=rep,
        config=config
    )

    latent_time, s_derivative = model.fit_likelihood()
    residual[:, idx] = s_derivative

    if 'latent_time' in adata.obs.columns:
        del adata.obs['latent_time']
    # adata.obs['raw_time'] = latent_time[:, 0]
    adata.obs['latent_time_gm'] = min_max(latent_time[:, 0])

    return residual, adata