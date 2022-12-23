#%%
import tensorflow as tf
import numpy as np
import logging
np.random.seed(42)

exp = tf.math.exp
pow = tf.math.pow
square = tf.math.square
sum = tf.math.reduce_sum
abs = tf.math.abs
mean = tf.math.reduce_mean
log = tf.math.log
sqrt = tf.math.sqrt

def inv(obj):
    return tf.math.reciprocal(obj + 1e-6)

def col_minmax(matrix, gene_id=None):
    if gene_id != None:
        if (np.max(matrix, axis=0) == np.min(matrix, axis=0)):
            print (gene_id)
            return matrix

    return (matrix - np.min(matrix, axis=0)) \
        / (np.max(matrix, axis=0) - np.min(matrix, axis=0))

def exp_args(adata, K=1):
    if adata.uns['base_function'] == 'Gaussian':
        columns = ['a', 'h', 'gamma', 'beta']
    else:
        columns = ['gamma', 'beta']
        columns.extend([f'a{k}' for k in range(K)])

    return columns

#%%
class Model_Utils():
    def __init__(
        self, 
        adata=None,
        var_names=None,
        Ms=None,
        Mu=None,
        config=None
    ):
        self.adata = adata
        self.var_names = var_names
        self.Ms, self.Mu = Ms, Mu
        self.K = 1
        self.win_size = 50
        self.config = config
        self.gene_log = []
        self.agenes_thres = int(config.AGENES_THRES * config.MAX_ITER)

    def init_vars(self):
        ngenes = self.Ms.shape[1]
        ones = tf.ones((1, ngenes), dtype=tf.float32)

        self.log_beta = tf.Variable(ones * 0, name='log_beta')
        self.intercept = tf.Variable(ones * 0, name='intercept')

        self.t = tf.Variable(ones * 0.5, name='t') #! mean of Gaussian
        self.log_a = tf.Variable(ones * 0, name='log_a') #! 1 / scaling of Gaussian
        self.offset = tf.Variable(ones * 0, name='offset')

        self.log_h = tf.Variable(ones * log(tf.math.reduce_max(self.Ms, axis=0)), name='log_h')

        init_gamma = self.adata.var['velocity_gamma'].values
        self.log_gamma = tf.Variable(
            log(tf.reshape(init_gamma, (1, self.adata.n_vars))), 
            name='log_gamma')

        for id in np.where(init_gamma <= 0)[0]:
            logging.info(f'name: {self.adata.var.index[id]}, gamma: {init_gamma[id]}')

        self.log_gamma = tf.Variable(
            tf.where(tf.math.is_finite(self.log_gamma), self.log_gamma, 0), 
            name='log_gamma')

        if self.config.VGENES == 'offset':
            init_inter = self.adata.var['velocity_inter'].values
            self.intercept = \
                tf.Variable(
                    tf.reshape(init_inter, (1, self.adata.n_vars)), 
                    name='intercept')

        if type(self.config.GENE_PRIOR) == list:
            vgenes_temp = []
            prior_t = np.ones((1, ngenes), dtype=np.float32) * 0.5
            for prior in self.config.GENE_PRIOR:
                prior_t[0][prior[2]] = 1.1 if prior[1] == 'increase' else -0.2
                vgenes_temp.append(prior + (prior_t[0][prior[2]], ))
            self.t = tf.Variable(prior_t, name='t')
            print (f'Modified GENE_PRIOR {vgenes_temp} with init_tau')

    def init_pars(self):
        self.default_pars_names = ['gamma', 'beta']
        self.default_pars_names += ['offset', 'a', 't', 'h', 'intercept']

    def init_weights(self, weighted=False):
        nonzero_s, nonzero_u = self.Ms > 0, self.Mu > 0
        weights = np.array(nonzero_s & nonzero_u, dtype=bool)

        if weighted:
            ub_s = np.percentile(self.s[weights], self.perc)
            ub_u = np.percentile(self.u[weights], self.perc)
            if ub_s > 0:
                weights &= np.ravel(self.s <= ub_s)
            if ub_u > 0:
                weights &= np.ravel(self.u <= ub_u)

        self.weights = tf.cast(weights, dtype=tf.float32)
        self.nobs = np.sum(weights, axis=0)

    def init_lr(self):
        if self.config.LEARNING_RATE == None:
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=1e-2,
                decay_steps=2000, 
                decay_rate=0.9,
                staircase=True
            )

        else:
            lr_schedule = self.config.LEARNING_RATE

        return lr_schedule

    def get_fit_s(self, args, t_cell):
        self.fit_s = exp(args[5]) * \
            exp(-exp(args[3]) * square(t_cell - args[4])) + \
            args[2]

        return self.fit_s
    
    def get_s_deri(self, args, t_cell):
        self.s_deri = (self.fit_s - args[2]) * \
            (-exp(args[3]) * 2 * (t_cell - args[4]))

        return self.s_deri

    def get_fit_u(self, args):
        return (self.s_deri + exp(args[0]) * self.fit_s) / exp(args[1]) + args[6]
    
    def get_s_u(self, args, t_cell):
        s = self.get_fit_s(args, t_cell)
        s_deri = self.get_s_deri(args, t_cell)
        u = self.get_fit_u(args)

        if self.config.ASSIGN_POS_U:
            s = tf.clip_by_value(s, 0, 1000)
            u = tf.clip_by_value(u, 0, 1000)

        return s, u

    def max_density(self, dis):
        if self.config.DENSITY == 'Max':
            sort_list = np.sort(dis)
            diff = np.insert(np.diff(sort_list), 0, 0, axis=1)
            kernel = np.ones(self.win_size)

            loc = []
            for row in range(dis.shape[0]):
                idx = np.argmin(np.convolve(diff[row, :], kernel, 'valid'))
                window = sort_list[row, idx:idx + self.win_size]
                loc.append(np.mean(window))

            return np.array(loc, dtype=np.float32)

        if self.config.DENSITY == 'SVD':
            s, u, v = tf.linalg.svd(dis)
            s = s[0:50]
            u = u[:, :tf.size(s)]
            v = v[:tf.size(s), :tf.size(s)]
            dis_approx = tf.matmul(u, tf.matmul(tf.linalg.diag(s), v, adjoint_b=True))
            return tf.cast(mean(dis_approx, axis=1), tf.float32)

        if self.config.DENSITY == 'Raw':
            # weight = self.gene_prior_perc(dis)
            return tf.cast(mean(dis, axis=1), tf.float32)

    def gene_prior_perc(self, dis):
        perc, perc_idx = [], []
        aggregrate_weight = np.ones((dis.shape[1], ), dtype=np.float32)
    
        for prior in self.config.GENE_PRIOR:
            expr = np.array(self.adata[:, prior[0]].layers['Ms'])
            perc.append(np.max(expr) * 0.75) # modify 0.75 for parameter tuning
        
        perc_total = np.sum(perc)
        perc = [perc[i] / perc_total for i in range(len(perc))]

        for sequence, prior in enumerate(self.config.GENE_PRIOR):
            temp_id = np.sum(self.boolean[:prior[2]])
            aggregrate_weight[temp_id] = perc[sequence] * self.config.GENE_PRIOR_SCALE
            perc_idx.append(temp_id)

        # print (f'assigned weights of GENE_PRIOR {list(np.around(np.array(perc), 2)), perc_idx}')
        weight_total = np.sum(aggregrate_weight)
        aggregrate_weight = [aggregrate_weight[i] / weight_total for i in range(len(aggregrate_weight))]
        return np.reshape(aggregrate_weight, (1, dis.shape[1]))

    def match_time(self, Ms, Mu, s_predict, u_predict, x, iter):
        val = x[1, :] - x[0, :]
        cell_time = np.zeros((Ms.shape[0], Ms.shape[2]))

        if (self.config.AGENES_R2 < 1) & (iter > self.agenes_thres):
            self.index_list, self.boolean = np.squeeze(tf.where(self.total_genes)), self.total_genes
        else:
            self.index_list, self.boolean = np.squeeze(tf.where(self.idx)), self.idx

        for index in range(Ms.shape[2]):
            if index in self.index_list:
                sobs = Ms[:, :, index:index + 1] # n * 1 * 1
                uobs = Mu[:, :, index:index + 1]
                spre = s_predict[:, :, index:index + 1] # 1 * 3000 * 1
                upre = u_predict[:, :, index:index + 1]

                u_r2 = square(uobs - upre) # n * 3000 * 1
                s_r2 = square(sobs - spre)
                euclidean = sqrt(u_r2 + s_r2)
                assign_loc = tf.math.argmin(euclidean, axis=1).numpy() # n * 1
                
                if self.config.REORDER_CELL == 'Soft_Reorder':
                    cell_time[:, index:index + 1] = \
                        col_minmax(self.reorder(assign_loc), self.adata.var.index[index])
                if self.config.REORDER_CELL == 'Soft':
                    cell_time[:, index:index + 1] = \
                        col_minmax(assign_loc, self.adata.var.index[index])
                if self.config.REORDER_CELL == 'Hard':
                    cell_time[:, index:index + 1] = \
                        x[0, index:index + 1] + val[index:index + 1] * assign_loc

        if self.config.AGGREGATE_T:
            return self.max_density(cell_time[:, self.index_list]) #! sampling?
        else:
            return cell_time

    def reorder(self, loc):
        new_loc = np.zeros(loc.shape)

        for gid in range(loc.shape[1]):
            ref = sorted([(val, idx) for idx, val in enumerate(loc[:, gid])], 
                            key=lambda x:x[0])

            pre, count, rep = ref[0][0], 0, 0
            for item in ref:
                if item[0] > pre:
                    count += rep
                    rep = 1
                else:
                    rep += 1

                new_loc[item[1], gid] = count
                pre = item[0]
        
        return new_loc

    def init_time(self, boundary, shape=None):
        x = tf.linspace(boundary[0], boundary[1], shape[0])

        try:
            if type(boundary[0]) == int or boundary[0].shape[1] == 1:
                x = tf.reshape(x, (-1, 1))
                x = tf.broadcast_to(x, shape)
            else:
                x = tf.squeeze(x)
        except:
            x = tf.reshape(x, (-1, 1))
            x = tf.broadcast_to(x, shape)

        return tf.cast(x, dtype=tf.float32)
    
    def get_opt_args(self, iter, args):
        remain = iter % 400
        
        if self.config.FIT_OPTION == '1':
            if iter < self.config.MAX_ITER / 2:
                args_to_optimize = [args[2], args[3], args[4], args[5]] \
                    if remain < 200 else [args[0], args[1], args[6]]

            else:
                args_to_optimize = [args[0], args[1], 
                                    args[2], args[3],
                                    args[4], args[5], args[6]]
        
        if self.config.FIT_OPTION == '2':
            if iter < self.config.MAX_ITER / 2:
                args_to_optimize = [args[3], args[5]] \
                    if remain < 200 else [args[0], args[1]]
                    
            else:
                args_to_optimize = [args[0], args[1], args[3], args[5]]
        
        return args_to_optimize
    
    def get_log(self, loss, amplify=False, iter=None):
        self.finite = tf.math.is_finite(loss) # location of weird genes out of 2000
        glog = self.adata.var.iloc[np.squeeze(tf.where(~self.finite))].index.values 
        
        for gene in glog:
            if gene not in self.gene_log:
                logging.info(f'{gene}, iter {iter}')
                self.gene_log.append(gene)
            
            if amplify:
                if self.adata.var.at[gene, 'amplify_genes'] == True:
                    logging.info(f'{gene}, iter {iter}, amplify')
                    self.adata.var.at[gene, 'amplify_genes'] = False
        
        loss = tf.where(self.finite, loss, 0)
        return loss

    def get_loss(self, iter, s_r2, u_r2):
        if iter < self.config.MAX_ITER / 2:
            remain = iter % 400
            loss = s_r2 if remain < 200 else u_r2
        else:
            loss = s_r2 + u_r2 

        loss = self.get_log(loss, amplify=False, iter=iter)
        return loss
    
    def get_stop_cond(self, iter, pre, obj):
        stop_s = tf.zeros(self.Ms.shape[1], tf.bool)
        stop_u = tf.zeros(self.Ms.shape[1], tf.bool)
        remain = iter % 400

        if remain > 1 and remain < 200:
            stop_s = abs(pre - obj) <= abs(pre) * 1e-4
        if remain > 201 and remain < 400:
            stop_u = abs(pre - obj) <= abs(pre) * 1e-4

        return tf.math.logical_and(stop_s, stop_u)

    def get_interim_t(self, t_cell, idx):
        if self.config.AGGREGATE_T:
            return t_cell
        
        else:
            #? modify this later for independent mode
            t_interim = np.zeros(t_cell.shape)

            for i in range(t_cell.shape[1]):
                if idx[i]:
                    temp = np.reshape(t_cell[:, i], (-1, 1))
                    t_interim[:, i] = \
                        np.squeeze(col_minmax(temp, self.adata.var.index[i]))

            t_interim = self.max_density(t_interim)
            t_interim = tf.reshape(t_interim, (-1, 1))
            t_interim = tf.broadcast_to(t_interim, self.adata.shape)
            return t_interim