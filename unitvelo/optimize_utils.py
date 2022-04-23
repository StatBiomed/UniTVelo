#%%
import tensorflow as tf
import numpy as np
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
        columns = ['a0', 'vars', 'varu', 'h0', 'gamma', 'beta']
    else:
        columns = ['vars', 'varu', 'gamma', 'beta']
        columns.extend([f'a{k}' for k in range(K)])

    return columns

def is_nan(name=None, data=None, phase=None):
    for id, para in enumerate(name):
        if np.sum(np.isnan(data[id])) > 0:
            print ('nan', para, phase)
            print (np.argwhere(np.isnan(data[id])).shape)
            import sys
            sys.exit()

def is_inf(name=None, data=None, phase=None):
    for id, para in enumerate(name):
        if np.sum(np.isinf(data[id])) > 0:
            print ('inf', para, phase)
            print (np.argwhere(np.isinf(data[id])).shape)
            import sys
            sys.exit()

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
        self.mode = config.BASE_FUNCTION
        self.var_names = var_names
        self.Ms, self.Mu = Ms, Mu
        self.K = 1
        self.win_size = 50
        self.config = config

    def init_vars(self):
        ngenes = len(self.var_names)
        ones = tf.ones((1, ngenes), dtype=tf.float32)

        self.gamma = tf.Variable(ones * 0, name='log_gamma')
        self.beta = tf.Variable(ones * 0, name='log_beta')
        self.vars = tf.Variable(ones * 1, name='vars')
        self.varu = tf.Variable(ones * 1, name='varu')
        self.intercept = tf.Variable(ones * 0, name='intercept')

        if self.mode == 'Gaussian':
            self.t = tf.Variable(ones * 0.5, name='t') #! mean of Gaussian
            self.a = tf.Variable(ones * 0, name='log_a') #! 1 / scaling of Gaussian
            self.offset = tf.Variable(ones * 0, name='offset')

            init_h = log(tf.math.reduce_max(self.Ms, axis=0))
            self.h = tf.Variable(ones * init_h, name='log_h')

            init_gamma = self.adata.var['velocity_gamma'].values
            self.gamma = tf.Variable(log(
                            tf.reshape(init_gamma, (1, self.adata.n_vars))), 
                            name='log_gamma')

            if self.config.OFFSET_GENES:
                init_inter = self.adata.var['velocity_inter'].values
                self.intercept = \
                    tf.Variable(
                        tf.reshape(init_inter, (1, self.adata.n_vars)), 
                        name='intercept')
         
        if self.mode == 'Mixture':
            init_t = np.reshape(np.sort(np.random.uniform(-1, 1, self.K)), (-1, 1))
            init_h = np.reshape(2 * np.array([k % 2 for k in range(self.K)]) - 1, (-1, 1))

            self.t = tf.Variable(tf.ones((self.K, ngenes)) * init_t, name='t')
            self.a = tf.Variable(tf.zeros((self.K, ngenes)), name='log_a')
            self.h = tf.Variable(tf.ones((self.K, ngenes)) * init_h * 7, name='h')
            self.offset = tf.Variable(tf.zeros((self.K, ngenes)), name='offset')

    def init_pars(self):
        self.default_pars_names = ['gamma', 'beta', 'vars', 'varu']
        self.default_pars_names += [f'offset{k}' for k in range(self.K)]
        self.default_pars_names += [f'a{k}' for k in range(self.K)]
        self.default_pars_names += [f't{k}' for k in range(self.K)]
        self.default_pars_names += [f'h{k}' for k in range(self.K)]
        self.default_pars_names += ['intercept']

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
        if self.mode == 'Gaussian':
            self.fit_s = exp(args[7]) * \
                exp(-exp(args[5]) * square(t_cell - args[6])) + \
                args[4]
        
        if self.mode == 'Mixture':
            fit_s = args[7][0, :] * \
                exp(-exp(args[5][0, :]) * square(t_cell - args[6][0, :])) + \
                args[4][0, :]
            fit_s = tf.expand_dims(fit_s, axis=0)

            for k in range(1, self.K):
                temp = args[7][k, :] * \
                    exp(-exp(args[5][k, :]) * square(t_cell - args[6][k, :])) + \
                    args[4][k, :]
                temp = tf.expand_dims(temp, axis=0)
                fit_s = tf.concat([fit_s, temp], axis=0)

            self.ori_s = fit_s
            self.fit_s = sum(fit_s, axis=0)

        return self.fit_s
    
    def get_s_deri(self, args, t_cell):
        if self.mode == 'Gaussian':
            self.s_deri = (self.fit_s - args[4]) * \
                (-exp(args[5]) * 2 * (t_cell - args[6]))
        
        if self.mode == 'Mixture':
            s_deri = (self.ori_s[0, :, :] - args[4][0, :]) * \
                (-exp(args[5][0, :]) * 2 * (t_cell - args[6][0, :]))
            s_deri = tf.expand_dims(s_deri, axis=0)

            for k in range(1, self.K):
                temp_deri = (self.ori_s[k, :, :] - args[4][k, :]) * \
                    (-exp(args[5][k, :]) * 2 * (t_cell - args[6][k, :]))
                temp_deri = tf.expand_dims(temp_deri, axis=0)
                s_deri = tf.concat([s_deri, temp_deri], axis=0)
            
            self.s_deri = sum(s_deri, axis=0)

        return self.s_deri

    def get_fit_u(self, args):
        return (self.s_deri + exp(args[0]) * self.fit_s) / exp(args[1]) + args[8]
    
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
            return tf.cast(mean(dis, axis=1), tf.float32)

    def match_time(self, Ms, Mu, s_predict, u_predict, x):
        val = x[1, :] - x[0, :]
        batch = 1
        cell_time = np.zeros((Ms.shape[0], Ms.shape[2]))

        for index in range(int(np.ceil(Ms.shape[2] / batch))):
            begin, end = batch * index, batch * (index + 1)
            sobs = Ms[:, :, begin:end] # n * 1 * batch
            uobs = Mu[:, :, begin:end]
            spre = s_predict[:, :, begin:end] # 1 * 3000 * batch
            upre = u_predict[:, :, begin:end]

            left = 0
            if False:
                if tf.where(upre < 0).shape[0] > 0:
                    cutoff = tf.where(upre < 0)[0][1]
                    
                    if cutoff > 0:
                        upre = upre[:, :cutoff, :]
                        spre = spre[:, :cutoff, :]

                    if cutoff == 0:
                        left = tf.where(upre >= 0)[0][1]
                        right = tf.where(upre >= 0)[-1][1]
                        upre = upre[:, left:right, :]
                        spre = spre[:, left:right, :]

            u_r2 = square(uobs - upre) # n * 3000 * batch
            s_r2 = square(sobs - spre)
            euclidean = sqrt(u_r2 + s_r2)
            assign_loc = tf.math.argmin(euclidean, axis=1).numpy() # n * batch
            
            if self.config.REORDER_CELL == 'Soft_Reorder':
                cell_time[:, begin:end] = \
                    col_minmax(self.reorder(assign_loc), self.adata.var.index[index])
            if self.config.REORDER_CELL == 'Soft':
                cell_time[:, begin:end] = \
                    col_minmax(assign_loc, self.adata.var.index[index])
            if self.config.REORDER_CELL == 'Hard':
                cell_time[:, begin:end] = x[left, begin:end] + val[begin:end] * assign_loc

        if self.config.AGGREGATE_T:
            return self.max_density(cell_time) #! sampling?
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
        if self.mode == 'Mixture':
            boundary = (0, 1)

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
                args_to_optimize = [args[4], args[5], args[6], args[7]] \
                    if remain < 200 else [args[0], args[1], args[8]]

            else:
                args_to_optimize = [args[0], args[1], 
                                    args[4], args[5],
                                    args[6], args[7], args[8]]
        
        if self.config.FIT_OPTION == '2':
            if iter < self.config.MAX_ITER / 2:
                args_to_optimize = [args[5], args[7]] \
                    if remain < 200 else [args[0], args[1]]
                    
            else:
                args_to_optimize = [args[0], args[1], args[5], args[7]]
        
        return args_to_optimize
    
    def get_loss(self, iter, s_r2, u_r2):
        if iter < self.config.MAX_ITER / 2:
            remain = iter % 400
            loss = s_r2 if remain < 200 else u_r2
        else:
            loss = s_r2 + u_r2 
        
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

    def get_interim_t(self, t_cell):
        if self.config.AGGREGATE_T:
            return t_cell
        
        else:
            t_interim = np.zeros(t_cell.shape)

            for i in range(t_cell.shape[1]):
                temp = np.reshape(t_cell[:, i], (-1, 1))
                t_interim[:, i] = \
                    np.squeeze(col_minmax(temp, self.adata.var.index[i]))

            t_interim = self.max_density(t_interim)
            t_interim = tf.reshape(t_interim, (-1, 1))
            t_interim = tf.broadcast_to(t_interim, self.adata.shape)
            return t_interim