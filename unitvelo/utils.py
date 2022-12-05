import numpy as np
import pandas as pd
import os
np.random.seed(42)
import logging
import scvelo as scv

def get_cgene_list():
    s_genes_list = \
        ['Mcm5', 'Pcna', 'Tyms', 'Fen1', 'Mcm2', 'Mcm4', 'Rrm1', 'Ung', 'Gins2',
        'Mcm6', 'Cdca7', 'Dtl', 'Prim1', 'Uhrf1', 'Mlf1ip', 'Hells', 'Rfc2',
        'Rpa2', 'Nasp', 'Rad51ap1', 'Gmnn', 'Wdr76', 'Slbp', 'Ccne2', 'Ubr7',
        'Pold3', 'Msh2', 'Atad2', 'Rad51', 'Rrm2', 'Cdc45', 'Cdc6', 'Exo1', 'Tipin',
        'Dscc1', 'Blm', 'Casp8ap2', 'Usp1', 'Clspn', 'Pola1', 'Chaf1b', 'Brip1', 'E2f8']

    g2m_genes_list = \
        ['Hmgb2', 'Cdk1', 'Nusap1', 'Ube2c', 'Birc5', 'Tpx2', 'Top2a', 'Ndc80',
        'Cks2', 'Nuf2', 'Cks1b', 'Mki67', 'Tmpo', 'Cenpf', 'Tacc3', 'Fam64a',
        'Smc4', 'Ccnb2', 'Ckap2l', 'Ckap2', 'Aurkb', 'Bub1', 'Kif11', 'Anp32e',
        'Tubb4b', 'Gtse1', 'Kif20b', 'Hjurp', 'Cdca3', 'Hn1', 'Cdc20', 'Ttk',
        'Cdc25c', 'Kif2c', 'Rangap1', 'Ncapd2', 'Dlgap5', 'Cdca2', 'Cdca8',
        'Ect2', 'Kif23', 'Hmmr', 'Aurka', 'Psrc1', 'Anln', 'Lbr', 'Ckap5',
        'Cenpe', 'Ctcf', 'Nek2', 'G2e3', 'Gas2l3', 'Cbx5', 'Cenpa']

    return s_genes_list, g2m_genes_list

def new_adata_col(adata, var_names, values):
    for i, name in enumerate(var_names):
        # adata.var[name] = np.zeros(adata.n_vars) * np.nan
        adata.var[name] = values[i]

def get_cycle_gene(adata):
    from .utils import get_cgene_list
    s_genes_list, g2m_genes_list = get_cgene_list()

    # var = adata.var.loc[adata.var['velocity_genes'] == True]
    phase_s, phase_g2 = [], []

    for gene in adata.var.index:
        if gene in s_genes_list:
            phase_s.append(gene)
        if gene in g2m_genes_list:
            phase_g2.append(gene)
    
    return phase_s, phase_g2

def col_corrcoef(raw, fit):
    from .utils import min_max

    res = []
    for col in range(raw.shape[1]):
        corr = np.corrcoef(min_max(raw[:, col]), min_max(fit[:, col]))[0][1]
        res.append(corr)
        
    return np.array(res)

def col_spearman(raw, fit):
    from scipy.stats import spearmanr
    from .utils import min_max

    res = []
    for col in range(raw.shape[1]):
        results, _ = spearmanr(min_max(raw[:, col]), min_max(fit[:, col]))
        res.append(results)

    return np.array(res)

def col_spearman_unorm(raw, fit):
    from scipy.stats import spearmanr

    res = []
    for col in range(raw.shape[1]):
        results, _ = spearmanr(raw[:, col], fit[:, col])
        res.append(results)

    return np.array(res)

def gene_level_spearman(adata, iroot=None, n_top_genes=2000, file_name=None):
    """
    Spearman correlation between the expression profiles fitness 
    of the top-down (scVelo Dynamical mode) or 
    bottom-up strategies (UniTVelo Independent mode)
    Could serve as a systematic comparison (e.g. the entire transcriptome) 
    between two methods    
    """
    
    import scvelo as scv
    import scanpy as sc
    import numpy as np
    import pandas as pd
    from .utils import col_spearman
    from scipy.stats import spearmanr

    print ('---> Running scVelo dynamical mode')
    scvdata = scv.read(adata.uns['datapath'])
    scv.pp.filter_and_normalize(scvdata, min_shared_counts=20, n_top_genes=n_top_genes)
    scv.pp.moments(scvdata, n_pcs=30, n_neighbors=30)
    scv.tl.recover_dynamics(scvdata, n_jobs=20)
    scv.tl.velocity(scvdata, mode='dynamical')
    scv.tl.latent_time(scvdata)

    print ('---> Caculating spearman correlation with reference')
    if 'organoids' in adata.uns['datapath']:
        scvdata.var.index = scvdata.var['gene'].values

    index = adata.var.loc[adata.var['velocity_genes'] == True].index.intersection(scvdata.var.loc[scvdata.var['velocity_genes'] == True].index)
    tadata = adata[:, index].layers['fit_t']
    tscv = scvdata[:, index].layers['fit_t']

    if iroot != None:
        print ('---> Generating reference with diffusion pseudotime')
        sc.tl.diffmap(adata)
        adata.uns['iroot'] = np.flatnonzero(adata.obs[adata.uns['label']] == iroot)[0]
        sc.tl.dpt(adata)

        gt = adata.obs['dpt_pseudotime'].values
        gt = np.broadcast_to(np.reshape(gt, (-1, 1)), tadata.shape)
        resadata = col_spearman(gt, tadata)
        resscv = col_spearman(gt, tscv)

        results, _ = spearmanr(min_max(scvdata.obs['latent_time']), min_max(gt))
        print (np.reshape(results, (-1, 1)))

    if file_name != None:
        print ('---> Generating reference pseudotime with Slingshot')
        slingshot = pd.read_csv(f'./slingshot/{file_name}')
        slingshot = slingshot['x'].values

        slingshot = np.broadcast_to(np.reshape(slingshot, (-1, 1)), tadata.shape)
        resadata = col_spearman(slingshot, tadata)
        resscv = col_spearman(slingshot, tscv)

        results, _ = spearmanr(min_max(scvdata.obs['latent_time']), min_max(slingshot[:, 0]))
        print (np.reshape(results, (-1, 1)))

    corr = pd.DataFrame(index=index, columns=['UniTVelo', 'scVelo'])
    corr['UniTVelo'] = np.reshape(resadata, (-1, 1))
    corr['scVelo'] = np.reshape(resscv, (-1, 1))

    return corr['UniTVelo'].values, corr['scVelo'].values, corr

def gene_level_comparison(corr):
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.boxplot(data=corr)
    plt.title('Comparison of gene-specific time', fontsize=12)
    plt.ylabel('Spearman Correlation', fontsize=12)
    plt.show()
    plt.close()

    sns.scatterplot(x='UniTVelo', y='scVelo', data=corr)
    plt.axline((0, 0), (0.5, 0.5), color='r')
    plt.show()
    plt.close()

def col_mse(raw, fit):
    from sklearn.metrics import mean_squared_error

    res = []
    for col in range(raw.shape[1]):
        results = mean_squared_error(raw[:, col], fit[:, col])
        res.append(results)
    return np.array(res)

def col_minmax(matrix):
    return (matrix - np.min(matrix, axis=0)) \
        / (np.max(matrix, axis=0) - np.min(matrix, axis=0))

def inv_prob(obs, fit):
    temp = np.abs(obs - fit)
    temp = np.log(np.sum(temp, axis=0))
    temp = np.exp(-temp)
    return temp / np.sum(temp)

def remove_dir(data_path, adata):
    import shutil
    dir = os.path.split(data_path)[0]
    filename = os.path.splitext(os.path.basename(data_path))[0]

    NEW_DIR = os.path.join(dir, filename)
    adata.uns['temp'] = NEW_DIR
    
    if os.path.exists(NEW_DIR):
        shutil.rmtree(NEW_DIR)
    os.mkdir(NEW_DIR)

def save_vars(
    adata, 
    args, 
    fits, 
    fitu, 
    K=1,
    scaling=1
):
    from .optimize_utils import exp_args

    s = pd.DataFrame(data=fits, index=adata.obs.index, columns=adata.var.index)
    u = pd.DataFrame(data=fitu, index=adata.obs.index, columns=adata.var.index)
    ms = pd.DataFrame(data=adata.layers['Ms'], index=adata.obs.index, columns=adata.var.index)
    mu = pd.DataFrame(data=adata.layers['Mu'], index=adata.obs.index, columns=adata.var.index)
    s['label'] = adata.obs[adata.uns['label']].values

    if adata.var.index[0].startswith('ENSMUSG'):
        adata.var.index = adata.var['gene']
        adata.var.index.name = 'index' 

    var = pd.DataFrame(data=np.zeros((adata.shape[1], )), index=adata.var.index)
    del var[0]

    pars = []
    for i in range(len(args)):
        if args[i].shape[0] > 1:
            for k in range(K):
                # par = np.zeros(adata.n_vars) * np.nan
                par = args[i][k, :].numpy()
                pars.append(par)
        else:
            # par = np.zeros(adata.n_vars) * np.nan
            par = args[i].numpy()
            pars.append(par)
    
    for i, name in enumerate(adata.uns['par_names']):
        var[name] = np.transpose(pars[i])
    
    columns = exp_args(adata, K=K)
    for col in columns:
        var[col] = np.exp(var[col])
    
    var['beta'] /= scaling
    var['intercept'] *= scaling
    
    NEW_DIR = adata.uns['temp']

    s.to_csv(f'{NEW_DIR}/fits.csv')
    u.to_csv(f'{NEW_DIR}/fitu.csv')
    var.to_csv(f'{NEW_DIR}/fitvar.csv')
    ms.to_csv(f'{NEW_DIR}/Ms.csv')
    mu.to_csv(f'{NEW_DIR}/Mu.csv')

def min_max(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def make_dense(X):
    from scipy.sparse import issparse

    XA = X.A if issparse(X) and X.ndim == 2 else X.A1 if issparse(X) else X
    if XA.ndim == 2:
        XA = XA[0] if XA.shape[0] == 1 else XA[:, 0] if XA.shape[1] == 1 else XA
    return np.array(XA)

def get_weight(x, y=None, perc=95):
    from scipy.sparse import issparse

    xy_norm = np.array(x.A if issparse(x) else x)
    if y is not None:
        if issparse(y):
            y = y.A
        xy_norm = xy_norm / np.clip(np.max(xy_norm, axis=0), 1e-3, None)
        xy_norm += y / np.clip(np.max(y, axis=0), 1e-3, None)

    if isinstance(perc, int):
        weights = xy_norm >= np.percentile(xy_norm, perc, axis=0)
    else:
        lb, ub = np.percentile(xy_norm, perc, axis=0)
        weights = (xy_norm <= lb) | (xy_norm >= ub)

    return weights

def R2(residual, total):
    r2 = np.ones(residual.shape[1]) - \
        np.sum(residual * residual, axis=0) / \
            np.sum(total * total, axis=0)
    r2[np.isnan(r2)] = 0
    return r2

def OLS(x, y):
    mean_x, mean_y = np.mean(x, axis=0), np.mean(y, axis=0)
    numerator = np.sum(x * y - mean_y * x, axis=0)
    denominator = np.sum(x ** 2 - mean_x * x, axis=0)

    coef_ = numerator / denominator
    inter_ = mean_y - coef_ * mean_x
    return coef_, inter_

def get_model_para(adata):
    var = adata.var.loc[adata.var['velocity_genes'] == True]
    var = var[[
        'scaling', 'fit_vars',
        'fit_varu', 'fit_gamma', 'fit_beta', 'fit_offset0', 
        'fit_a0', 'fit_t0', 'fit_h0', 
        'fit_intercept', 'fit_loss', 'fit_bic', 
        'fit_llf', 'fit_sr2', 'fit_ur2'
    ]]

    if 're_transit' in adata.var.columns:
        var[0].extend(['qua_r2', 'rbf_r2', 're_transit'])

    return var

def reverse_transient(adata, time_metric='latent_time'):
    from scipy.optimize import curve_fit
    import numpy as np
    from tqdm.notebook import tqdm

    adata.var['re_transit'] = False
    adata.var['qua_r2'] = -1.
    adata.var['rbf_r2'] = -1.
    sigma_max = np.max(adata.var.loc[adata.var['velocity_genes'] == True]['fit_a0'])
    celltime = adata.obs[time_metric].values
    
    def quadratic(x, a, b, c):
        return a * (x ** 2) + b * x + c

    def rbf(x, h, sigma, tau):
        return h * np.exp(-sigma * (x - tau) * (x - tau))

    for index, row in tqdm(adata.var.iterrows()):
        if row['velocity_genes']:
            spliced = np.squeeze(np.array(adata[:, index].layers['Ms']))
            popt, _ = curve_fit(quadratic, celltime, spliced)

            fitted = quadratic(celltime, popt[0], popt[1], popt[2])
            ss_res = np.sum((spliced - fitted) ** 2)
            ss_tot = np.sum((spliced - np.mean(spliced)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            adata.var.loc[index, 'qua_r2'] = r2

            try:
                popt_rbf, _ = curve_fit(rbf, celltime, spliced, 
                    maxfev=10000, 
                    bounds=([1e-2, 1e-3, -np.inf], 
                            [np.max(spliced), sigma_max, +np.inf])
                )

                fitted_rbf = rbf(celltime, popt_rbf[0], popt_rbf[1], popt_rbf[2])
                ss_res_rbf = np.sum((spliced - fitted_rbf) ** 2)
                ss_tot_rbf = np.sum((spliced - np.mean(spliced)) ** 2)
                r2_rbf= 1 - (ss_res_rbf / ss_tot_rbf)            
                adata.var.loc[index, 'rbf_r2'] = r2_rbf
            
            except:
                r2_rbf = 0
                adata.var.loc[index, 'rbf_r2'] = r2_rbf

            if r2 - r2_rbf > 0.075:
                adata.var.loc[index, 're_transit'] = True
    
    from .pl import plot_reverse_tran_scatter
    plot_reverse_tran_scatter(adata)

    re_tran_num = adata.var.loc[adata.var['re_transit'] == True].shape[0]
    re_tran_perc = re_tran_num / adata.var.loc[adata.var['velocity_genes'] == True].shape[0]
    logging.info(f'# of genes which are identified as reverse transient {re_tran_num}')
    logging.info(f'percentage of genes which are identified as reverse transient {re_tran_perc}')
    
    return adata

def choose_mode(adata, label=None):
    print ('This function works as a reference only.')
    print ('For less certain scenario, we also suggest users to try both.')
    print ('---> Checking cell cycle scores...')

    from .utils import get_cgene_list
    import scvelo as scv
    scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)

    s, g2m = get_cgene_list()
    num_s = len(adata.var.index.intersection(s))
    num_g2m = len(adata.var.index.intersection(g2m))

    print (f'---> Number of S genes {num_s}/{len(s)}')
    print (f'---> Number of G2M genes {num_g2m}/{len(g2m)}')

    if (num_s / len(s) > 0.5) or (num_g2m / len(g2m) > 0.5):
        print ('Independent mode is recommended, consider setting config.FIT_OPTION = 2')

    else:
        print ('# of cycle genes failed to pass thresholds')
        print ('---> Checking sparse cell types...')
        scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
        adata.obs['cid'] = list(range(adata.shape[0]))

        try:
            neighbors = adata.uns['neighbors']['indices']
        except:
            scv.pp.neighbors(adata, n_pcs=30, n_neighbors=30)
            neighbors = adata.uns['neighbors']['indices']

        ctype_perc = []
        ctype = list(set(adata.obs[label].values))

        for type in ctype:
            temp = adata[adata.obs.loc[adata.obs[label] == type].index, :]
            temp_id = temp.obs['cid'].values
            temp_nei = neighbors[temp_id, 1:].flatten()

            temp_nei = [True if nei in temp_id else False for nei in temp_nei]
            ctype_perc.append(np.sum(temp_nei) / len(temp_nei))
        
        if np.sum(np.array(ctype_perc) > 0.95) >= 3:
            print ('More than two sparse cell types have been detected')
            print ('Independent mode is recommended, consider setting config.FIT_OPTION = 2')
        else:
            print ('Unified-time mode is recommended, consider setting config.FIT_OPTION = 1')

def subset_adata(adata, label=None, proportion=0.5, min_cells=50):
    adata.obs['cid'] = list(range(adata.shape[0]))
    ctype = list(set(adata.obs[label].values))

    subset = []

    for type in ctype:
        temp = adata[adata.obs.loc[adata.obs[label] == type].index, :]
        temp_id = temp.obs['cid'].values

        if len(temp_id) <= min_cells:
            subset.extend(list(temp_id))
        elif int(temp.shape[0] * proportion) <= min_cells:
            subset.extend(list(np.random.choice(temp_id, size=min_cells, replace=False)))
        else:
            subset.extend(list(np.random.choice(temp_id, size=int(temp.shape[0] * proportion), replace=False)))
        
    return adata[np.array(subset), :]

def subset_prediction(adata_subset, adata, config=None):
    from .optimize_utils import Model_Utils

    model = Model_Utils(adata, config=config)
    x = model.init_time((0, 1), (3000, adata.n_vars))
    
    adata.var = adata_subset.var.copy()
    adata.uns['basis'] = adata_subset.uns['basis']
    adata.uns['label'] = adata_subset.uns['label']
    adata.uns['par_names'] = adata_subset.uns['par_names']
    adata.uns['base_function'] = adata_subset.uns['base_function']

    model.total_genes = adata.var['velocity_genes'].values
    model.idx = adata.var['velocity_genes'].values
    scaling = adata.var['scaling'].values

    args_shape = (1, adata.n_vars)
    args = [
        np.broadcast_to(np.log(np.array(adata.var['fit_gamma'].values)), args_shape), 
        np.broadcast_to(np.log(np.array(adata.var['fit_beta'].values * scaling)), args_shape), 
        np.broadcast_to(np.array(adata.var['fit_offset0'].values), args_shape), 
        np.broadcast_to(np.log(np.array(adata.var['fit_a0'].values)), args_shape), 
        np.broadcast_to(np.array(adata.var['fit_t0'].values), args_shape), 
        np.broadcast_to(np.log(np.array(adata.var['fit_h0'].values)), args_shape), 
        np.broadcast_to(np.array(adata.var['fit_intercept'].values / scaling), args_shape)
    ]

    s_predict, s_deri_predict, u_predict = \
        model.get_fit_s(args, x), model.get_s_deri(args, x), model.get_fit_u(x)
    s_predict = tf.expand_dims(s_predict, axis=0) # 1 3000 d
    u_predict = tf.expand_dims(u_predict, axis=0)
    Mu = tf.expand_dims(adata.layers['Mu'] / scaling, axis=1) # n 1 d
    Ms = tf.expand_dims(adata.layers['Ms'], axis=1)

    t_cell = model.match_time(Ms, Mu, s_predict, u_predict, x.numpy(), config.MAX_ITER)
    t_cell = np.reshape(t_cell, (-1, 1))
    t_cell = np.broadcast_to(t_cell, adata.shape)

    adata.layers['fit_t'] = t_cell.copy()
    adata.layers['fit_t'][:, ~adata.var['velocity_genes'].values] = np.nan
    
    model.fit_s = model.get_fit_s(args, t_cell).numpy()
    model.s_deri = model.get_s_deri(args, t_cell).numpy()
    model.fit_u = model.get_fit_u(args).numpy()
    adata.layers['velocity'] = model.s_deri
    
    adata.obs['latent_time_gm'] = min_max(np.nanmean(adata.layers['fit_t'], axis=1))
    scv.tl.velocity_graph(adata, sqrt_transform=True)
    scv.tl.velocity_embedding(adata, basis=adata.uns['basis'])
    scv.tl.latent_time(adata, min_likelihood=None)
        
    if config.FIT_OPTION == '1':
        adata.obs['latent_time'] = adata.obs['latent_time_gm']
        del adata.obs['latent_time_gm']
    
    if os.path.exists(os.path.join(adata_subset.uns['temp'], 'prediction')):
        pass
    else: os.mkdir(os.path.join(adata_subset.uns['temp'], 'prediction'))
    adata.uns['temp'] = os.path.join(adata_subset.uns['temp'], 'prediction')

    import shutil
    shutil.copyfile(os.path.join(adata_subset.uns['temp'], 'fitvar.csv'), 
                    os.path.join(adata.uns['temp'], 'fitvar.csv'))
    
    import pandas as pd
    s = pd.DataFrame(data=model.fit_s, index=adata.obs.index, columns=adata.var.index)
    u = pd.DataFrame(data=model.fit_u, index=adata.obs.index, columns=adata.var.index)
    ms = pd.DataFrame(data=adata.layers['Ms'], index=adata.obs.index, columns=adata.var.index)
    mu = pd.DataFrame(data=adata.layers['Mu'], index=adata.obs.index, columns=adata.var.index)
    s['label'] = adata.obs[adata.uns['label']].values

    NEW_DIR = adata.uns['temp']
    s.to_csv(f'{NEW_DIR}/fits.csv')
    u.to_csv(f'{NEW_DIR}/fitu.csv')
    ms.to_csv(f'{NEW_DIR}/Ms.csv')
    mu.to_csv(f'{NEW_DIR}/Mu.csv')
    adata.write(os.path.join(NEW_DIR, f'predict_adata.h5ad'))

    return adata

def prior_trend_valid(adata, gene_list=None, name='IROOT'):
    import sys
    vgenes_temp = []
    vgenes = adata.var.loc[adata.var['velocity_genes'] == True].index

    file_path = os.path.join(adata.uns['temp'], 'vgenes.txt')
    with open(file_path, 'w') as fp:
        for item in vgenes:
            fp.write("%s\n" % item)

    for prior in gene_list:
        if prior[0] not in vgenes:
            print (f'{prior[0]} of {name} has not been identified as a velocity gene')
        else:
            vgenes_temp.append(prior + (list(adata.var.index).index(prior[0]), ))
    
    if len(vgenes_temp) == 0:
        print (f'---> No genes has been identified as velocity genes')
        print (f'Consider selecting one from {file_path}')
        sys.exit()
    else:
        print (f'Modified {name} {vgenes_temp} with index')
        return vgenes_temp

def init_config_summary(config=None):
    from .config import Configuration
    if config == None:
        print (f'Model configuration file not specified. Default settings with unified-time mode will be used.')
        config = Configuration()

    if config.FIT_OPTION == '1':
        config.DENSITY = 'SVD' if config.GENE_PRIOR == None else 'Raw'
        config.REORDER_CELL = 'Soft_Reorder'
        config.AGGREGATE_T = True

    elif config.FIT_OPTION == '2':
        config.DENSITY = 'Raw'
        config.REORDER_CELL = 'Hard'
        config.AGGREGATE_T = False

    else:
        raise ValueError('config.FIT_OPTION is invalid')

    print ('------> Manully Specified Parameters <------')
    config_ref = Configuration()
    dict_input, dict_ref = vars(config), vars(config_ref)

    para_used = []
    for parameter in dict_ref:
        if dict_input[parameter] != dict_ref[parameter]:
            print (parameter, dict_input[parameter], sep=f':\t')
            para_used.append(parameter)

    print ('------> Model Configuration Settings <------')
    default_para = ['N_TOP_GENES', 
                    'LEARNING_RATE', 
                    'FIT_OPTION', 
                    'DENSITY', 
                    'REORDER_CELL', 
                    'AGGREGATE_T', 
                    'R2_ADJUST', 
                    'GENE_PRIOR', 
                    'VGENES', 
                    'IROOT']

    for parameter in default_para:
        if parameter not in para_used:
            print (parameter, dict_ref[parameter], sep=f':\t')
    
    print ('--------------------------------------------')
    print ('')
    return config, para_used

def init_adata_and_logs(adata, config, normalize=True):
    if type(adata) == str:
        data_path = adata
        adata = scv.read(data_path)

    else:
        cwd = os.getcwd()
        if os.path.exists(os.path.join(cwd, 'res')):
            pass
        else: os.mkdir(os.path.join(cwd, 'res'))

        print (f'Current working dir is {cwd}.')
        print (f'Results will be stored in res folder')
        data_path = os.path.join(cwd, 'res', 'temp.h5ad')
    
    from .utils import remove_dir
    remove_dir(data_path, adata)
    logging.basicConfig(filename=os.path.join(adata.uns['temp'], 'logging.txt'),
                        filemode='a',
                        format='%(asctime)s, %(levelname)s, %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)

    if normalize:
        scv.pp.filter_and_normalize(adata, 
                                    min_shared_counts=config.MIN_SHARED_COUNTS, 
                                    n_top_genes=config.N_TOP_GENES)
        print (f"Extracted {adata.var[adata.var['highly_variable'] == True].shape[0]} highly variable genes.")

        print (f'Computing moments for {len(adata.var)} genes with n_neighbors: {config.N_NEIGHBORS} and n_pcs: {config.N_PCS}')
        scv.pp.moments(adata, 
                        n_pcs=config.N_PCS, 
                        n_neighbors=config.N_NEIGHBORS)
    else:
        scv.pp.neighbors(adata)

    print ('')
    return adata, data_path