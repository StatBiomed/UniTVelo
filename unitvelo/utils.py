import numpy as np
import pandas as pd
import os
np.random.seed(42)

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

def new_adata_col(adata, idx, var_names, values):
    for i, name in enumerate(var_names):
        adata.var[name] = np.zeros(adata.n_vars) * np.nan
        adata.var[name][idx] = values[i]

def get_cycle_gene(adata):
    from .utils import get_cgene_list
    s_genes_list, g2m_genes_list = get_cgene_list()

    var = adata.var.loc[adata.var['velocity_genes'] == True]
    phase_s, phase_g2 = [], []

    for gene in var.index:
        if gene in s_genes_list:
            phase_s.append(gene)
        if gene in g2m_genes_list:
            phase_g2.append(gene)
    
    return phase_s, phase_g2

def col_corrcoef(raw, fit):
    res = []
    for col in range(raw.shape[1]):
        corr = np.corrcoef(min_max(raw[:, col]), min_max(fit[:, col]))[0][1]
        res.append(corr)
    return np.array(res)

def col_spearman(raw, fit):
    from scipy.stats import spearmanr

    res = []
    for col in range(raw.shape[1]):
        results, _ = spearmanr(min_max(raw[:, col]), min_max(fit[:, col]))
        res.append(results)
    return np.array(res)

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
    K=1
):
    from .optimize_utils import exp_args

    s = pd.DataFrame(data=fits, index=adata.obs.index, columns=adata.var.index)
    u = pd.DataFrame(data=fitu, index=adata.obs.index, columns=adata.var.index)
    ms = pd.DataFrame(data=adata.layers['Ms'], index=adata.obs.index, columns=adata.var.index)
    mu = pd.DataFrame(data=adata.layers['Mu_scale'], index=adata.obs.index, columns=adata.var.index)
    s['label'] = adata.obs[adata.uns['label']].values

    if adata.var.index[0].startswith('ENSMUSG'):
        adata.var.index = adata.var['gene']
        adata.var.index.name = 'index' 

    var = pd.DataFrame(data=np.zeros((adata.shape[1],)), index=adata.var.index)
    del var[0]

    pars = []
    for i in range(len(args)):
        if args[i].shape[0] > 1:
            for k in range(K):
                par = np.zeros(adata.n_vars) * np.nan
                par = args[i][k, :].numpy()
                pars.append(par)
        else:
            par = np.zeros(adata.n_vars) * np.nan
            par = args[i].numpy()
            pars.append(par)
    
    for i, name in enumerate(adata.uns['par_names']):
        var[name] = np.transpose(pars[i])
    
    columns = exp_args(adata, K=K)
    for col in columns:
        var[col] = np.exp(var[col])
    
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
        'fit_loss', 'fit_bic', 'fit_llf',
        'fit_gamma', 'fit_beta', 'fit_vars', 'fit_varu', 
        'fit_a0', 'fit_t0', 'fit_h0', 
        'li_coef', 'li_loss', 'li_r2', 
        'li_llf', 'li_bic'
    ]]
    return var