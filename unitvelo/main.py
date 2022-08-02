from .utils import remove_dir, min_max
from .velocity import Velocity
import scvelo as scv
import os
import numpy as np

def run_model(
    data_path,
    label,
    config_file=None,
):
    """Preparation and pre-processing function of RNA velocity calculation.
    
    Args:
        data_path (str): 
            relative of absolute path of Anndata object.
        label (str): 
            column name in adata.var indicating cell clusters.
        config_file (object): 
            model configuration object, default: None.
        
    Returns:
        adata: 
            modified Anndata object.
    
    """

    if config_file == None:
        print (
            f'Model configuration file not specified.\n'
            f'Default settings with unified-time mode will be used.'
        )
        from .config import Configuration
        config = Configuration()

    else:
        config = config_file

    if config.FIT_OPTION == '1':
        config.DENSITY = 'SVD'
        config.REORDER_CELL = 'Soft_Reorder'
        config.AGGREGATE_T = True
        config.ASSIGN_POS_U = False
        config.REG_LOSS = True

    elif config.FIT_OPTION == '2':
        config.DENSITY = 'Raw'
        config.REORDER_CELL = 'Hard'
        config.AGGREGATE_T = False
        config.AGENES_R2 = 1
    
    else:
        raise ValueError('config.FIT_OPTION is invalid')

    config.MAX_ITER = config.MAX_ITER if config.MAX_ITER > 12000 else 12000

    print ('-------> Model Configuration Settings <-------')
    for ix, item in enumerate(vars(config).items()):
        print ("%s: %s" % item, end=f'\t') if ix % 3 != 0 \
            else print ('\n', "%s: %s" % item, end=f'\t')
    print (f'\n')

    scv.settings.presenter_view = True
    scv.settings.verbosity = 0
    scv.settings.figdir = './figures/'
    scv.settings.file_format_figs = 'png'

    replicates, pre = None, 1e15

    for rep in range(config.NUM_REP):
        adata = scv.read(data_path)

        if config.IROOT == 'gcount':
            adata.obs['gcount'] = np.sum(adata.X.todense() > 0, axis=1)
            init_time = 1 - min_max(adata.obs.groupby(label)['gcount'].mean())

            for id in list(init_time.index):
                adata.obs.loc[adata.obs[label] == id, 'gcount'] = init_time[id]

        adata.uns['datapath'] = data_path
        adata.uns['label'] = label
        adata.uns['base_function'] = config.BASE_FUNCTION
        
        if rep == 1:
            adata.obs['latent_time_gm'] = pre_time_gm

        if 'true_alpha' not in adata.var.columns:
            if config.BASIS is None:
                basis_keys = ["pca", "tsne", "umap"]
                basis = [key for key in basis_keys if f"X_{key}" in adata.obsm.keys()][-1]
            elif f"X_{config.BASIS}" in adata.obsm.keys():
                basis = config.BASIS
            else:
                raise ValueError('Invalid embedding parameter self.BASIS')
            adata.uns['basis'] = basis

        else:
            basis = None

        remove_dir(adata.uns['datapath'], adata)
        scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=config.N_TOP_GENES)
        scv.pp.moments(adata, n_pcs=30, n_neighbors=30)

        import logging
        logging.basicConfig(filename=os.path.join(adata.uns['temp'], 'logging.txt'),
                            filemode='a',
                            format='%(asctime)s, %(levelname)s, %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.INFO)

        if 'scNT' in data_path:
            import pandas as pd
            gene_ids = pd.read_csv('../data/scNT/brie_neuron_splicing_time.tsv', delimiter='\t', index_col='GeneID')
            config.VGENES = list(gene_ids.loc[gene_ids['time_FDR'] < 0.01].index)

        model = Velocity(adata, config=config)
        model.get_velo_genes()

        adata = model.fit_velo_genes(basis=basis, rep=rep)
        pre_time_gm = adata.obs['latent_time'].values

        if config.GENERAL != 'Linear':
            replicates = adata if adata.uns['loss'] < pre else replicates
            pre = adata.uns['loss'] if adata.uns['loss'] < pre else pre

    #? change adata to replicates?
    adata.write(os.path.join(adata.uns['temp'], 'temp.h5ad'))
    
    if 'examine_genes' in adata.uns.keys():
        from .individual_gene import exam_genes
        exam_genes(adata, adata.uns['examine_genes'])

    return replicates if config.GENERAL != 'Linear' else adata