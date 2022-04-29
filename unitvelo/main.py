from .utils import remove_dir
from .velocity import Velocity
import scvelo as scv
import os

def run_model(
    data_path,
    label,
    config_file=None,
    gene_ids=None
):
    """Preparation and pre-processing function of RNA velocity calculation.
    
    Args:
        data_path (str): 
            relative of absolute path of Anndata object.
        label (str): 
            column name in adata.var indicating cell clusters.
        config_file (object): 
            model configuration object, default: None.
        gene_ids (list): 
            List of user defined genes used for velocity calculation, default: None.
        
    Returns:
        adata: 
            modified Anndata object.
    
    """

    if config_file == None:
        from config import Configuration
        config = Configuration()
    else:
        config = config_file

    if config.FIT_OPTION == '1':
        config.DENSITY = 'SVD'
        config.REORDER_CELL = 'Soft_Reorder'
        config.AGGREGATE_T = True
        config.ASSIGN_POS_U = False
        config.REG_LOSS = True

    if config.FIT_OPTION == '2':
        config.DENSITY = 'Raw'
        config.REORDER_CELL = 'Hard'
        config.AGGREGATE_T = False
        # config.ASSIGN_POS_U = True

    # if config_file != None and config.FIT_OPTION == '2':
        # config.ASSIGN_POS_U = config.ASSIGN_POS_U == config_file.ASSIGN_POS_U

    config.NUM_REPEAT = config.NUM_REPEAT if config.IROOT == None else 1
    config.MAX_ITER = config.MAX_ITER if config.MAX_ITER > 10000 else 10000

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

    for rep in range(config.NUM_REPEAT):
        adata = scv.read(data_path)

        adata.uns['datapath'] = data_path
        adata.uns['label'] = label
        adata.uns['base_function'] = config.BASE_FUNCTION
        # adata.uns['config'] = config

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

        if gene_ids is None:
            if 'scNT' in data_path:
                import pandas as pd
                gene_ids = pd.read_csv('../data/scNT/brie_neuron_splicing_time.tsv', delimiter='\t', index_col='GeneID')
                gene_ids = list(gene_ids.loc[gene_ids['time_FDR'] < 0.01].index)

        model = Velocity(adata, config=config)
        model.get_velo_genes(gene_ids=gene_ids)

        adata = model.fit_velo_genes(basis=basis, rep=rep)

        if config.GENERAL != 'Linear':
            replicates = adata if adata.uns['loss'] < pre else replicates
            pre = adata.uns['loss'] if adata.uns['loss'] < pre else pre

    if False:
        scv.tl.velocity_confidence(adata)
        keys = 'velocity_length', 'velocity_confidence'
        scv.pl.scatter(adata, basis=basis, c=keys, cmap='coolwarm', perc=[5, 95])
        adata.write('pseudotime.h5ad', compression='gzip')

        df = adata.obs.groupby(label)[keys].mean().T
        df.style.background_gradient(cmap='coolwarm', axis=1)
        print (df)

    adata.write(os.path.join(adata.uns['temp'], 'temp.h5ad'), compression='gzip')
    
    if config.EXAMINE_GENE:
        from .individual_gene import exam_genes
        exam_genes(adata, config.EXAMINE_GENE)

    return replicates if config.GENERAL != 'Linear' else adata