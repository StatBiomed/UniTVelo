from .velocity import Velocity
import scvelo as scv
import os

def run_model(
    adata,
    label,
    config_file=None,
    normalize=True,
):
    """Preparation and pre-processing function of RNA velocity calculation.
    
    Args:
        adata (str): 
            takes relative of absolute path of Anndata object as input or directly adata object as well.
        label (str): 
            column name in adata.var indicating cell clusters.
        config_file (object): 
            model configuration object, default: None.
        
    Returns:
        adata: 
            modified Anndata object.
    
    """

    from .utils import init_config_summary, init_adata_and_logs
    config, _ = init_config_summary(config=config_file)
    adata, data_path = init_adata_and_logs(adata, config, normalize=normalize)

    scv.settings.presenter_view = True
    scv.settings.verbosity = 0
    scv.settings.file_format_figs = 'png'

    replicates, pre = None, 1e15
    adata_second = adata.copy()

    for rep in range(config.NUM_REP):        
        if rep >= 1:
            adata = adata_second.copy()
            adata.obs['latent_time_gm'] = pre_time_gm

        adata.uns['datapath'] = data_path
        adata.uns['label'] = label
        adata.uns['base_function'] = 'Gaussian'

        if config.BASIS is None:
            basis_keys = ["pca", "tsne", "umap"]
            basis = [key for key in basis_keys if f"X_{key}" in adata.obsm.keys()][-1]
        elif f"X_{config.BASIS}" in adata.obsm.keys():
            basis = config.BASIS
        else:
            raise ValueError('Invalid embedding parameter config.BASIS')
        adata.uns['basis'] = basis

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
    replicates.write(os.path.join(adata.uns['temp'], f'temp_{config.FIT_OPTION}.h5ad'))
    
    if 'examine_genes' in adata.uns.keys():
        from .individual_gene import exam_genes
        exam_genes(adata, adata.uns['examine_genes'])

    return replicates if config.GENERAL != 'Linear' else adata