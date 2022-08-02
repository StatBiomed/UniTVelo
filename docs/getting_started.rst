Getting Started
===============

Public Datasets
---------------

Examples of UniTVelo and steps for reproducible results are provided in Jupyter Notebook under notebooks_ folder. 
For start, please refer to records analyzing `Mouse Erythroid`_ and `Human BoneMarrow`_ datasets.

RNA Velocity on New Dataset
---------------------------

UniTVelo provides an integrated function for velocity analysis by default whilst specific configurations might need to be adjusted accordingly.

# 1. Import package::

    import unitvelo as utv

# 2. Sub-class and override base configuration file (here lists a few frequently used), please refer `config.py` for detailed arguments::

    velo = utv.config.Configuration()
    velo.R2_ADJUST = True 
    velo.IROOT = None
    velo.FIT_OPTION = '1'
    velo.GPU = 0

Arguments:

- `velo.R2_ADJUST` (bool), linear regression R-squared on extreme quantile (default) or full data (adjusted).
- `velo.IROOT` (str), specify root cell cluster would enable diffusion map based time initialization, default None.
- `velo.FIT_OPTION` (str), '1' Unified-time mode (default), '2' Independent mode
- `velo.GPU` (int), specify the GPU card used for fitting, -1 will switch to CPU mode, default 0

# 3. Run model (label refers to column name in adata.obs specifying celltypes)::

    adata = utv.run_model(path_to_adata, label, config_file=velo)
    scv.pl.velocity_embedding_stream(adata, color=label, dpi=100, title='')

# 4. Evaluation metrics (Optional)::

    # Cross Boundary Direction Correctness
    # Ground truth should be given via `cluster_edges`
    metrics = {}
    metrics = utv.evaluate(adata, cluster_edges, label, 'velocity')

    # Latent time estimation
    scv.pl.scatter(adata, color='latent_time', color_map='gnuplot', size=20)

    # Phase portraits for individual genes (experimental)
    utv.pl.plot_range(gene_name, adata, velo, show_ax=True, time_metric='latent_time')

.. _notebooks: https://github.com/StatBiomed/UniTVelo/tree/main/notebooks
.. _`Mouse Erythroid`: ./Figure2_ErythroidMouse
.. _`Human BoneMarrow`: ./Figure3_BoneMarrow