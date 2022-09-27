#%%
#! Base Configuration Class
#! Don't use this class directly. 
#! Instead, sub-class it and override the configurations you need to change.


class Preprocessing(object):
    def __init__(self):
        self.MIN_SHARED_COUNTS = 20

        # (int) # of highly variable genes selected for pre-processing, default 2000
        # consider decreasing to 1500 when # of cells > 10k
        self.N_TOP_GENES = 2000

        self.N_PCS = 30
        self.N_NEIGHBORS = 30

        # (bool) use raw un/spliced counts or first order moments
        self.USE_RAW = False

        # (bool) rescaled Mu/Ms as input based on variance, default True 
        self.RESCALE_DATA = True

class Regularization(object):
    def __init__(self):
        # (bool) regularization on loss function to push peak time away from 0.5
        # mainly used in unified time mode for linear phase portraits
        self.REG_LOSS = True
        # (float) gloablly adjust the magnitude of the penalty, recommend < 0.1
        self.REG_TIMES = 0.075
        # (float) scaling parameter of the regularizer
        self.REG_SCALE = 1

        # (list of tuples) [(gene1, trend1), (gene2, trend2), (gene3, trend3), ...], 
        # a list of genes with trend can be one of {increase, decrease}, default None
        self.GENE_PRIOR = None
        self.GENE_PRIOR_SCALE = 5000

class Optimizer(object):
    def __init__(self):
        # (float) learning rate of the main optimizer
        self.LEARNING_RATE = 1e-2
        # (int) maximum iteration rate of main optimizer
        self.MAX_ITER = 12000

class FittingOption(object):
    def __init__(self):
        # Fitting options under Gaussian model 
        # '1' = Unified-time mode 
        # '2' = Independent mode
        self.FIT_OPTION = '1'

        # (str, experimental) methods to aggregate time metrix, default 'SVD'
        # Max SVD Raw
        self.DENSITY = 'SVD'
        # (str) whether to reorder cell based on relative positions for time assignment
        # Soft_Reorder (default) Hard (for Independent mode)
        self.REORDER_CELL = 'Soft_Reorder'
        # (bool) aggregate gene-specific time to cell time during fitting
        # controlled by self.FIT_OPTION
        self.AGGREGATE_T = True

        # (bool, experimental), whether clip negative predictions to 0, default False
        self.ASSIGN_POS_U = False

        # (bool, experimental) cell time restricted to (0, 1) if False, default False
        self.RESCALE_TIME = False

class VelocityGenes(object):
    def __init__(self):
        # (bool) linear regression $R^2$ on extreme quantile (default) or full data (adjusted)
        # valid when self.VGENES = 'basic'
        self.R2_ADJUST = True

        # (str) selection creteria for velocity genes used in RNA velocity construction, default basic
        # 1. raws, all highly variable genes specified by self.N_TOP_GENES will be used
        # 2. offset, linear regression $R^2$ and coefficient with offset, will override self.R2_ADJUST
        # 3. basic, linear regression $R^2$ and coefficient without offset
        # 4. single gene name, fit this designated gene alone, for model validation purpose only
        # 5. [list of gene names], manually provide a list of genes as velocity genes in string, might improve performance, see scNT
        self.VGENES = 'basic'

        # (float) threshold of R2 at later stage of the optimization proces
        # to capture the dynamics of more genes beside initially selected velocity genes
        # Note: self.AGENES_R2 = 1 will switch to origianl mode with no amplification
        self.AGENES_R2 = 1
        self.AGENES_THRES = 0.61

        # (bool, experimental) exclude cell that have 0 expression in either un/spliced when contributing to loss function
        self.FILTER_CELLS = False

class CellInitialization(object):
    def __init__(self):
        # (str) criteria for cell latent time initialization, default None
        # 1. None, initialized based on the exact order of input expression matrix
        # 2. gcount, str, initialized based on gene counts (https://www.science.org/doi/abs/10.1126/science.aax0249)
        # 3. cluster name, str, use diffusion map based time as initialization
        # 4. [(gene1, trend1), (gene2, trend2), (gene3, trend3), ...], list of tuples, 
        #    a list of genes with trend can be one of {increase, decrease} 
        self.IROOT = None

        # (int) number of random initializations of time points, default 1
        # in rare cases, velocity field generated might be reversed, possibly because stably and monotonically changed genes
        # change this parameter to 2 might do the trick
        self.NUM_REP = 1
        # when self.NUM_REP = 2, the following parameter will determine how the second time will be initialized 
        # re_pre, reverse the inferred cell time of first run
        # re_init, reverse the initialization time of first run
        self.NUM_REP_TIME = 're_pre'

class Configuration():
    def __init__(self):
        Preprocessing.__init__(self)
        Regularization.__init__(self)
        Optimizer.__init__(self)
        FittingOption.__init__(self)
        VelocityGenes.__init__(self)
        CellInitialization.__init__(self)

        # (int) speficy the GPU card for acceleration, default 0
        # -1 will switch to CPU mode
        self.GPU = 0

        # Gaussian Mixture
        self.BASE_FUNCTION = 'Gaussian'

        # Deterministic Curve Linear
        self.GENERAL = 'Curve'

        # (str) embedding format of adata, e.g. pca, tsne, umap, 
        # if None (default), algorithm will choose one automatically
        self.BASIS = None

        # (int, experimental) window size for sliding smoothing of distribution with highest probability
        # useful when self.DENSITY == 'Max'
        # self.WIN_SIZE = 50