#%%
#! Base Configuration Class
#! Don't use this class directly. 
#! Instead, sub-class it and override the configurations you need to change.

class Configuration(object):
    def __init__(self):
        # (int) speficy the GPU card for acceleration, default 0
        # -1 will switch to CPU mode
        self.GPU = 0

        # (str) relevant path for saving scv plots
        # self.FIG_DIR = './figures/'

        # Gaussian Mixture
        self.BASE_FUNCTION = 'Gaussian'

        # Deterministic Curve Linear
        self.GENERAL = 'Curve'

        # (str) embedding format of adata, e.g. t_sne, u_map, 
        # if None (default), algorithm will choose one automatically
        self.BASIS = None

        # (int) # of highly variable genes selected for pre-processing, default 2000
        # consider decreasing to 1500 when # of cells > 10k
        self.N_TOP_GENES = 2000

        # (bool) linear regression $R^2$ on extreme quantile (default) or full data (adjusted)
        self.R2_ADJUST = True

        # (bool) linear regression $R^2$ with offset, default False
        # if True, would override self.R2_ADJUST
        self.OFFSET_GENES = False

        # (bool, experimental) exclude cell that have 0 expression in either un/spliced when contributing to loss function
        self.FILTER_CELLS = False

        # (bool) 
        self.EXAMINE_GENE = False

        # (bool, experimental) cell time restricted to (0, 1) if False, default False
        self.RESCALE_TIME = False

        # (bool) rescaled Mu/Ms as input based on variance, default True 
        self.RESCALE_DATA = True

        # (str) name root cell cluster 
        # if specified, use diffusion map based time as initialization, default None
        # would override self.NUM_REPEAT and have improved performance
        self.IROOT = None

        # (int, experimental) number of random initializations of time points, default 1
        # in rare cases, velocity field generated might be reversed, possibly because stably and monotonically changed genes
        # change this parameter to 2 might do the trick
        self.NUM_REPEAT = 1

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

        # (bool) regularization on loss function to push peak time away from 0.5
        # mainly used in unified time mode for linear phase portraits
        self.REG_LOSS = True
        # (float) gloablly adjust the magnitude of the penalty, recommend < 0.1
        self.REG_TIMES = 0.075
        # (float) scaling parameter of the regularizer
        self.REG_SCALE = 1

        # (int, experimental) window size for sliding smoothing of distribution with highest probability
        # useful when self.DENSITY == 'Max'
        # self.WIN_SIZE = 50

        # (float) learning rate of the main optimizer
        self.LEARNING_RATE = 1e-2

        # (int) maximum iteration rate of main optimizer
        self.MAX_ITER = 12000

        # (bool) use raw un/spliced counts or first order moments
        self.USE_RAW = False

        # (bool) selected genes / All 2000 raw genes
        # if True, would override self.R2_ADJUST and self.OFFSET_GENES
        self.RAW_GENES = False