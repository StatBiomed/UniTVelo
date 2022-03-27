#%%
#! Base Configuration Class
#! Don't use this class directly. 
#! Instead, sub-class it and override the configurations you need to change.

class Configuration(object):
    def __init__(self):
        self.GPU = 2
        self.FIG_DIR = './figures/'
        # Gaussian RBF Mixture
        self.BASE_FUNCTION = 'Gaussian'
        # Deterministic Curve Linear
        self.GENERAL = 'Curve'
        self.BASIS = None
        self.N_TOP_GENES = 2000

        self.OFFSET_GENES = False
        # Exclude cell that have 0 expression in either un/spliced 
        # When contributing to loss function
        self.FILTER_CELLS = False

        # Fit inidividual gene
        self.EXAMINE_GENE = False

        # Cell time restricted to (0, 1) if False, default False
        self.RESCALE_TIME = False
        # Ms/Mu or rescaled Ms/Mu based on variance
        self.RESCALE_DATA = True
        # Seems to be dynamical original genes? Not exactly the same
        self.R2_ADJUST = True

        # Set root cell clusters, if using DPT as time initialization
        self.IROOT = None
        # Number of random initializations of time points
        # Invalid if self.IROOT != None
        self.NUM_REPEAT = 1
        # Fitting options under Gaussian model 
        # '1' = gene-shared time points 
        # '2' = gene-specific time points 
        self.FIT_OPTION = '1'

        # Max SVD Raw
        self.DENSITY = 'SVD'
        # Soft_Reorder Soft Hard
        self.REORDER_CELL = 'Soft_Reorder'
        # Aggregate gene-specific time to cell time during fitting
        self.AGGREGATE_T = True
        # Only take unspliced counts > 0 when assigning timepoints
        self.ASSIGN_POS_U = True

        # Useful when DENSITY == 'Max'
        self.WIN_SIZE = 50

        # learning rate of the main optimizer
        self.LEARNING_RATE = 1e-2
        self.MAX_ITER = 10000

        # Use raw un/spliced counts or first order moments
        self.USE_RAW = False
        # Selected genes / All 2000 raw genes
        self.RAW_GENES = False