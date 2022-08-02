#%%
import os
from time import gmtime, strftime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    from setuptools_scm import get_version
    __version__ = get_version(root="..", relative_to=__file__)
    del get_version
    
except (LookupError, ImportError):
    try:
        from importlib_metadata import version 
    except:
        from importlib.metadata import version 
    __version__ = version(__name__)
    del version

print (f'(Running UniTVelo {__version__})')
print (strftime("%Y-%m-%d %H:%M:%S", gmtime()))

from .main import run_model
from .config import Configuration
from .eval_utils import evaluate
from .gene_influence import influence
from .utils import choose_mode, subset_adata