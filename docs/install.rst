Installation
============

GPU Acceleration
----------------

UniTVelo is designed based on TensorFlow's automatic differentiation architecture. 
Please make sure TensorFlow_ and relative CUDA_ dependencies are correctly installed.

Please use the following scripts to confirm TensorFlow is using the GPU::

    import tensorflow as tf
    print ("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

If GPU is not available, UniTVelo will automatically switch to CPU for model fitting or it can be spcified in `config.py` (see `Getting Started`_).

Main Module
-----------

UniTVelo requires Python 3.7 or later. 
We recommend to use Anaconda_ environment for version control and to avoid potential conflicts::

    conda create -n unitvelo python=3.7
    conda activate unitvelo

UniTVelo package can be conveniently installed via PyPI (for stable version) ::

    pip install unitvelo

or directly from GitHub repository (for development version)::

    pip install git+https://github.com/StatBiomed/UniTVelo

Dependencies
------------

Most required dependencies are automatically installed, e.g.

- `scvelo <https://scvelo.readthedocs.io/>`_ for a few pre- and post-processing analysis
- `statsmodels <https://www.statsmodels.org/stable/index.html>`_ for regression analysis
- `jupyter <https://jupyter.org/>`_ for running RNA velocity within notebooks

If you run into any issues or errors are raised during the installation process, feel free to contact us at GitHub_.

.. _Tensorflow: https://www.tensorflow.org/install
.. _CUDA: https://developer.nvidia.com/cuda-downloads
.. _Anaconda: https://www.anaconda.com/
.. _GitHub: https://github.com/StatBiomed/UniTVelo
.. _`Getting Started`: getting_started