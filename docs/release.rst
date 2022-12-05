Release History
===============

Version 0.2.5
-------------
- Fix issues # 19_ and # 20_ in GitHub repo
- Fix bugs in plotting function, plot_range
- Structurize model outputs

Version 0.2.4
-------------
- Support the input of one or multiple genes trends to initialize cell time, see config.py, parameter self.IROOT
- Re-formulate the structure of configuration file 
- Fixed bugs on progress bar

Version 0.2.3
-------------
- Change threshold for self.AGENES_R2

Version 0.2.2
-------------
- Add benchmarking function to scVelo
- Provide prediction script which uses down-sampled data to predict RNA velocity and cell time on entire dataset
- Add notebooks for auxiliary functions

Version 0.2.1
-------------
- Support input of both raw path and adata objects
- Fix bugs on logging file

Version 0.2.0
-------------
- Beta version of UniTVelo released
- Provide option of using gene counts for model initialization
- Provide reference script for choosing unified-time mode or independent mode
- Provide sampling script when dataset is oversized and GPU memory is bottleneck
- Re-organize configuration file
- Number of velocity genes can be amplified (an adjustable hyper-parameter) during optimization which allows post-analysis on more genes 

Version 0.1.6
-------------
- Fix bug for early stopping scenarios

Version 0.1.5
-------------
- Fix bug in saving files
- Add adjustable parameters for penalty in configuration file

Version 0.1.4
-------------
- Add penalty on loss function
- Support informative gene selection in unified-time mode

Version 0.1.3
-------------
- Support CPU mode for model fitting
- Fix bugs in documentations

Version 0.1.2
-------------
- Fix bugs in setup.py
- Add tutorials and documentations

Version 0.1.0
-------------
- Alpha version of UniTVelo released

.. _19: https://github.com/StatBiomed/UniTVelo/issues/19
.. _20: https://github.com/StatBiomed/UniTVelo/issues/20
