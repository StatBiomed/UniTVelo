|PyPI| |Docs| 

.. |PyPI| image:: https://img.shields.io/pypi/v/unitvelo.svg
    :target: https://pypi.org/project/unitvelo
.. |Docs| image:: https://readthedocs.org/projects/unitvelo/badge/?version=latest
   :target: https://unitvelo.readthedocs.io

UniTVelo for temporally unified RNA velocity analysis
=====================================================

Temporally unified RNA velocity for single cell trajectory inference (UniTVelo) is implementated on Python 3 and TensorFlow 2. 
The model estimates velocity of each gene and updates cell time based on phase portraits concurrently.

.. image:: https://github.com/StatBiomed/UniTVelo/blob/e5d6f62df122b22a631c1081512faccc0fca640a/figures/HumanBoneMarrow.png?raw=true
   :width: 300px
   :align: center

The major features of UniTVelo are,

* Using spliced RNA oriented design to model RNA velocity and transcription rates
* Introducing a unified latent time (`Unified-time mode`) across whole transcriptome to incorporate stably and monotonically changed genes
* Retaining gene-spcific time matrics (`Independent mode`) for complex datasets

UniTVelo has proved its robustness in 10 different datasets. Details can be found via our manuscript in bioRxiv which is currently under review [UniTVelo].


References
==========





.. toctree::
   :caption: Home
   :maxdepth: 1
   :hidden:
   
   index
   install
   release
   references

.. toctree::
   :caption: Tutorials
   :maxdepth: 1
   :hidden:

   getting_started
   mouse_erythroid
   intestinal_organoid
   human_bonemarrow