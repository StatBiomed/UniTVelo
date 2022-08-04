|PyPI| |Docs| 

.. |PyPI| image:: https://img.shields.io/pypi/v/unitvelo.svg
    :target: https://pypi.org/project/unitvelo
.. |Docs| image:: https://readthedocs.org/projects/unitvelo/badge/?version=latest
   :target: https://unitvelo.readthedocs.io

UniTVelo for temporally unified RNA velocity analysis
=====================================================

Temporally unified RNA velocity for single cell trajectory inference (UniTVelo) is implementated on Python 3 and TensorFlow 2. 
The model estimates velocity of each gene and updates cell time based on phase portraits concurrently.

.. image:: https://github.com/StatBiomed/UniTVelo/blob/main/figures/HumanBoneMarrow.png?raw=true
   :width: 300px
   :align: center

The major features of UniTVelo are,

* Using spliced RNA oriented design to model RNA velocity and transcription rates
* Introducing a unified latent time (`Unified-time mode`) across whole transcriptome to incorporate stably and monotonically changed genes
* Retaining gene-spcific time matrics (`Independent mode`) for complex datasets

UniTVelo has proved its robustness in 10 different datasets. Details can be found via our manuscript in bioRxiv which is currently under review (UniTVelo_).

.. _UniTVelo: https://www.biorxiv.org/content/10.1101/2022.04.27.489808v1

.. toctree::
   :caption: Home
   :maxdepth: 1
   :hidden:
   
   about
   install
   release
   references

.. toctree::
   :caption: Tutorials
   :maxdepth: 1
   :hidden:

   getting_started
   Figure2_ErythroidMouse
   Figure3_BoneMarrow
   Figure4_IntestinalOrganoid