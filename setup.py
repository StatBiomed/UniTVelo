#%%
import io
import os
from setuptools import setup, find_packages

# version of cudatoolkit and tensorflow
REQUIRED = [
    'numpy',
    'scikit-learn>=0.22',
    'pandas',
    'scipy>=1.4.1',
    'seaborn',
    'matplotlib>=3.3.0',
    'tqdm',
    'scanpy>=1.5',
    'statsmodels',
    'anndata>=0.7.5',
    'scvelo>=0.2.2',
    'IPython',
    'ipykernel',
    'IProgress',
    'ipywidgets',
    'jupyter',
    'tensorflow>=2.4.1'
]

here = os.path.abspath(os.path.dirname(__file__))
with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = '\n' + f.read()

setup(
    name='unitvelo',
    version='0.1.6',
    # use_scm_version=True,
    # setup_requires=['setuptools_scm'],

    description='Temporally unified RNA velocity inference',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Mingze Gao',
    author_email='gmz1229@connect.hku.hk',
    python_requires='>=3.7.0',
    url='https://github.com/StatBiomed/UniTVelo',
    packages=find_packages(),

    entry_points={
        'console_scripts': ['unitvelo = unitvelo.main:run_model'],
    },

    install_requires=REQUIRED,
    extras_require={'parallel': ['multiprocessing>=3.8']},
    include_package_data=True,
    license='BSD',
    keywords=[
        'RNA velocity',
        'Unified time',
        'Transcriptomics',
        'Kinetic',
        'Trajectory inference'
    ],

    classifiers=[
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: BSD License',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Visualization'
    ]
)