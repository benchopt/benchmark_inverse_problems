
My Benchopt Benchmark
=====================
|Build Status| |Python 3.6+|

Benchopt is a package to simplify and make more transparent and
reproducible comparisons of optimization methods.
This benchmark is dedicated to solvers of inverse problems. It contains

Datasets:

- CBSD68/Set3c (train/test)
- BSD500/CBSD68 (train/test)
- BSD500

Solvers:

- DPIR
- Wavelet
- DRUNet

Tasks:

- Denoising
- Deblurring
- Super-Resolution
- Inpainting

Install
--------

Set up Benchopt and Clone repository
************************************

This benchmark can be retrieved using the following commands:

.. code-block::

   $ pip install -U benchopt
   $ git clone https://github.com/benchopt/benchmark_inverse_problems

Customize config file
*********************

Edit the ``config.yml`` file in the root of the benchmark

.. code-block::

    data_home: ____________          # Path to the main folder that will contain the data
    data_paths:
        Set3c_CBSD68: ____________   # Path where the DeepInv built dataset and physic will be saved
        CBSD68_BSD500: ____________  # Path where the DeepInv built dataset and physic will be saved
        BSD500: ____________         # Path to the folder `images` from `data_home` of the BSD500 dataset (see : Download data section)

For more information on how config file works see :

- https://benchopt.github.io/benchmark_workflow/config_benchopt.html
- https://benchopt.github.io/user_guide/tweak_datasets.html

Download data
*************

Set3c
^^^^^

This dataset is automatically downloaded from HuggingFace **deepinv/set3c**. You have nothing to do.

CBSD68
^^^^^^

This dataset is automatically downloaded from HuggingFace **deepinv/CBSD68**. You have nothing to do.

BSD500
^^^^^^

Please download the dataset by following this url : https://web.archive.org/web/20120508113820/http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz

Extract the archive somewhere in your ``data_home`` path and update the BSD500 key of the config file to point towards the ``images`` directory in ``BSR/BSDS500/data/images``

Run the benchmark
-----------------

You can now run the benchmark by using the following command

.. code-block::

    benchopt run path/to/the/benchmark

Apart from the problem, options can be passed to ``benchopt run``, to restrict the benchmarks to some solvers or datasets, e.g.:

.. code-block::

	$ benchopt run benchmark_inverse_problems -s solver1 -d dataset2 --max-runs 10 --n-repetitions 10


Use ``benchopt run -h`` for more details about these options, or visit https://benchopt.github.io/api.html.

.. |Build Status| image:: https://github.com/benchopt/benchmark_inverse_problems/actoiworkflows/main.yml/badge.svg
   :target: https://github.com/benchopt/benchmark_inverse_problems/actions
.. |Python 3.6+| image:: https://img.shields.io/badge/python-3.6%2B-blue
   :target: https://www.python.org/downloads/release/python-360/
