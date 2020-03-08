.. |setigen| replace:: :mod:`setigen`
.. |blimpy| replace:: ``blimpy``
.. _blimpy: https://github.com/UCBerkeleySETI/blimpy
.. |h5py| replace:: ``h5py``
.. _h5py: https://github.com/h5py/h5py
.. |bitshuffle| replace:: ``bitshuffle``
.. _bitshuffle: https://github.com/kiyo-masui/bitshuffle


============
Installation
============

You can use `pip` to install the package automatically:

.. code-block:: bash

    pip install setigen
    
Alternately, you can clone the repository and install it directly. At the command line, execute:

.. code-block:: bash

    git clone git@github.com:bbrzycki/setigen.git
    python setup.py install

One of the dependencies for |setigen| is |blimpy|, which is used for working with BL filterbank data products. Note that you can still generate synthetic data frames even without observational data!

Because of how the |bitshuffle| package was written, if you are working with HDF5 data products (e.g. ending with ".hdf5" or ".h5"), you may also need to do the following, especially if you'd like to save |setigen| frame data as HDF5 files:

.. code-block:: bash

    pip install -U git+https://github.com/h5py/h5py
    pip install git+https://github.com/kiyo-masui/bitshuffle
    
Note: this can lead to |h5py| compatibility issues with older versions of Tensorflow. Some work-arounds: if possible, work primarily with filterbank files, or use multiple Python environments to separate data handling and Tensorflow work. 