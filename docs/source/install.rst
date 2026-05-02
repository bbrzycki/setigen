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

|setigen| currently targets Python 3.10 through 3.14.

You can use ``pip`` to install the package automatically:

.. code-block:: bash

    pip install setigen
    
Alternately, you can clone the repository and install it directly. At the 
command line, execute:

.. code-block:: bash

    git clone git@github.com:bbrzycki/setigen.git
    python -m pip install .

For local development, install the package in editable mode with the project
extras:

.. code-block:: bash

    python -m pip install -e ".[dev]"

|setigen| includes a compatibility pin on ``setuptools`` because |blimpy|
currently imports ``pkg_resources`` at runtime. Normal installs pick this up
automatically.

One of the dependencies for |setigen| is |blimpy|, which is used for working 
with BL filterbank data products. Note that you can still generate synthetic 
data frames even without observational data!

Because of how the |bitshuffle| package was written, if you are working with 
HDF5 data products (e.g. ending with ".hdf5" or ".h5"), you may also need to 
do the following, especially if you'd like to save |setigen| frame data as 
HDF5 files:

.. code-block:: bash

    pip install -U git+https://github.com/h5py/h5py
    pip install git+https://github.com/kiyo-masui/bitshuffle
    
Note: this can lead to |h5py| compatibility issues with older versions of 
Tensorflow. Some work-arounds: if possible, work primarily with filterbank 
files, or use multiple Python environments to separate data handling and 
Tensorflow work. 

To use GPU with setigen.voltage
-------------------------------

``setigen.voltage``'s GPU acceleration is powered by CuPy 
(https://docs.cupy.dev/en/stable/install.html). Installation is not required 
to use vanilla |setigen| or the voltage module, but it is highly recommended 
to accelerate voltage computations. Enable it with
``stg.voltage.set_backend('cupy')`` before constructing voltage objects. Use
``stg.voltage.set_backend('numpy')`` to force CPU execution. The legacy
``SETIGEN_ENABLE_GPU=1`` environment variable is still supported for existing
scripts. The current public backend API is voltage-focused; the shared
NumPy/CuPy policy is described in :doc:`array_backends`. While it isn't used directly by |setigen|,
you may also find it helpful to install ``cusignal`` 
(https://github.com/rapidsai/cusignal) for access to CUDA-enabled versions of 
``scipy`` functions when writing custom voltage signal source functions.
