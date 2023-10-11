.. |setigen| replace:: :mod:`setigen`
.. _setigen.funcs: https://setigen.readthedocs.io/en/master/setigen.funcs.html
.. _`Getting started`: https://setigen.readthedocs.io/en/master/getting_started.html
.. _`observational data`: https://setigen.readthedocs.io/en/master/advanced.html#creating-custom-observational-noise-distributions

Frame methods
=============

Getting frame data
------------------

To just grab the underlying intensity data, you can do

.. code-block:: Python

    data = frame.get_data(use_db=False)

As it implies, if you switch the :code:`use_db` flag to true, it will express
the intensities in terms of decibels. This can help visualize data a little better,
depending on the application.

Plotting frames
---------------

Examples of the built-in plotting utilities are on the `Getting started`_ page:

.. code-block:: Python

    frame.plot()

This method uses :code:`matplotlib.pyplot.imshow` behind
the scenes, which means you can still control plot parameters before and after
these function calls, e.g.

.. code-block:: Python

    fig = plt.figure(figsize=(10, 6))
    frame.plot()
    plt.title('My awesome title')
    plt.savefig('frame.png')
    plt.show()
    
Frame integration
-----------------

To time integrate to get a spectrum, or to frequency integrate to get time series 
intensities, you can use :code:`stg.Frame.integrate()`:

.. code-block:: Python
    
    spectrum = frame.integrate() # stg.integrate(frame)
    time_series = frame.integrate(axis='f') # or axis=1
    
This function is a wrapper for `stg.integrate()`, with the same parameters. The
`axis` parameter can be either `t` or `0` to integrate along the time axis, or `f` or 
`1` to integrate along the frequency axis. The `mode` parameter can be either 'mean' or
'sum' to determine the manner of integration.

Frame slicing
-------------

Given frequency boundary indices `l` and `r`, we can "slice" a frame by using 
:code:`stg.Frame.get_slice()`, a wrapper for :code:`stg.get_slice()`:

.. code-block:: Python

    s_fr = frame.get_slice(l, r) # stg.get_slice(frame, l, r)
    
Slicing is analogous to Numpy slicing, e.g. :code:`A[l:r]`, along the frequency axis.
This method returns a new frame with only the sliced data. This is useful when chained
together with boundary detection methods, or simply to isolate sections of a frame
for analysis.

Doppler dedrifting
------------------

If you have a frame containing a Doppler drifting signal, you can "dedrift" the frame
using :code:`stg.dedrift()`, specifying a target drift rate (Hz/s):

.. code-block:: Python

    dd_fr = stg.dedrift(frame, drift_rate=2)
    
This returns a new frame with only the dedrifted data; this will be smaller in
the frequency dimension depending on the drift rate and frame resolution. 

Alternatively, if 'drift_rate' is contained in the frame's metadata 
(:code:`frame.metadata`), the function will automatically dedrift the frame using that 
value. 

.. code-block:: Python

    drift_rate = 2
    frame.metadata['drift_rate'] = drift_rate
    dd_fr = stg.dedrift(frame)