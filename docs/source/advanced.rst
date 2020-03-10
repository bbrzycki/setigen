.. |setigen| replace:: :mod:`setigen`
.. _setigen.funcs: https://setigen.readthedocs.io/en/master/setigen.funcs.html

Advanced topics
===============

Advanced signal creation
------------------------

Behind the scenes, :func:`~setigen.frame.Frame.add_signal` uses signal parameter
functions to compute intensity for each time, frequency pair in the data. This
is kept quite general to allow for the creation of complex signals. In this
section, we explore some of the flexibility behind :func:`~setigen.frame.Frame.add_signal`.

Writing custom signal functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can go beyond :mod:`setigen`'s pre-written signal functions by
writing your own. For each :func:`~setigen.frame.Frame.add_signal` input parameter
(:code:`path`, :code:`t_profile`, :code:`f_profile`, and :code:`bp_profile`),
you can pass in your own custom functions. Note that these inputs are themselves functions.

It's important that the functions you pass into each parameter have the correct
inputs and outputs. Specifically:

    :code:`path`
        Function that takes in time [array] ``t`` and outputs a frequency [array]

    :code:`t_profile`
        Function that takes in time [array] ``t`` and outputs an intensity [array]

    :code:`f_profile`
        Function that takes in frequency [array] ``f`` and a reference central
        frequency [array] ``f_center``, and outputs an intensity [array]

    :code:`bp_profile`
        Function that takes in frequency [array] ``f`` and outputs an intensity [array]

For example, here's the code behind the sine path shape:

.. code-block:: Python

    def my_sine_path(f_start, drift_rate, period, amplitude):
        def path(t):
            return f_start + amplitude * np.sin(2 * np.pi * t / period) + drift_rate * t
        return path

Alternately, you can use the lambda operator:

.. code-block:: Python

    def my_sine_path(f_start, drift_rate, period, amplitude):
        return lambda t: return f_start + amplitude * np.sin(2 * np.pi * t / period) + drift_rate * t

These can then be incorporated as:

.. code-block:: Python

    signal = frame.add_signal(my_sine_path(f_start=frame.get_frequency(200),
                                           drift_rate=2*u.Hz/u.s,
                                           period=100*u.s,
                                           amplitude=100*u.Hz),
                              stg.constant_t_profile(level=1),
                              stg.box_f_profile(width=20*u.Hz),
                              stg.constant_bp_profile(level=1))

To see more examples on how you might write your own parameter functions, check out the
source code behind the pre-written functions (setigen.funcs_).


Using arrays as signal parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sometimes it can be difficult to wrap up a desired signal property into a
separate function, or perhaps there is external existing code that calculates
desired properties. In these cases, we can also use arrays to describe these signals,
instead of functions.

.. code-block:: python

    from astropy import units as u
    import numpy as np
    import setigen as stg
    import matplotlib.pyplot as plt

    frame = stg.Frame(fchans=1024*u.pixel,
                      tchans=32*u.pixel,
                      df=2.7939677238464355*u.Hz,
                      dt=18.25361108*u.s,
                      fch1=6095.214842353016*u.MHz)
    frame.add_noise(x_mean=5, x_std=2, x_min=0)

    path_array = np.random.rand(frame.get_frequency(200),
                                frame.get_frequency(400),
                                32)
    t_profile_array = np.random.rand(frame.get_intensity(snr=20),
                                     frame.get_intensity(snr=40),
                                     32)

    frame.add_signal(path_array,
                     t_profile_array,
                     stg.gaussian_f_profile(width=40*u.Hz),
                     stg.constant_bp_profile(level=1))

    fig = plt.figure(figsize=(10, 6))
    frame.render()
    plt.savefig('frame.png', bbox_inches='tight')
    plt.show()

.. image:: images/advanced_array_synth.png



Optimization and accuracy
^^^^^^^^^^^^^^^^^^^^^^^^^








Creating custom observational noise distributions
--------------------------------------------------------






Create a dataset using existing observations
----------------------------------------------------
