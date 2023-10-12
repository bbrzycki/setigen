.. |setigen| replace:: :mod:`setigen`


Getting started
===============

The heart of |setigen| is the Frame object. For signal injection and manipulation,
we call each snippet of time-frequency data a "frame." There are two main ways
to initialize frames, starting from either resolution/size parameters or existing
observational data.

Here's a minimal working example for a purely synthetic frame, injecting a constant
intensity signal into a background of chi-squared noise. Parameters in |setigen| are
specified either in terms of SI units (Hz, s) or :code:`astropy.units`, as in the example:

.. code-block:: python

    from astropy import units as u
    import setigen as stg
    import matplotlib.pyplot as plt

    frame = stg.Frame(fchans=1024*u.pixel,
                      tchans=32*u.pixel,
                      df=2.7939677238464355*u.Hz,
                      dt=18.253611008*u.s,
                      fch1=6095.214842353016*u.MHz)
    noise = frame.add_noise(x_mean=10, noise_type='chi2')
    signal = frame.add_signal(stg.constant_path(f_start=frame.get_frequency(index=200),
                                                drift_rate=2*u.Hz/u.s),
                              stg.constant_t_profile(level=frame.get_intensity(snr=30)),
                              stg.gaussian_f_profile(width=40*u.Hz),
                              stg.constant_bp_profile(level=1))

    fig = plt.figure(figsize=(10, 6))
    frame.plot(xtype="px", db=False)
    plt.savefig("frame.png", bbox_inches='tight')
    plt.show()

.. image:: images/gs_synth.png

This simple signal can also be generated using the method :func:`~setigen.frame.Frame.add_constant_signal`,
which is optimized for created signals of constant intensity and drift rate in large frames:

.. code-block:: python

    frame.add_constant_signal(f_start=frame.get_frequency(200),
                              drift_rate=2*u.Hz/u.s,
                              level=frame.get_intensity(snr=30),
                              width=40*u.Hz,
                              f_profile_type='sinc2')

Similarly, here's a minimal working example for injecting a signal into a frame of
observational data. Note that in this example, the observational data also has 
dimensions 32x1024 to make it easy to visualize here.

.. code-block:: python

    from astropy import units as u
    import setigen as stg
    import matplotlib.pyplot as plt

    data_path = 'path/to/data.fil'
    frame = stg.Frame(waterfall=data_path)
    frame.add_signal(stg.constant_path(f_start=frame.get_frequency(200),
                                       drift_rate=2*u.Hz/u.s),
                     stg.constant_t_profile(level=frame.get_intensity(snr=30)),
                     stg.gaussian_f_profile(width=40*u.Hz),
                     stg.constant_bp_profile(level=1))

    fig = plt.figure(figsize=(10, 6))
    frame.plot(db=False)
    plt.show()

.. image:: images/gs_obs.png

We can also view this data in decibels, the common method for plotting radio 
spectrograms and the default in |setigen|:

.. code-block:: python

    fig = plt.figure(figsize=(10, 6))
    frame.plot()
    plt.show()

.. image:: images/gs_obs_db.png

Usually, filterbank data is saved with frequencies in descending order, with the first
frequency bin centered at :code:`fch1`. |setigen| works with data in increasing frequency
order, and will reverse the data order when appropriate if the frame is initialized with such 
an observation. However, if you are working with data or would like to synthesize
data for which :code:`fch1` should be the minimum frequency, set :code:`ascending=True` when 
initializing the Frame object. Note that if you initialize Frame using a filterbank file with
frequencies in increasing order, you do not need to set :code:`ascending` manually.

.. code-block:: python

    frame = stg.Frame(fchans=1024*u.pixel,
                      tchans=32*u.pixel,
                      df=2.7939677238464355*u.Hz,
                      dt=18.253611008*u.s,
                      fch1=6095.214842353016*u.MHz,
                      ascending=True)

Assuming you have access to a data array, with corresponding resolution information, you can
can also initialize a frame as follows. Just make sure that your data is already arranged in the desired frequency order; setting the :code:`ascending` parameter will only affect the frequency 
values that are mapped to the provided data array.

.. code-block:: python

    my_data = # your 2D array
    frame = stg.Frame.from_data(df=2.7939677238464355*u.Hz,
                                dt=18.253611008*u.s,
                                fch1=6095.214842353016*u.MHz,
                                ascending=True,
                                data=my_data)
                                
    frame.plot()