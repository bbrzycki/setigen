.. |setigen| replace:: :mod:`setigen`
.. _setigen.voltage: https://setigen.readthedocs.io/en/master/setigen.voltage.html

Voltage synthesis (setigen.voltage)
===================================

The setigen.voltage_ module extends |setigen| to the voltage regime. Instead of directly synthesizing spectrogram data, we can produce real voltages, pass them through a software pipeline based on a polyphase filterbank, and record to file in GUPPI RAW format. As this process models actual hardware used by Breakthrough Listen for recording raw voltages, this enables lower level testing and experimentation.

The basic pipeline structure
----------------------------

.. image:: images/setigen_voltage_diagram_h.png

The basic layout of a voltage pipeline written using :code:`setigen.voltage` is shown in the image. 

First, we have an Antenna, which contains DataStreams for each polarization (1 or 2 total). Noise and signals are added to individual DataStreams, so that polarizations are unique and not necessarily correlated. These are added as functions, which accept an array of times in seconds and return an array of voltages, corresponding to random noise or defined signals. This allows us to obtain voltage samples on demand from each DataStream, and by extension from the Antenna. 

The main backend elements are the digitizer, filterbank, and requantizer. The digitizer quantizes input voltages to a desired number of bits, and a desired full width at half maximum (FWHM) in the quantized voltage space. The filterbank implements a software polyphase filterbank, coarsely channelizing input voltages. The requantizer takes the resulting complex voltages, and quantizes each component to either 8 or 4 bits, suitable for saving into GUPPI RAW format. 

All of these elements are wrapped into the RawVoltageBackend, which connects each piece together. The main method RawVoltageBackend.record() automatically retrieves real voltages as needed and passes them through each backend element, finally saving out the quantized complex voltages to disk.

A minimal working example of the pipeline is as follows:

.. code-block:: python

    from astropy import units as u
    import setigen as stg

    antenna = stg.voltage.Antenna(sample_rate=3e9*u.Hz, 
                                  fch1=6000e6*u.Hz,
                                  ascending=True,
                                  num_pols=1)
                                  
    antenna.x.add_noise(v_mean=0, 
                        v_std=1)
                        
    antenna.x.add_constant_signal(f_start=6002.2e6*u.Hz, 
                                  drift_rate=-2*u.Hz/u.s, 
                                  level=0.002)
                                  
    digitizer = stg.voltage.RealQuantizer(target_fwhm=32,
                                          num_bits=8)

    filterbank = stg.voltage.PolyphaseFilterbank(num_taps=8, 
                                                 num_branches=1024)

    requantizer = stg.voltage.ComplexQuantizer(target_fwhm=32,
                                               num_bits=8)

    rvb = stg.voltage.RawVoltageBackend(antenna,
                                        digitizer=digitizer,
                                        filterbank=filterbank,
                                        requantizer=requantizer,
                                        start_chan=0,
                                        num_chans=64,
                                        block_size=134217728,
                                        blocks_per_file=128,
                                        num_subblocks=32)
                                        
    rvb.record(raw_file_stem='example_1block',
               num_blocks=1, 
               length_mode='num_blocks',
               header_dict={'HELLO': 'test_value',
                            'TELESCOP': 'GBT'},
               verbose=True)