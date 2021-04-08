.. setigen documentation master file, created by
   sphinx-quickstart on Wed Jul  4 14:40:16 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. |setigen| replace:: ``setigen``
.. _setigen: https://github.com/bbrzycki/setigen
.. |blimpy| replace:: ``blimpy``
.. _blimpy: https://github.com/UCBerkeleySETI/blimpy


Welcome to setigen's documentation!
===================================

.. image:: images/flashy_synthetic.png
    :align: center
    :scale: 75

|setigen|_ is a Python library for generating and injecting artificial
narrow-band signals into radio requency data. |setigen|_ interfaces
primarily with two types of data: spectrograms or dynamic spectra, saved in two-dimensional NumPy arrays or filterbank files (:code:`.fil` extension), and raw voltages (GUPPI RAW files). Both data formats are instrumental to Breakthrough Listen's data collection and analysis pipelines.

|setigen|_ allows the user to generate synthetic signals quickly in the
time-frequency domain in the form of data Frames. Furthermore, the user may inject these synthetic signals
into real observational data loaded from filterbank files. |setigen|_ plays well with the |blimpy|_ package.

The :code:`setigen.voltage` module enables the synthesis of GUPPI RAW files via synthetic real voltage "observations" and a software signal processing pipeline that implements a polyphase filterbank, mirroring actual BL hardware. The voltage module supports single and multi-antenna RAW files, and can be GPU accelerated via CuPy. 

Breakthrough Listen @ Berkeley: https://seti.berkeley.edu/listen/

Table of Contents
=================

.. toctree::
   :maxdepth: 1

   install
   getting_started
   basics
   advanced
   voltages
   setigen
   setigen.voltage

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
