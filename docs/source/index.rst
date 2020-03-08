.. setigen documentation master file, created by
   sphinx-quickstart on Wed Jul  4 14:40:16 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. |setigen| replace:: :mod:`setigen`
.. |blimpy| replace:: ``blimpy``
.. _blimpy: https://github.com/UCBerkeleySETI/blimpy


Welcome to setigen's documentation!
===================================

.. image:: images/flashy_synthetic.png
    :align: center
    :scale: 75

|setigen| is a Python library for generating and injecting artificial
narrow-band signals into time-frequency data. |setigen| interfaces
primarily with data saved in two-dimensional NumPy arrays or filterbank files
(:code:`.fil` extension).

|setigen| allows the user to generate synthetic signals in the
time-frequency domain. Furthermore, the user may inject these synthetic signals
into real observational data using tools that rely on the |blimpy|_ package
(maintained by Breakthrough Listen based at UC Berkeley).

Table of Contents
=================

.. toctree::
   :maxdepth: 1

   install
   getting_started
   generate
   add_noise
   filterbank
   setigen

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
