
A hybrid optimization suite for biological neural networks
===============================================================

.. image:: https://travis-ci.com/MarcusJP/Odin.svg?token=vE9NzRRorvxxsXbKCSRZ&branch=master
    :target: https://travis-ci.com/MarcusJP/Odin

Motivation
------------
Optimization of biological neural models (Integrate and Fire, Hodgkin-Huxley...) as well as circuits connected via synapses.

.. image:: img/final_goal.png
    :width: 800px
    :align: center
    :height: 500px
    :scale: 50
    :alt: alternate text

.. image:: img/inhexc.png
    :width: 800px
    :align: center
    :height: 500px
    :scale: 50
    :alt: alternate text

Description
------------

- Parallel Simulation
- Parallel Optimization
- User defined model
- Tensorflow
- LSTM
- Unit tests
- Documentation
- Tutorial


Getting started
---------------

You need python 3.5 or higher !

Run in the root directory :

1) install the required libraries

        make init

2) install the package

        python3 setup.py install

2) Launch tests

        make test

Folders
---------------

- docs : files for creating documentation with Sphinx
- img : images
- opthh : package python files
- tests : unit tests
- tutorial : notebook to run with Jupyter


TODO
---------------

- documentation
- tutorial
- add noise target
- target value plots
0) lstm model to generate the outputs of AVA and AVB.
1) full HH model.
2) full LSTM model with our synapses.
3) change only AVA with LSTM and all others with HH
4) change both AVA and AVB with LSTM and the rest with HH
5) change all sensory neurons with LSTM
6) change all interneurons with HH.
7) make all with LSTM cells except sensors.
8) make all with LSTM cells except the AVA and AVB
