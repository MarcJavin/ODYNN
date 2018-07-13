
A hybrid optimization suite for biological neural networks
===============================================================

Motivation
------------
Optimization of biological models of neurons (Integrate and Fire, Hodgkin-Huxley...) as well as circuits connected via synapses.

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

Commands
---------------
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

- Run one step LSTM
- implement LSTM
- inhibitory synapses : default and constraints
- heatmaps for circuit
- documentation
- tutorial

