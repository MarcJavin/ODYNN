
ODYNN : Optimization for DYnamic Neural Networks
===============================================================

.. image:: https://travis-ci.com/MarcusJP/ODYNN.svg?branch=master
    :target: https://travis-ci.com/MarcusJP/ODYNN
.. image:: https://codecov.io/gh/MarcusJP/ODYNN/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/MarcusJP/ODYNN
.. image:: https://readthedocs.org/projects/odynn/badge/?version=latest
    :target: https://odynn.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

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

Warnings
----------------

ODYNN is still in development and its syntax might change.

TODO
---------------

- more documentation
- more tutorials
