Fergal's Plotters
=================
This library provides some convenience functions for doing some plotting
operations I need to do regularly with matplotlib. For a demo on how to use
them, look at the `Plotters Functions Demo`__ notebook.

__ https://github.com/fbcotter/plotters/blob/master/Plotters%20Functions%20Demo.ipynb


.. _installation-label:

Installation
------------
Direct install from github (useful if you use pip freeze). To get the master
branch, try::

    $ pip install -e git+https://github.com/fbcotter/plotters#egg=plotters

or for a specific tag (e.g. 0.0.1), try::

    $ pip install -e git+https://github.com/fbcotter/plotters.git@0.0.1#egg=plotters

Download and pip install from Git::

    $ git clone https://github.com/fbcotter/plotters
    $ cd plotters
    $ pip install -r requirements.txt
    $ pip install -e .

I would recommend to download and install (with the editable flag), as it is
likely you'll want to tweak things/add functions more quickly than I can handle
pull requests.

Further documentation
---------------------

There is `more documentation <http://plotters.readthedocs.io>`_
available online and you can build your own copy via the Sphinx documentation
system::

    $ python setup.py build_sphinx

Compiled documentation may be found in ``build/docs/html/`` (index.html will be
the homepage)
