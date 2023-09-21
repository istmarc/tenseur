Installation
============

Install from source
-------------------
Tenseur can be installed from its source code available at https://github.com/istmarco/Tenseur.

The following instructions describe how to clone the repository and install Tenseur on Linux and MacOS:

.. code-block:: bash

#: #dcdcdc
   $ git clone https://github.com/istmarco/Tenseur.git
   $ cd Tenseur
   $ mkdir build
   $ cd build
   $ cmake ..
   $ make install

Set the install prefix:

.. code-block:: bash

   $ cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
   $ make install

