Quickstart
==========


Import SÍGAME
-------------
SÍGAME has been tested in iPython v3.6.5 so far. We recommend running SÍGAME in a jupyter notebook, but the code can also be started from command line Python. In both cases, the SÍGAME module can be imported like this:

.. code-block:: python

  import sigame as si

Other than the standard modules that come with most python distributions, 
the user might need to install the module `PyTables <https://www.pytables.org>`_ with e.g.:

.. code-block:: python

  pip install tables

If all goes well, the following output should be printed to the terminal:

::

     ==============================================================
        
        .oOOOo.  ooOoOOo  .oOOOo.     Oo    Oo      oO o.OOoOoo
        o     o     O    .O     o    o  O   O O    o o  O      
        O.          o    o          O    o  o  o  O  O  o      
         `OOoo.     O    O         oOooOoOo O   Oo   O  ooOO   
              `O    o    O   .oOOo o      O O        o  O      
               o    O    o.      O O      o o        O  o      
        O.    .O    O     O.    oO o      O o        O  O      
         `oooO'  ooOOoOo   `OooO'  O.     O O        o ooOooOoO
        
     ==============================================================
         SImulator of GAlaxy Millimeter/submillimeter Emission
  -----  A code to simulate the far-IR emission lines of the ISM  -----
  --------------  in galaxies from hydrodynamical codes ---------------
  ------  for the interpretation and prediction of observations. ------
  -- Contact: Karen Pardos Olsen, kpolsen (at) protonmail.com (2019) --

The user will then be asked to choose a redshift for the sample of model galaxies to be analyzed. 
The default is z=0, corresponding to the test galaxy that comes with the newest release. 
Choosing z=0 will prompt the code to look for a file called "parameters_z0.txt" which 
contains the setup under which SÍGAME will run.

At the bottom of the parameter file, the following lines let the user select which tasks 
will be executed on the galaxy sample:

| BACKEND TASKS
| -1 extract_galaxy
| +1 subgrid
| -1 interpolate
| -1 datacubes
| FRONTEND TASKS
| -1 regions
| -1 PCR_analysis
|

For example, the above selection will run the "subgrid task" on the galaxies 
(step number 2 described in the `Overview <https://kpolsen.github.io/SIGAME/code_doc/index.html>`_) and 
nothing else on the galaxy sample. To execute the tasks selected, 
save the parameter file, restart python and type:

.. code-block:: python

  import sigame as si
  si.run()

.. note:: 

  Some of the backend tasks (subgrid and datacubes) make use of the **multiprocessing module** in python, 
  assuming by default that 4 cores are available. The code might run very slow if the computer in use 
  has fewer cores to work with, or if several cores are being idle. To change the number of cores available, 
  and hence used by SÍGAME, 
  go to the parameter file (parameters_z0.txt) and edit the following part:

  | Number of cores available [N_cores]
  | 4


Running the code
----------------

Start (or restart) python, and call the run() function located in backend.py like this:

.. code-block:: python

  import sigame as si
  si.run()


Tutorials
---------
The current release (v2.0.0) comes with 5 tutorials that briefly shows how to execute the basic SIGAME commands and plot the results. 
The tutorials are in jupyter notebook format and are described below.

Tutorial 0 - Plot histrograms.ipynb
***********************************
This tutorial examines the supplied z=0 galaxy and can generate histograms of different quantities for either stars, gas or dark matter. By default, the script will plot a histogram of gas mass in the simulation output: 

Tutorial 1 - Subgrid step.ipynb
*******************************
This tutorial will **subgrid the simulation gas particles**, which includes: 1) adding FUV field and external cloud pressure to the particle data, 2) creating GMCs and diffuse gas clouds. The FUV luminosity has been calculated with Starburst99 and supplied here as a look-up table to determine the FUV flux at each gas particle position. 

Tutorial 2 - Interpolation step.ipynb
*************************************
Here, SIGAME will try to **interpolate in a library of pre-calculated cloud models** to sum up line emission from GMC, DNG and DIG ISM phases. The library was calculated with Cloudy v17.01 for z=0 and supplied with the release.

Tutorial 3 - Datacube creation step.ipynb
*****************************************
Here, the results from the interpolation step are **drizzled onto datacube grids** in space and velocity.

Tutorial 4 - Vizualize results.ipynb
************************************
Finally, the resulting line luminosities and datacube can be **turned into figures viewed here**.

.. note:: 

  The release version comes with one test galaxy and all the tools to create line emission at z=0 for the following lines; [CII]158 and [NII]205. For other redshifts or lines, please contact us. 

