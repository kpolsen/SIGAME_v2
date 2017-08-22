## SÍGAME: SImulator of GAlaxy Millimeter/submillimeter Emission

### Description
This is a bite of code from SÍGAME; a method to simulate the emission lines of the ISM in galaxies from hydrodynamical codes for the interpretation and prediction of observations.

### Obtaining the SÍGAME code
Clone this repository to an empty folder on your computer by giving the following command in the terminal:
``` 
git clone https://github.com/kpolsen/SIGAME.git
```
You need python version 2 and a few packages.

### Running SÍGAME

All modules of SÍGAME are found in the sigame/ directory and loaded into python with:
``` 
import sigame as si
```
A text file must be supplied with the general parameters for SÍGAME (redshift, name of galaxies etc.). See 'parameters_z6.txt' for an example and modify as needed. Whenever you change the parameter file, remember to reload sigame in order to get those changes into the code:
``` 
reload(si)
```
If you decide to switch to another parameter file, you will have to change the filename in sigame/__init__.py (initiated with each reload of sigame). For example, to switch to a parameter file at z=2:
``` 
# First remember to edit and select a parameter file:
params_file = 'parameters_z2.txt'
``` 
You run SÍGAME with the command:
``` 
si.run()
```
which will call the program run() in the sigame/main.py module. What will be executed depends on what you select at the bottom of the parameter file. For example setting:
```
-1 load_galaxy
+1 subgrid
-1 create_GMCs
-1 line_calc_GMC
-1 create_dif_gas
-1 line_calc_dif
```
will only subgrid the selected model galaxies.

### Making changes to SIGAME modules
You make changes to a module, and then reload SIGAME to have python register those changes. 
Instead of loading submodules and reloading, I recommend changing the ipython configuration to reload by default everytime you type a python command. This is done by adding the two following lines to~/.ipython/profile_default/ipython_config.py (create this file if it doesn't exit):
```
c.InteractiveShellApp.extensions = ['autoreload']
c.InteractiveShellApp.exec_lines = ['%autoreload 2']
```
For more detail, see [this discussion](https://support.enthought.com/hc/en-us/articles/204469240-Jupyter-IPython-After-editing-a-module-changes-are-not-effective-without-kernel-restart?page=1#comment_203342093).


### References
  - Olsen, K. P., Greve, T. R., Narayanan, D., Thompson, R., Toft, S. Brinch, C.: Simulator of Galaxy Millimeter/Submillimeter Emission (SIGAME): The [CII]-SFR Relationship of Massive z=2 Main Sequence Galaxies, arXiv: [1507.00362](http://arxiv.org/abs/1507.00362)
  - Olsen, K. P., Greve, T. R., Brinch, C., Sommer-Larsen, J., Rasmussen, J., Toft, S., Zirm, A.: SImulator of GAlaxy Millimeter/submillimeter Emission (SIGAME): CO emission from massive z=2 main sequence galaxies, arXiv: [1507.00012](http://arxiv.org/abs/1507.00012)

