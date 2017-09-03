## SÍGAME: SImulator of GAlaxy Millimeter/submillimeter Emission

### Description
This is a bite of code from SÍGAME; a method to simulate the emission lines of the ISM in galaxies from hydrodynamical codes for the interpretation and prediction of observations.
By running this code, you can reproduce the line emission from one z=6 galaxy also used in this 2017 paper: https://arxiv.org/abs/1708.04936. See instructions below


### Obtaining the SÍGAME code
Clone this repository to an empty folder on your computer by giving the following command in the terminal:
``` 
git clone https://github.com/kpolsen/SIGAME.git
```
You need python version 2 and a few packages.

### Requirements
This distribution has been tested on python v2.7.13 with the following package version:
- matplotlib v2.0.2

### Importing SÍGAME in python/ipython
All modules of SÍGAME are found in the sigame/ directory and are loaded into python with:
``` 
import sigame as si
```
Whenever you make changes to your copy of SGAME, remember to load those changes with:
``` 
reload(si)
```
A text file must be supplied with the general parameters for SÍGAME (redshift, name of galaxies etc.). For this distribution, the parameter file 'parameters_z6.txt' has been provided with the setup for one example galaxy. At the bottom of this file, you can change what SÍGAME is to do, by switching a -1 to a +1.

### Runnning SÍGAME
1: The first step of SÍGAME is to calculate the local FUV field, pressure and velocity dispersion. This is done within the submodule sigame.subgrid_module(). To start this procedure change the following line in 'parameters_z6.txt':
``` 
-1 subgrid
-1 create_GMCs
-1 line_calc_GMC
-1 create_dif_gas
-1 line_calc_dif
``` 
to:
``` 
+1 subgrid
-1 create_GMCs
-1 line_calc_GMC
-1 create_dif_gas
-1 line_calc_dif
``` 
Then start python and type:
``` 
import sigame as si
si.run()
```
This might take a little while and will create a save file in sigame/temp/sim_FUV/.

2: The second step subgrid the fluid elements of the simulation into diffuse and dense gas parts. Again, go to 'parameters_z6.txt' and change the bottom lines to:
``` 
-1 subgrid
+1 create_GMCs
-1 line_calc_GMC
+1 create_dif_gas
-1 line_calc_dif
``` 
(where the first line is important, so that SÍGAME does not subgrid again.) Now, you should see save files in sigame/temp/GMC/ and sigame/temp/dif.

3: The third step is to calculate the line emission. Again, go to 'parameters_z6.txt' and change the bottom lines to:
``` 
-1 subgrid
-1 create_GMCs
+1 line_calc_GMC
-1 create_dif_gas
+1 line_calc_dif
``` 
This should create files in sigame/temp/GMC/emission/ and sigame/temp/dif/emission/

4. Now you're ready to analyze the results and do some plotting! First try to see the outcome of the run with this command:
```
si.print_results()
```

5. To plot the results, the following functions are included, that can recreate plots in the [paper](https://arxiv.org/abs/1708.04936):
- Fig 1: SFR-Mstar relation: si.plot.SFR_Mstar()
- Fig 3: Histogram of GMC radii: si.plot.histos(), choose: gmc, Rgmc, default options otherwise…
- Figs 4,5: Histograms of cloudy grid parameters: si.plot.grid_parameters(), keywords ISM_phase = ‘GMC’ and ‘dif’
- Fig 6: CII-SFR relation at z~6: si.plot.CII_SFR_z6()
- Fig 7 top: Mass fractions of different ISM phases: si.analysis.ISM_mass_contribution()
- Fig 7 middle: Line luminosity fractions of different ISM phases: si.analysis.ISM_line_contribution()
- Fig 7 bottom: Line efficiency of different ISM phases: si.analysis.ISM_line_efficiency()
- Fig 8: OI-SFR and OIII-SFR relations at z~6: si.plot.OI_OIII_SFR()

### A note on reloading in python
You can make changes to a module, and then reload SIGAME to have python register those changes. 
Instead of loading submodules and reloading, I recommend changing the ipython configuration to reload by default everytime you type a python command. This is done by adding the two following lines to~/.ipython/profile_default/ipython_config.py (create this file if it doesn't exit):
```
c.InteractiveShellApp.extensions = ['autoreload']
c.InteractiveShellApp.exec_lines = ['%autoreload 2']
```
For more detail, see [this discussion](https://support.enthought.com/hc/en-us/articles/204469240-Jupyter-IPython-After-editing-a-module-changes-are-not-effective-without-kernel-restart?page=1#comment_203342093).


### References
  - Olsen, K. P., Greve, T. R., Narayanan, D., Thompson, R., Davé, R., Niebla-Rios, L., Stawinski, S.: SIGAME simulations of the [CII], [OI] and [OIII] line emission from star forming galaxies at z ~ 6, arXiv: [1708.04936](https://arxiv.org/abs/1708.04936), [ADS link](http://adsabs.harvard.edu/abs/2017arXiv170804936O).
  - Olsen, K. P., Greve, T. R., Narayanan, D., Thompson, R., Toft, S. Brinch, C.: Simulator of Galaxy Millimeter/Submillimeter Emission (SIGAME): The [CII]-SFR Relationship of Massive z=2 Main Sequence Galaxies, 2015, arXiv: [1507.00362](http://arxiv.org/abs/1507.00362), [ADS link](http://adsabs.harvard.edu/abs/2015ApJ...814...76O).
  - Olsen, K. P., Greve, T. R., Brinch, C., Sommer-Larsen, J., Rasmussen, J., Toft, S., Zirm, A.: SImulator of GAlaxy Millimeter/submillimeter Emission (SIGAME): CO emission from massive z=2 main sequence galaxies, 2016, arXiv: [1507.00012](http://arxiv.org/abs/1507.00012), [ADS link](http://adsabs.harvard.edu/abs/2016MNRAS.457.3306O).

