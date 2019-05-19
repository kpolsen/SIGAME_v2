## SÍGAME: SImulator of GAlaxy Millimeter/submillimeter Emission

###
Jump straigt to the auto-generated code documentation:
https://kpolsen.github.io/SIGAME_dev/

### Description
This is a code to simulate the emission lines of the ISM in galaxies from hydrodynamical codes for the interpretation and prediction of observations.

### Obtaining the SÍGAME code
Clone this repository to an empty folder on your computer by giving the following command in the terminal:
``` 
git clone https://github.com/kpolsen/SIGAME.git
```

OBS: For Linux users! If you're using the terminal, here the commands we found useful so far:

For updating from the master branch:
``` 
git pull origin master
```
When you make changes (before or after), please switch to your own personal branch with a name of your choice:
```
git branch -b NAME-OF-YOUR-BRANCH
```
For making a pull request with all your recent changes:
``` 
git add .
git commit -m "commmit message"
git push --set-upstream origin NAME-OF-YOUR-BRANCH
git request-pull origin/master NAME-OF-YOUR-BRANCH
```
where NAME-OF-YOUR-BRANCH is the name of your branch.

### Running SÍGAME

All modules of SÍGAME are found in the sigame/ directory and loaded into python with:
``` 
import sigame as si
```
Importing sigame will ask you what redshift you're working at and who you are to set up a path for external large files not tracked by github. To change (or add) the path for your user, go into __init__.py and edit the relevant part.

Depending on the chosen redshift, a specific paramter file while be loaded and must be supplied with the general parameters for SÍGAME (redshift, resolution of datacubes etc.). See 'parameters_z0.txt' for an example and modify as needed. Whenever you change the parameter file, remember to reload sigame in order to get those changes into the code:
``` 
reload(si)
```
To create datacubes of line emission, you run SÍGAME with the command:
``` 
si.run()
```
which will call the program run() in the sigame/backend.py module. What will be executed depends on what you select at the bottom of the parameter file. For example setting:
```
+1 extract_galaxy
-1 subgrid
-1 interpolate
-1 datacubes
```
will only extract, center and cut out the selected model galaxies.

### Making changes to SÍGAME modules
If you make changes to a module, say to the function histos() in the sigame/plot.py module, then you can try to load in those changes with:
```
import sigame.plot as siplot
reload(siplot)
```
Now you're ready to run e.g. plot.histos() with the changes implemented.
```
siplot.histos()
```
This will not always work though, so instead of loading submodules and reloading or exiting and re-entering python, I recommend changing the ipython configuration to reload by default everytime you type a python command. This is done by adding the two following lines to~/.ipython/profile_default/ipython_config.py (create this file if it doesn't exit):
```
c.InteractiveShellApp.extensions = ['autoreload']
c.InteractiveShellApp.exec_lines = ['%autoreload 2']
```
For more detail, see [this discussion](https://support.enthought.com/hc/en-us/articles/204469240-Jupyter-IPython-After-editing-a-module-changes-are-not-effective-without-kernel-restart?page=1#comment_203342093).

### Coding rules
SÍGAME is a module under construction that may in the future be prepared for the stand-alone-use by anyone interested. But for the moment, it will only be viewed and used by people of this group, meaning that each has freedom to write functions/modules in the way that they prefer. However, we try to stick to these general rules:
- pull from the master before making edits
- docstrings in numpy format: http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html

... and let me know if you think of overall ways to improve the code! Happy python coding :)

### Collaborators (active in developing the code)
Karen Pardos Olsen, kpolsen (at) asu.edu
Daisy Leung, 
Thomas Greve, 
Lily Whitler, lwhitler (at) asu.edu

### References
  - 2018: Olsen, K. P., Greve, T. R., Narayanan, D., Thompson, R., Davé, Niebla Rios, L., Stawinsi, S.: "Erratum: SIGAME Simulations of the [CII], [OI], and [OIII] Line Emission from Star-forming Galaxies at z~6 (2018)", ApJ 857 2, [ADS link](http://adsabs.harvard.edu/abs/2018ApJ...857..148O)
  - 2017: Olsen, K. P., Greve, T. R., Narayanan, D., Thompson, R., Davé, Niebla Rios, L., Stawinsi, S.: "SIGAME Simulations of the [CII], [OI], and [OIII] Line Emission from Star-forming Galaxies at z~6 (2017)", ApJ 846 2, arXiv: [1708.04936](https://arxiv.org/abs/1708.04936)
  - 2017: Olsen, K. P., Greve, T. R., Brinch, C., Sommer-Larsen, J., Rasmussen, J., Toft, S., Zirm, A.: "SImulator of GAlaxy Millimeter/submillimeter Emission (SIGAME): CO emission from massive z=2 main sequence galaxies", arXiv: [1507.00012](http://arxiv.org/abs/1507.00012)
  - 2015: Olsen, K. P., Greve, T. R., Narayanan, D., Thompson, R., Toft, S. Brinch, C.: "Simulator of Galaxy Millimeter/Submillimeter Emission (SIGAME): The [CII]-SFR Relationship of Massive z=2 Main Sequence Galaxies", MNRAS 457 3, arXiv: [1507.00362](http://arxiv.org/abs/1507.00362)
