
__all__ = ["main","backend","frontend","global_results","galaxy","plot","aux","param"]

# from . import *
import numpy as np
import pandas as pd
import os as os
import sys
import pdb
import re
import subprocess as sub


print(''+'\n'+\
 '   ======================================================='+'\n'+\
 '   '+'\n'+\
 '   .oOOOo.  ooOoOOo  .oOOOo.     Oo    Oo      oO o.OOoOoo'+'\n'+\
 '   o     o     O    .O     o    o  O   O O    o o  O      '+'\n'+\
 '   O.          o    o          O    o  o  o  O  O  o      '+'\n'+\
 '    `OOoo.     O    O         oOooOoOo O   Oo   O  ooOO   '+'\n'+\
 '         `O    o    O   .oOOo o      O O        o  O      '+'\n'+\
 '          o    O    o.      O O      o o        O  o      '+'\n'+\
 '   O.    .O    O     O.    oO o      O o        O  O      '+'\n'+\
 "    `oooO'  ooOOoOo   `OooO'  O.     O O        o ooOooOoO"+'\n'+\
 '   '+'\n'+\
 '   ======================================================='+'\n'+\
 '    SImulator of GAlaxy Millimeter/submillimeter Emission'+'\n'+\
 '-- A code to simulate the far-IR emission lines of the ISM  --'+'\n'+\
 '----------- in galaxies from hydrodynamical codes ------------'+'\n'+\
 '--- for the interpretation and prediction of observations. ---'+'\n'+\
 '-- Contact: Karen Pardos Olsen, kpolsen (at) asu.edu (2018) --'+'\n'+\
 ''+'\n')
 # style from http://www.kammerl.de/ascii/AsciiSignature.php
 # (alternatives: epic, roman, blocks, varsity, pepples, soft, standard, starwars)

################ Some instructions ######################

# Remember to select and edit a parameter file:
z           =   input('At which redshift? (default: 0)')
if z == '': z = '0'
if re.match("^\d+$",z) != None: z = 'z'+z # if z is a number, then add 'z' as a description
params_file =   'parameters_'+z+'.txt'

try:
    f = open(params_file)
    print('Reading parameter file: ['+params_file+'] ... ')
except IOError:
    sys.exit('Could not read parameter file: ['+params_file+'] ... ')

# Start new collection of parameters
params = {}

# Set up user specific root locations where larger temporary data files are stored
params['parent']    =   ''
params['d_data']    =   'sigame/temp/%s_data_files/' % z
print('\nwill look for code in %s%s' % (os.getcwd(),'/sigame/'))
print('will look for cloudy data in %s/%s\n' % (os.getcwd(),params['d_data']))

# Create parameter file
from sigame.param import *
read_params(params_file,params)


# Edit aux.py to find the temporary parameter file
file_in     =    open(params['parent']+'sigame/aux.py','r')
file_out    =    open(params['parent']+'sigame/aux_temp.py','w')
for line in file_in:
    if 'insert external parent here' in line:
        line = "    params                      =   np.load('"+params['parent']+"temp_params.npy').item() # insert external parent here\n"
    file_out.write(line)
file_in.close()
file_out.close()
delete          =   sub.Popen('mv '+params['parent']+'sigame/aux_temp.py '+params['parent']+'sigame/aux.py',shell=True)
stdout,stderr   =   delete.communicate()   # wait until cloudy is done

# Set up directory tree integrating any local and external parent directories
params['d_temp']            =   params['parent'] + 'sigame/temp/'

print('\n--------------------------------------------------------------')

# Import main SIGAME modules
from sigame.main import *

# Check that modules are of the right version
# aux.check_version(np,[1,12,0])

print('\nReady to continue!\n')
