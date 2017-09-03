"""
submodules
"""

__all__ = ["main","load_module","subgrid_module","GMC_module","dif_module","plot","aux"]

# from . import *
import numpy as np
import pandas as pd
import os.path
import sys
import pdb

# print(''+'\n'+\
#  '   ======================================================='+'\n'+\
#  '   '+'\n'+\
#  '   .oOOOo.  ooOoOOo  .oOOOo.     Oo    Oo      oO o.OOoOoo'+'\n'+\
#  '   o     o     O    .O     o    o  O   O O    o o  O      '+'\n'+\
#  '   O.          o    o          O    o  o  o  O  O  o      '+'\n'+\
#  '    `OOoo.     O    O         oOooOoOo O   Oo   O  ooOO   '+'\n'+\
#  '         `O    o    O   .oOOo o      O O        o  O      '+'\n'+\
#  '          o    O    o.      O O      o o        O  o      '+'\n'+\
#  '   O.    .O    O     O.    oO o      O o        O  O      '+'\n'+\
#  "    `oooO'  ooOOoOo   `OooO'  O.     O O        o ooOooOoO"+'\n'+\
#  '   '+'\n'+\
#  '   ======================================================='+'\n'+\
#  '    SImulator of GAlaxy Millimeter/submillimeter Emission'+'\n'+\
#  '-- A code to simulate the far-IR emission lines of the ISM  --'+'\n'+\
#  '----------- in galaxies from hydrodynamical codes ------------'+'\n'+\
#  '--- for the interpretation and prediction of observations. ---'+'\n'+\
#  '-- Contact: Karen Pardos Olsen, kpolsen (at) asu.edu (2016) --'+'\n'+\
#  ''+'\n')
 # style from http://www.kammerl.de/ascii/AsciiSignature.php
 # (alternatives: epic, roman, blocks, varsity, pepples, soft, standard, starwars)

################ Some instructions ######################

# Remember to select and edit a parameter file:
params_file = 'parameters_z6.txt'

print('--------------------------------------------------------------\n')

try:
	f = open(params_file)
	print('Reading parameter file: ['+params_file+'] ... ')
except IOError:
	sys.exit('Could not read parameter file: ['+params_file+'] ... ')

# Import parameters
from param_module import *
read_params(params_file)

# Import main SIGAME modules
from main import *

print('\nReady to continue!\n')










