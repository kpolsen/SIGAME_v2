###     Module: main.py of SIGAME                   ###

# Import other SIGAME modules
import sigame.backend as backend
import sigame.frontend as frontend
import sigame.global_results as glo
import sigame.aux as aux
import sigame.plot as plot

#===============================================================================
"""  Load parameters """
#-------------------------------------------------------------------------------

params                      =   aux.load_parameters()
for key,val in params.items():
    exec(key + '=val')

#===============================================================================
"""  Initialize a global result object (GR) """
#-------------------------------------------------------------------------------

def run():
    '''Main controller that determines what tasks will be carried out
    '''

    GR                  	=   glo.global_results()

    print('\n Number of galaxies in selection: %s ' % GR.N_gal)


    print('** This is the main controller running SIGAME for the given galaxy sample **')

    if do_subgrid: backend.subgrid(GR)

    if do_interpolate: backend.interpolate(GR)

    if do_datacubes: backend.datacubes(GR)
