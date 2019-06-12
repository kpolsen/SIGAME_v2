###     Module: backend.py of SIGAME                   ###

# Import other SIGAME modules
import sigame.galaxy as gal
import sigame.aux as aux

# Import other modules
import pdb as pdb
import os

#===============================================================================
"""  Load parameters """
#-------------------------------------------------------------------------------

params                      =   aux.load_parameters()
for key,val in params.items():
    exec(key + '=val')        

def subgrid(GR):
    '''Adds subgrid features to galaxy object.
    '''
    print('\n--- Subgridding ---')      
    for gal_index in range(0,GR.N_gal):
        # Initiate subgrid object that inherits all the attributes of gal_ob
        # GR.print_galaxy_properties(gal_index=gal_index)
        sub_obj                 =   gal.subgrid_galaxy(gal_index=gal_index)
        sub_obj.setup_tasks()
        if sub_obj.do_FUV:              sub_obj.add_FUV()
        if sub_obj.do_P_ext:            sub_obj.add_P_ext()
        if sub_obj.do_GMCs:             sub_obj.add_GMCs()
        if sub_obj.do_dif:              sub_obj.add_dif()
    print('\n--------------------------------------------------------------\n')

def interpolate(GR):
    '''Interpolates in cloud models for gas clouds in this galaxy.
    '''
    print('\n--- Interpolating in cloud models ---')      
    for gal_index in range(0,GR.N_gal):
        clo_obj                 =   gal.interpolate_clouds(gal_index=gal_index)
        clo_obj.setup_tasks()
        if clo_obj.do_interpolate_GMCs: clo_obj.interpolate_GMCs()
        if clo_obj.do_interpolate_dif:  clo_obj.interpolate_dif()
    print('\n--------------------------------------------------------------\n')

def datacubes(GR):
    '''Creates datacubes.
    '''
    print('\n--- Datacube creation ---')  
    for gal_index in range(0,GR.N_gal):
        print("Check which datacubes we're doing...")
        gal_ob                 =   gal.galaxy(gal_index=gal_index)
        if ow:
            do_GMC_dc  =   True
            do_DNG_dc  =   True
            do_DIG_dc  =   True
            print('Overwrite is ON')
        # If not overwriting, check if subgridding has been done
        if not ow:
            do_GMC_dc  =   False
            do_DNG_dc  =   False
            do_DIG_dc  =   False
            # Try to load datacubes
            print(aux.get_file_location(gal_ob=gal_ob,gal_ob_present=True,target=target,ISM_dc_phase='GMC'))
            if not os.path.isfile(aux.get_file_location(gal_ob=gal_ob,gal_ob_present=True,target=target,ISM_dc_phase='GMC')) : do_GMC_dc = True
            if not os.path.isfile(aux.get_file_location(gal_ob=gal_ob,gal_ob_present=True,target=target,ISM_dc_phase='DNG')) : do_DNG_dc = True
            if not os.path.isfile(aux.get_file_location(gal_ob=gal_ob,gal_ob_present=True,target=target,ISM_dc_phase='DIG')) : do_DIG_dc = True
            print('Overwrite is OFF, will do:')
            if do_GMC_dc: print('- Create datacube for GMCs')
            if do_DNG_dc: print('- Create datacube for DNG')
            if do_DIG_dc: print('- Create datacube for DIG')
            if do_GMC_dc + do_DNG_dc + do_DIG_dc == 0: print('Nothing!')
        # if do_GMC_dc:gal_ob.datacube.create_dc(ISM_dc_phase='GMC') 
        if do_DNG_dc:gal_ob.datacube.create_dc(ISM_dc_phase='DNG') 
        if do_DIG_dc:gal_ob.datacube.create_dc(ISM_dc_phase='DIG') 
    print('\n--------------------------------------------------------------\n')







