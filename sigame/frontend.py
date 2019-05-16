###     Module: frontend.py of SIGAME                   ###

# Import other SIGAME modules
import sigame.global_results as glo
import sigame.galaxy as gal
import sigame.aux as aux

# Import other modules
import pdb as pdb
import numpy as np
import pandas as pd
import os

#===============================================================================
"""  Load parameters """
#-------------------------------------------------------------------------------

params                      =   aux.load_parameters()
for key,val in params.items():
    exec(key + '=val')

def regions(GR,**kwargs):
    '''Creates regions and extracts info from them.

    Parameters
    ----------
    size_arcsec : float
        Size (diameter) of region in arcsec, default: 16.8 arcsec (Croxall+17 Herschel)

    max_regions_per_gal : int
        Max number of regions accepted per galaxy, default: 1000

    line1 : str
        Line ID in line1/line2 ratio, default: 'CII'

    line2 : str
        Line ID in line1/line2 ratio, default: 'NII_205'

    '''
    print('\n--- Creating regions! ---')      

    # handle default values and kwargs
    args                    =   dict(size_arcsec=16.6, max_regions_per_gal=1000,search_within=0.25,ISM_phase='GMC',line1='CII',line2='NII_205',extract_from='regions',units='Wm2')
    args                    =   aux.update_dictionary(args,kwargs)
    for key in args: exec(key + '=args[key]')

    

    print('Check if a regions file exist...')
    create_regions          =   False
    if os.path.isfile(d_t+'model_line_ratios_'+line1+'_'+line2) : 
        if ow:      
            print('Regions file does exist, but overwrite [ow] is on, will create one! ')
            create_regions = True
        if not ow:  
            print('Regions file does exist and overwrite [ow] is off, will do nothing. ')
    if not os.path.isfile(d_t+'model_line_ratios_'+line1+'_'+line2) : 
        print('Regions file does NOT exist, will create one! ')
        create_regions = True

    if create_regions:
        model_line1             =   np.array([])
        model_line2             =   np.array([])
        model_SFRsds_exact      =   np.array([])
        model_SFRsds_CII        =   np.array([])
        f_CII_neus              =   np.array([])
        f_neus                  =   np.array([])
        model_F_CII_ergs        =   np.array([])
        model_Zs_exact          =   np.array([])
        model_Zs_OH             =   np.array([])
        gal_indices             =   np.array([])
        for gal_index in range(0,GR.N_gal):
                print('\nNow doing galaxy number %s' % (gal_index+1))
                # Initiate galaxy object
                gal_ob             =   gal.galaxy(gal_index=gal_index)
                # Add region(s) as attribute
                gal_ob.add_attr('regions',**args)
                # Evaluate properties and emission from these regions:
                results             =   gal_ob.regions.evaluate_regions(gal_ob,evaluation_list=['Z','SFRsd','line_lum','f_CII_neu','f_neu'],lines=[line1,line2])
                SFRsds_exact        =   results['SFRsd']['exact']
                SFRsds_CII          =   results['SFRsd']['CII']
                F_CII_ergs          =   results['SFRsd']['F_CII_ergs'] # ergs/s/kpc^2
                F_line1             =   aux.Jykm_s_to_W_m2(line1,gal_ob.zred,results[line1])/gal_ob.regions.size_sr # W/m/sr
                F_line2             =   aux.Jykm_s_to_W_m2(line2,gal_ob.zred,results[line2])/gal_ob.regions.size_sr # W/m/sr
                f_CII_neu           =   results['f_CII_neu']
                f_neu               =   results['f_neu']
                Zs_exact            =   results['Z']['exact']
                Zs_OH               =   results['Z']['OH']
                # Convert to line flux in W/m^2
                region_indices      =   np.arange(len(F_line1))
                # remove regions with no SFRsd or very low line lum:
                mask                =   (SFRsds_exact > 0) & (F_line1 > Herschel_limits[line1])
                F_line1             =   F_line1[mask]
                F_line2             =   F_line2[mask]
                SFRsds_exact        =   SFRsds_exact[mask]
                SFRsds_CII          =   SFRsds_CII[mask]
                F_CII_ergs          =   F_CII_ergs[mask]
                f_CII_neu           =   f_CII_neu[mask]
                f_neu               =   f_neu[mask]
                region_indices      =   region_indices[mask]
                Zs_exact            =   Zs_exact[mask]
                Zs_OH               =   Zs_OH[mask]
                # Collect results for this galaxy
                model_line1         =   np.append(model_line1,F_line1)
                model_line2         =   np.append(model_line2,F_line2)
                model_SFRsds_exact  =   np.append(model_SFRsds_exact,SFRsds_exact)
                model_SFRsds_CII    =   np.append(model_SFRsds_CII,SFRsds_CII)
                f_CII_neus          =   np.append(f_CII_neus,f_CII_neu)
                f_neus              =   np.append(f_neus,f_neu)
                model_F_CII_ergs    =   np.append(model_F_CII_ergs,F_CII_ergs)
                model_Zs_exact      =   np.append(model_Zs_exact,Zs_exact)
                model_Zs_OH         =   np.append(model_Zs_OH,Zs_OH)
                gal_indices         =   np.append(gal_indices,np.zeros(len(F_line1))+gal_index)
                print('Made %s region(s) in %s ' % (len(F_line1),GR.galnames[gal_index]))
                print('SFR of this galaxy: %s' % gal_ob.SFR)
                print('SFR-weighted Z of this galaxy: %s' % gal_ob.Zsfr)
                print('Median of mass-weighted mean metallicities extracted for this galaxy: %s' % (np.median(Zs_exact.values)))
                # Visualize region(s)
                gal_ob.regions.plot_positions(gal_ob,region_indices,line='CII',convolve=True,units='Wm2_sr')
        # Save result:
        results             =   pd.DataFrame(dict(SFRsd_exact=model_SFRsds_exact,SFRsd_CII=model_SFRsds_CII,\
                                f_CII_neu=f_CII_neus,f_neu=f_neus,line1=model_line1,line2=model_line2,F_CII_ergs=model_F_CII_ergs,\
                                Z_exact=model_Zs_exact,Z_OH=model_Zs_OH,gal_indices=gal_indices))
        results['line_ratio']   =   results['line1']/results['line2']
        results.to_pickle(d_t+'model_line_ratios_'+line1+'_'+line2)
        print('Done! In total: %s region(s) with detectable flux\n' % (len(results)))






