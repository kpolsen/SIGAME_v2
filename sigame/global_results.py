"""
Module: global_results
"""

# Import other SIGAME modules
import sigame.aux as aux

# Import other modules
import numpy as np
import pandas as pd
import pdb as pdb
import os
import glob

#===============================================================================
"""  Load parameters """
#-------------------------------------------------------------------------------

params                      =   aux.load_parameters()
g = globals()
for key,val in params.items():
    exec(key + '=val',g)

# Needs update: why does Zsfr turn out nan sometimes?
class global_results:
    '''An object referring to the global results of a selection of galaxies, containing global properties of as attributes.

    Example
    -------
    >>> import global_results as glo
    >>> GR = glo.global_results()
    >>> GR.print_results()
    '''

    def __init__(self,**kwargs):

        # get global results as dictionary
        GR                      =   self.__get_file()
        GR['lum_dist']          =   GR['lum_dist']*0.+10. # putting all z=0 galaxies at 10 kpc

        for attr in ['M_dense','M_GMC','M_dif','Zmw']:
            if attr not in GR.keys(): GR = self.__set_attr(GR,attr)
        for line in lines:
            if attr not in GR.keys(): GR = self.__set_attr(GR,'L_'+line)

        # save global_results dictionary
        filename    =   self.__get_file_location()
        GR.to_pickle(filename)

        # add all dictionary entries as parameters to global_results instance
        for key in GR: setattr(self,key,GR[key].values)
        setattr(self,'N_gal',len(GR['galnames']))

    def __get_file(self,**kwargs):

        # get filename
        filename    =   self.__get_file_location()

        # create file if it doesn't exist
        if not os.path.isfile(filename):
            print("\ncould not find file at %s \n... creating Global Results file!" % filename)
            self.__create_file(**kwargs)

        return pd.read_pickle(filename)

    def __get_file_location(self):

        file_location   =   d_temp + 'global_results/%s_%sgals%s%s' % (z1,nGal,ext_DENSE,ext_DIFFUSE)

        return file_location

    def __create_file(self,**kwargs):

        # create empty container for global_results
        GR  =   {}

        extracted_gals  =   pd.read_pickle(d_temp+'galaxies/%s_extracted_gals' % z1)

        zreds,galnames  =   extracted_gals['zreds_unsorted'], extracted_gals['galnames_unsorted']
        GR['galnames']  =   galnames
        GR['zreds']     =   zreds
        GR['N_gal']     =   len(galnames)

        # convert DF to DataFrame
        GR          =   pd.DataFrame(GR)

        # add un-populated targets to DF
        targets =   ['R_gal', 'M_gas', 'M_star', 'M_GMC', 'M_dense', 'M_dm', 'SFR', 'SFRsd', 'Zsfr']
        N_gal   =   GR.shape[0]
        for target in targets: GR[target] = np.zeros(N_gal)

        # collect sim data
        for i,zred,galname in zip(range(N_gal),zreds,galnames):

            # get data
            gal_ob = dict( galname=galname, zred=zred, gal_index=i) # dummy gal object
            simgas  =   aux.load_temp_file(gal_ob=gal_ob, sim_type='gas', gal_ob_present=True)
            simstar =   aux.load_temp_file(gal_ob=gal_ob, sim_type='star', gal_ob_present=True)
            simdm   =   aux.load_temp_file(gal_ob=gal_ob, sim_type='dm', gal_ob_present=True)
            # check if f_H21 is there, if not, make a copy so the rest of the code works (temporary fix):
            if 'f_H21' not in simgas.keys():
                simgas['f_H21'] = simgas['f_H2'].values
                gal_ob = dict( galname=galname, zred=zred, gal_index=i+1) # dummy gal object
                aux.save_temp_file(simgas,gal_ob=gal_ob,sim_type='gas',gal_ob_present=True)

            # get radii
            r_gas   =   np.sqrt(sum([simgas[x].values**2. for x in ['x','y','z']]))
            r_star  =   np.sqrt(sum([simstar[x].values**2. for x in ['x','y','z']]))

            # set R_gal
            rmax    =   np.max(r_gas)
            GR.at[i, 'R_gal'] = rmax

            # set M_*
            # mask    =   r_star < rmax
            # age     =   simstar['age'].values[mask]
            # m_stars =   simstar['m'].values[mask]
            age     =   simstar['age'].values
            m_stars = simstar['m'].values
            GR.at[i, 'M_star'] = np.sum(m_stars)
            GR.at[i, 'M_dm'] = np.sum(simdm['m'].values)
            GR.at[i, 'M_gas'] = np.sum(simgas['m'].values)
            GR.at[i, 'f_dense'] = (np.sum(simgas['m'].values*simgas['f_H21'].values))/np.sum(simgas['m'].values)

            # ratios and metalicities
            if np.sum(m_stars) <= 0.5:
                print(i)
                import pdb; pdb.set_trace()

            GR.at[i, 'mw_age'] =  (np.sum(age*m_stars)/np.sum(m_stars)) # mass-weighted age
            GR.at[i,'SFR'] = np.sum(m_stars[age < 100.])/100e6
            GR.at[i,'SFRsd'] = GR.SFR[i]/(np.pi*rmax**2.)
            GR.at[i,'Zsfr'] = np.sum(simgas['Z'].values*(simgas['SFR'].values + 1.e-8))/np.sum(simgas['SFR'].values + 1.e-8)
        for attr in ['lum_dist','M_dense','M_GMC','M_dif','Zmw']:
            GR = self.__set_attr(GR,attr)
        for line in lines:
            GR = self.__set_attr(GR,'L_'+line)

        # # set M_tot
        # DF['M_tot'] =   DF['M_gas'].values + DF['M_star'].values + DF['M_dm'].values

        # Only consider galaxies with a SFR > 0
        GR  =   GR[GR['SFR'] > 0].reset_index(drop=True)

        # If z=0, only consider low-z galaxies (out to ~ 100 Mpc)
        if z1 == 'z0': GR  =   GR[GR['zreds'] < 0.04].reset_index(drop=True)

        # sort DF by stellar mass
        GR  =   GR.sort_values('M_star').reset_index(drop=True)
        # Make sure we only extract a sample of size nGal
        # GR  =   GR[0:nGal]

        # save DF of global results
        filename    =   self.__get_file_location()
        GR.to_pickle(filename)

        return GR

    def __set_attr(self,GR_int,attr):
        # Get missing attributes

        for gal_index in range(0,GR_int['N_gal'][0]):
            gal_ob = dict(galname=GR_int['galnames'][gal_index], zred=GR_int['zreds'][gal_index], gal_index=gal_index) # dummy gal object

            if attr == 'lum_dist':
                LD                      =   aux.get_lum_dist(GR_int['zreds'])
                GR_int['lum_dist']          =   LD

            if attr == 'M_GMC':

                if gal_index == 0:
                    M_GMC                   =   np.zeros(GR_int['N_gal'][0]) # get length of sample, only once (ugly method)
                GMCgas                  =   aux.load_temp_file(gal_ob=gal_ob, verbose=False, ISM_phase='GMC')
                if type(GMCgas) != int: M_GMC[gal_index] = np.sum(GMCgas['m'])
                GR_int['M_GMC']             =    M_GMC

            if attr == 'M_dense':
                if gal_index == 0:
                    M_dense                 =   np.zeros(GR_int['N_gal'][0]) # get length of sample, only once (ugly method)

                simgas                  =   pd.read_pickle(aux.get_file_location(gal_ob=gal_ob, sim_type='gas', gal_ob_present=True))
                # simgas                  =   aux.load_temp_file(gal_ob=gal_ob, verbose=False, sim_type='gas')
                if type(simgas) != int: M_dense[gal_index] = np.sum(simgas['m'].values*simgas['f_H21'].values)
                GR_int['M_dense']           =    M_dense

            if attr == 'M_dif':
                if gal_index == 0:
                    M_dif                   =   np.zeros(GR_int['N_gal'][0])
                difgas                  =   aux.load_temp_file(gal_ob=gal_ob, verbose=False, ISM_phase='dif')
                if type(difgas) != int: M_dif[gal_index] = np.sum(difgas['m'])
                GR_int['M_dif']             =    M_dif

            if attr == 'Zmw':
                if gal_index == 0:
                    Zmw                     =   np.zeros(GR_int['N_gal'][0])
                difgas                  =   aux.load_temp_file(gal_ob=gal_ob, verbose=False, ISM_phase='dif')
                GMCgas                  =   aux.load_temp_file(gal_ob=gal_ob, verbose=False, ISM_phase='GMC')
                if (type(difgas) != int) & (type(GMCgas) != int):
                    Zmw[gal_index] = np.sum((difgas['Z']*difgas['m']+difgas['Z']*difgas['m'])/(np.sum(difgas['m'])+np.sum(GMCgas['m'])))
                GR_int['Zmw']               =    Zmw

            if 'L_' in attr:
                if gal_index == 0:
                    L                       =   np.zeros(GR_int['N_gal'][0])
                    L_DNG                   =   np.zeros(GR_int['N_gal'][0])
                    L_DIG                   =   np.zeros(GR_int['N_gal'][0])

                # from GMCs
                GMCgas                  =   aux.load_temp_file(gal_ob=gal_ob, verbose=False, ISM_phase='GMC')
                if type(GMCgas) != int:
                    try:
                        L[gal_index]   = np.sum(GMCgas[attr].values)
                    except:
                        L[gal_index] = 0
                GR_int[attr+'_GMC']                =    L

                # from dif
                difgas                  =   aux.load_temp_file(gal_ob=gal_ob, verbose=False, ISM_phase='dif')


                if type(difgas) != int:
                    try:
                        L_DNG[gal_index]    =   np.sum(difgas[attr + '_DNG'].values)
                        L_DIG[gal_index]    =   np.sum(difgas[attr + '_DIG'].values)
                    except:
                        L_DNG[gal_index]    =  0
                        L_DIG[gal_index]    =  0
                GR_int[attr+'_DNG']                =    L_DNG
                GR_int[attr+'_DIG']                =    L_DIG

                GR_int[attr]                       =    L + L_DNG + L_DIG

        return(GR_int)

    def print_results(self):

        # Rerun global results file creating
        GR = self.__create_file()

        print('\n BASIC SIMULATION INFO FOR THIS GALAXY SAMPLE')
        print('+%80s+' % ((5+20+15+15+10+10+10+10+10+8)*'-'))
        print('|%5s|%20s|%15s|%15s|%10s|%10s|%10s|%10s|%10s' % ('Name'.center(5), 'sim name'.center(20), 'Stellar mass'.center(15), 'Gas mass'.center(15), 'f_dense'.center(10), 'SFR'.center(10), 'Sigma_SFR'.center(10), 'z'.center(10), 'Z_SFR'.center(10)))
        print('|%5s|%20s|%15s|%15s|%10s|%10s|%10s|%10s|%10s' % (''.center(5), ''.center(20), '[10^9 Msun]'.center(15), '[10^9 Msun]'.center(15), '[%]'.center(10), '[Msun/yr]'.center(10), '[MW units]'.center(10), ''.center(10), '[solar]'.center(10)))
        print('+%80s+' % ((5+20+15+15+10+10+10+10+10+8)*'-'))
        for gal_index in range(0,len(GR.galnames)):
            # print(GR.R_gal[gal_index])
            # if GR.zreds[gal_index] == 0:
            # print("size of galaxy {0:.2f} [kpc]".format(GR.R_gal[gal_index]))
            # print("dense gas mass {:.2f} [Msun]".format(GR.M_dense[gal_index]))
            print('|%5s|%20s|%15s|%15s|%10s|%10s|%10s|%10s|%10s' % ('G'+str(gal_index+1),GR.galnames[gal_index].center(20),\
                '{:.2e}'.format(GR.M_star[gal_index]/1e9),\
                '{:.2e}'.format(GR.M_gas[gal_index]/1e9),\
                '{:.2f}'.format(GR.f_dense[gal_index]*100.),\
                '{:.4f}'.format(GR.SFR[gal_index]),\
                '{:.4f}'.format(GR.SFRsd[gal_index]/SFRsd_MW),\
                '{:.4f}'.format(GR.zreds[gal_index]),\
                '{:.4f}'.format(GR.Zsfr[gal_index])))
        print('+%80s+' % ((5+20+15+15+10+10+10+10+10+8)*'-'))

        if hasattr(self,'L_CII'):
            print('\n ISM PROPERTIES AND LINE LUMINOSITIES')
            print('+%80s+' % ((5+20+15+15+10+10+10+8)*'-'))
            print('|%5s|%20s|%15s|%15s|%10s|%10s|%10s' % ('Name'.center(5), 'sim name'.center(20), 'D_L'.center(15), 'M_GMC'.center(15), 'M_dif'.center(10), 'L_CII'.center(10), 'R_gal'.center(10)))
            print('|%5s|%20s|%15s|%15s|%10s|%10s|%10s' % (''.center(5), ''.center(20), '[Mpc]'.center(15), '[10^9 Msun]'.center(15), '[10^9 Msun]'.center(10), '[L_sun]'.center(10), '[kpc]'.center(10)))
            print('+%80s+' % ((5+20+15+15+10+10+10+8)*'-'))
            for gal_index in range(0,len(GR.galnames)):
                print('|%5s|%20s|%15s|%15s|%10s|%10s|%10s' % ('G'+str(gal_index+1),GR.galnames[gal_index].center(20),\
                    '{:.4f}'.format(GR.lum_dist[gal_index]),\
                    '{:.2e}'.format(GR.M_GMC[gal_index]/1e9),\
                    '{:.2e}'.format(GR.M_dif[gal_index]/1e9),\
                    '{:.2e}'.format(GR.L_CII[gal_index]),\
                    '{:.4f}'.format(GR.R_gal[gal_index])))
            print('+%80s+' % ((5+20+15+15+10+10+10+8)*'-'))


    def print_galaxy_properties(self,**kwargs):

        args        =   dict(gal_index=0)
        args        =   aux.update_dictionary(args,kwargs)
        for key,val in args.items():
            exec(key + '=val')

        print('\nProperties of Galaxy number %s, %s, at redshift %s' % (gal_index+1,self.galnames[gal_index],self.zreds[gal_index]))

        # Print these properties
        print('+%20s+%20s+%15s+' % ((20*'-'), (20*'-'), (15*'-')))
        print('|%20s|%20s|%15s|' % ('Property'.center(20), 'Value'.center(20), 'Name in code'.center(15)))
        print('+%20s+%20s+%15s+' % ((20*'-'), (20*'-'), (15*'-')))
        print('|%20s|%20s|%15s|' % ('Redshift'.center(20), '{:.3f}'.format(self.zreds[gal_index]), 'zred'.center(15)))
        print('|%20s|%20s|%15s|' % ('Radius'.center(20), '{:.3f}'.format(np.max(self.R_gal[gal_index])), 'R_gal'.center(15)))
        print('|%20s|%20s|%15s|' % ('Stellar mass'.center(20), '{:.3e}'.format(self.M_star[gal_index]), 'M_star'.center(15)))
        print('|%20s|%20s|%15s|' % ('ISM mass'.center(20), '{:.3e}'.format(self.M_gas[gal_index]), 'M_gas'.center(15)))
        print('|%20s|%20s|%15s|' % ('Dense gas mass fraction'.center(20), '{:.3e}'.format(self.f_dense[gal_index]*100.), 'f_dense'.center(15)))
        print('|%20s|%20s|%15s|' % ('DM mass'.center(20), '{:.3e}'.format(self.M_dm[gal_index]), 'M_dm'.center(15)))
        print('|%20s|%20s|%15s|' % ('SFR'.center(20), '{:.3f}'.format(self.SFR[gal_index]), 'SFR'.center(15)))
        print('|%20s|%20s|%15s|' % ('SFR surface density'.center(20), '{:.4f}'.format(self.SFRsd[gal_index]), 'SFRsd'.center(15)))
        print('|%20s|%20s|%15s|' % ('SFR-weighted Z'.center(20), '{:.4f}'.format(self.Zsfr[gal_index]), 'Zsfr'.center(15)))
        print('+%20s+%20s+%15s+' % ((20*'-'), (20*'-'), (15*'-')))
