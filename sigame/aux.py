# coding=utf-8
"""
Module: aux
"""

import numpy as np
import numexpr as ne
import pandas as pd
import pdb as pdb
import scipy as scipy
from scipy import optimize
import scipy.stats as stats
import scipy.integrate as integrate
from scipy.interpolate import RegularGridInterpolator,griddata
from scipy.interpolate import interp1d
import multiprocessing as mp
import os
import time
import matplotlib.pyplot as plt
import linecache as lc
import re as re
import sys as sys
# import cPickle
import sympy as sy
import astropy as astropy


#===============================================================================
"""  Load parameters """
#-------------------------------------------------------------------------------

def load_parameters():
    params                      =   np.load('temp_params.npy', allow_pickle=True).item() # insert external parent here
    return(params)

params = load_parameters()
g = globals()
for key,val in params.items():
    exec(key + '=val',g)

#===========================================================================
""" Paths to data etc. """
#---------------------------------------------------------------------------

def get_file_location(**kwargs):
    """
    Finds correct location and file name for a certain file type and galaxy
    """
    import os

    globals()['sim_type'],globals()['ISM_phase'],globals()['ISM_dc_phase'],globals()['map_type'] = '','','tot',''

    for key,val in kwargs.items():
        exec('globals()["' + key + '"]' + '=val')

    if gal_ob_present:
        try:
            globals()['zred'] = kwargs['gal_ob'].zred
            globals()['galname'] = kwargs['gal_ob'].name
            globals()['gal_index'] = kwargs['gal_ob'].gal_index
        except:
            # Assume gal_ob is actually a dictionary
            globals()['zred'] = kwargs['gal_ob']['zred']
            globals()['galname'] = kwargs['gal_ob']['galname']
            globals()['gal_index'] = kwargs['gal_ob']['gal_index']

    # sim particle data name and location
    if sim_type != '':
        path = d_data+'particle_data/sim_data/'
        filename = os.path.join(path, 'z'+'{:.2f}'.format(zred)+'_'+galname+'_sim.'+sim_type)
        try:
            path = d_data+'particle_data/sim_data/'
            if not os.path.exists(path):
                 os.mkdir(path)
            filename = os.path.join(path, 'z'+'{:.2f}'.format(zred)+'_'+galname+'_sim.'+sim_type)
        except:
            print("Need the following to create sim data name: gal_ob, sim_type")
            raise NameError

    # ISM particle data name and location
    if ISM_phase != '':
        try:
            path = d_data+'particle_data/ISM_data/'
            if not os.path.exists(path):
                 os.mkdir(path)
            filename = os.path.join(path, 'z'+'{:.2f}'.format(zred)+'_'+galname+'_'+ISM_phase+'.h5')
        except:
            print("Need the following to create ISM data name: gal_ob, ISM_type")
            raise NameError

    # Datacube name and location
    if ISM_dc_phase != 'tot':
        try:
            target_ext = target
            if target in lines:
                target_ext = 'L_' + target

            path = d_data+'datacubes/'
            if not os.path.exists(path):
                 os.mkdir(path)

            filename = os.path.join(path, '%s_%s_i%s_%s_%s.h5' % (z1, target_ext, inc_dc, galname, ISM_dc_phase))
        except:
            print("Need the following to create datacube name: z1, target, ISM_dc_phase, inc_dc, gal_ob")
            raise NameError

    if map_type != '':
        try:
            path = parent+'sigame/temp/maps/'
            if not os.path.exists(path):
                 os.mkdir(path)
            filename = os.path.join(path, '%s_%s_G%s.h5' % (z1, map_type, gal_index+1))
        except:
            print("Need the following to create map file name: z1, map_type, gal_ob")
            raise NameError

    if debug:
        print("Debugging mode...\n Filename: {:s}...".format(filename))
        import pdb; pdb.set_trace()

    return(filename)

#===========================================================================
""" For classes in general """
#---------------------------------------------------------------------------

def get_UV_str(z1,SFRsd):
    """
    Reads in SFR surface density and compares with SFR surface density of the MW.
    Then finds nearest FUV grid point in cloudy models and returns it as a string.
    """

    if z1 == 'z6':
        UV                  =   [5]
        UV_str              =   str(int(UV[np.argmin(np.abs(np.array(UV)-SFRsd/SFRsd_MW))]))
    if z1 == 'z2':
        UV                  =   [0.001,0.02]
        UV_str              =   str(UV[np.argmin(np.abs(np.array(UV)-SFRsd/SFRsd_MW))])
    if z1 == 'z0':
        UV                  =   [0.1,0.6]
        UV_str              =   str(UV[np.argmin(np.abs(np.array(UV)-SFRsd/SFRsd_MW))])

    return(UV_str)

def update_dictionary(values,new_values):
    """ updates the entries to values with entries from new_values that have
    matching keys. """
    for key in values:
        if key in new_values:
            values[key]     =   new_values[key]
    return values

def save_temp_file(data, subgrid=None, **kwargs):
    """
    Stores temporary files according to their sim or ISM type and stage of processing.
    """
    sim_type,ISM_phase,ISM_dc_phase = '','',''

    for key,val in kwargs.items():
        exec('globals()["' + key + '"]' + '=val')
        # for some reason, the variable ISM_phase is not recognized in the interpolate step if we don't explicitly define using the following (even though we are defining it as a global variable)...
        if key == 'ISM_phase' and 'ISM_phase' in globals():
            ISM_phase = val

        if key == 'ISM_dc_phase' and 'ISM_dc_phase' in globals():
            ISM_dc_phase = val

    filename    =   get_file_location(**kwargs)

    if subgrid or sim_type != '':
        print("saving to pickle: ", filename)
        data.to_pickle(filename)

        # work around so that galaxy.add_FUV() works... while doesn't break add_GMCs() and add_dif() in the subgrid step.
        if subgrid:
            ISM_phase, ISM_dc_phase, map_type = '', '', ''

    try:
        ISM_phase
        if ISM_phase != '':
            print("saving data h5store", filename)
            h5store(data, 'data', filename, **kwargs)
    except NameError:
        ISM_phase = None

    try:
        if ISM_dc_phase != '':
            print("saving data h5store", filename)
            h5store(data, 'data', filename, **kwargs)
    except NameError:
        ISM_dc_phase = None
        print(ISM_dc_phase)

    try:
        map_type
        if map_type != '':
            print("saving data h5store", filename)
            h5store(data, 'data', filename, **kwargs)
    except NameError:
        map_type = None

def load_temp_file(**kwargs):
    """Way to load metadata with dataframe
    """

    for key,val in kwargs.items():
        exec('globals()["' + key + '"]' + '=val')

    filename    =   get_file_location(**kwargs)
    # print(filename)
    try:
        data = pd.read_hdf(filename)
        try:
            data            =   data['data'][0]
        except:
            data            =   data
    except:
        try:
            data            =   pd.read_pickle(filename)
        except:
            if verbose: print('Did not find file at '+filename)
            data            =   0

    if verbose:
        if type(data) != int: print('Loaded file at %s' % filename)
    return data

def h5store(df, dc_name, filename, **kwargs):
    """Way to store metadata with dataframe
    """

    # try:
    #     metadata            =   df.metadata
    # except:
    #     metadata            =   {}

    # store = pd.HDFStore(filename)
    # store.put(dc_name, df)
    # store.get_storer(dc_name).attrs.metadata = metadata
    # store.close()

    # Let's try this, not metadata
    df.to_hdf(filename,dc_name)

#===========================================================================
""" For subgrid step"""
#---------------------------------------------------------------------------

def GMC_generator(index_range,Mneu,h,Mmin,Mmax,b,tol,nn,n_cores):
    '''
    Purpose
    ---------
    Draws mass randomly from GMC mass spectrum (called by create_GMCs)

    Arguments
    ---------
    '''

    if n_cores == 1:

        f1,newm_gmc,newx,newy,newz = [0]*len(index_range),[0]*len(index_range),[0]*len(index_range),[0]*len(index_range),[0]*len(index_range)
        #,np.zeros(len(index_range)),np.zeros(len(index_range)),np.zeros(len(index_range)),np.zeros(len(index_range))

        for i in index_range:

            ra      =   np.random.rand(nn)  # draw nn random numbers between 0 and 1
            Mg      =   np.zeros(nn)
            newm_gmc[i]     =   np.zeros(1)

            if Mneu[i] < Mmin+tol:
                newm_gmc[i]         =   Mneu[i]

            else:
                for ii in range(0,nn):
                    k           =   (1/(1-b)*(Mmax**(1-b)-Mmin**(1-b)))**(-1)       # Normalization constant (changing with Mmax)
                    Mg[ii]      =   (ra[ii]*(1-b)/k+Mmin**(1-b))**(1./(1-b))        # Draw mass (from cumulative distribution function)
                    if ii==0 and np.sum(newm_gmc[i])+Mg[ii] < Mneu[i]+tol:             # Is it the 1st draw and is the GMC mass below total neutral gas mass (M_neutral) available?
                        newm_gmc[i]         =   np.array(Mg[ii])                            # - then add that GMC mass
                    if ii>0 and np.sum(newm_gmc[i])+Mg[ii] < Mneu[i]+tol:              # Is the sum of GMC masses still below M_neutral+tolerance?
                        newm_gmc[i]         =   np.append(newm_gmc[i],Mg[ii])                       # - then add that GMC mass
                    if np.sum(newm_gmc[i]) > Mneu[i]-tol:                              # Is the sum of GMC masses above M_neutral-tolerance?
                        break                                                       # - fine! then stop here
                # If we're still very fra from Mneu, repeat loop!
                while np.sum(newm_gmc[i]) < 0.9*Mneu[i]:
                    ra      =   np.random.rand(nn)  # draw nn random numbers between 0 and 1
                    for ii in range(0,nn):
                        k           =   (1/(1-b)*(Mmax**(1-b)-Mmin**(1-b)))**(-1)       # Normalization constant (changing with Mmax)
                        Mg[ii]      =   (ra[ii]*(1-b)/k+Mmin**(1-b))**(1./(1-b))        # Draw mass (from cumulative distribution function)
                        if ii==0 and np.sum(newm_gmc[i])+Mg[ii] < Mneu[i]+tol:             # Is it the 1st draw and is the GMC mass below total neutral gas mass (M_neutral) available?
                            newm_gmc[i]         =   np.array(Mg[ii])                            # - then add that GMC mass
                        if ii>0 and np.sum(newm_gmc[i])+Mg[ii] < Mneu[i]+tol:              # Is the sum of GMC masses still below M_neutral+tolerance?
                            newm_gmc[i]         =   np.append(newm_gmc[i],Mg[ii])                       # - then add that GMC mass
                        if np.sum(newm_gmc[i]) > Mneu[i]-tol:                              # Is the sum of GMC masses above M_neutral-tolerance?
                            break                                                       # - fine! then stop here

            # Add SPH info to new DataFrame (same for all GMCs to this SPH parent)
            # Save indices of original SPH particles
            f1[i]      =   np.size(newm_gmc[i])
            if f1[i] == 1:
                if newm_gmc == 0:
                    print('No GMCs created for this one?')
                    pdb.set_trace()

            SPHindex    =   np.zeros(f1[i])+i
            # but change coordinates!!
            ra          =   np.random.rand(f1[i])  # draw nn random numbers between 0 and 1
            ra1         =   np.random.rand(f1[i])  # draw nn random numbers between 0 and 1
            ra2         =   np.random.rand(f1[i])  # draw nn random numbers between 0 and 1
            ra_R        =   ra*frac_h*h[i]
            ra_phi      =   ra1*2*np.pi
            ra_theta    =   ra2*np.pi
            ra          =   [ra_R*np.sin(ra_theta)*np.cos(ra_phi),+\
                ra_R*np.sin(ra_theta)*np.sin(ra_phi),+\
                ra_R*np.cos(ra_theta)]
            newx[i]        =   np.array(ra)[0,:]
            newy[i]        =   np.array(ra)[1,:]
            newz[i]        =   np.array(ra)[2,:]
            # Neutral mass that remains is distributed in equal fractions around the GMCs:
        return f1,newm_gmc,newx,newy,newz

    else:

        f1, newm_gmc, newx, newy, newz = [0], [0], [0], [0], [0]

        ra = np.random.rand(nn)  # draw nn random numbers between 0 and 1
        Mg = np.zeros(nn)
        newm_gmc = np.zeros(1)

        if Mneu[index_range] < Mmin + tol:
            newm_gmc = Mneu[index_range]

        else:
            for ii in range(0, nn):
                # Normalization constant (changing with Mmax)
                k = (1 / (1 - b) * (Mmax**(1 - b) - Mmin**(1 - b)))**(-1)

                # Draw mass (from cumulative distribution function)
                Mg[ii] = (ra[ii] * (1 - b) / k + Mmin**(1 - b))**(1. / (1 - b))
                # Is it the 1st draw and is the GMC mass below total neutral gas
                # mass (M_neutral) available?
                if ii == 0 and np.sum(newm_gmc) + Mg[ii] < Mneu[index_range] + tol:
                    # - then add that GMC mass
                    newm_gmc = np.array(Mg[ii])

                # Is the sum of GMC masses still below M_neutral+tolerance?
                if ii > 0 and np.sum(newm_gmc) + Mg[ii] < Mneu[index_range] + tol:
                    # - then add that GMC mass
                    newm_gmc = np.append(newm_gmc, Mg[ii])

                # Is the sum of GMC masses above M_neutral-tolerance?
                if np.sum(newm_gmc) > Mneu[index_range] - tol:
                    break                                                       # - fine! then stop here

            # If we're still very fra from Mneu, repeat loop!
            while np.sum(newm_gmc) < 0.9 * Mneu[index_range]:
                ra = np.random.rand(nn)  # draw nn random numbers between 0 and 1
                for ii in range(0, nn):
                    # Normalization constant (changing with Mmax)
                    k = (1 / (1 - b) * (Mmax**(1 - b) - Mmin**(1 - b)))**(-1)
                    # Draw mass (from cumulative distribution function)
                    Mg[ii] = (ra[ii] * (1 - b) / k + Mmin**(1 - b))**(1. / (1 - b))
                    # Is it the 1st draw and is the GMC mass below total neutral gas
                    # mass (M_neutral) available?
                    if ii == 0 and np.sum(newm_gmc) + Mg[ii] < Mneu[index_range] + tol:
                        # - then add that GMC mass
                        newm_gmc = np.array(Mg[ii])
                    # Is the sum of GMC masses still below M_neutral+tolerance?
                    if ii > 0 and np.sum(newm_gmc) + Mg[ii] < Mneu[index_range] + tol:
                        # - then add that GMC mass
                        newm_gmc = np.append(newm_gmc, Mg[ii])
                    # Is the sum of GMC masses above M_neutral-tolerance?
                    if np.sum(newm_gmc) > Mneu[index_range] - tol:
                        break                                                       # - fine! then stop here

        # Add SPH info to new DataFrame (same for all GMCs to this SPH parent)
        # Save indices of original SPH particles
        f1 = np.size(newm_gmc)
        if f1 == 1:
            if newm_gmc == 0:
                print('No GMCs created for this one?')
                pdb.set_trace()

        # but change coordinates!!
        ra = np.random.rand(f1)  # draw nn random numbers between 0 and 1
        ra1 = np.random.rand(f1)  # draw nn random numbers between 0 and 1
        ra2 = np.random.rand(f1)  # draw nn random numbers between 0 and 1
        ra_R = ra * frac_h * h[index_range]
        ra_phi = ra1 * 2 * np.pi
        ra_theta = ra2 * np.pi
        ra = [ra_R * np.sin(ra_theta) * np.cos(ra_phi), +
              ra_R * np.sin(ra_theta) * np.sin(ra_phi), +
              ra_R * np.cos(ra_theta)]
        newx = np.array(ra)[0, :]
        newy = np.array(ra)[1, :]
        newz = np.array(ra)[2, :]

        return index_range, f1,newm_gmc,newx,newy,newz

def get_FUV_grid_results(z1):
    """
    Loads results from starburst99 as a 2D grid of FUV luminosities
    for selected stellar Zs and ages.
    """

    # Read grid parameters for FUV grid of 1e4 Msun stellar populations
    FUVgrid                 =   pd.read_pickle(d_t+'FUV/FUV_'+z1+'_noneb')
    # Read corresponding [age,Z,L_FUV] values
    # FUV                     =   pd.read_pickle(d_t+'FUV/FUV_'+z1+'_noneb')
    # l_FUV                   =   FUV['L_FUV'].values
    # l_FUV                   =   l_FUV.reshape((len(grid['Ages']),len(grid['Zs'])))

    return(FUVgrid['Z'].values,FUVgrid['Age'].values,FUVgrid['L_FUV'].values)

def FUVfunc(i,simstar,simgas,L_FUV):
    """
    Sums up total FUV flux from all nearby stars within 1 h
    """
    dist        =   rad((simstar[pos]-simgas.loc[i][pos]).astype(np.float64),pos).values
    dist[dist == 0]     =   1000
    return i, sum(L_FUV[dist < simgas['h'][i]]/(4*np.pi*dist[dist < simgas['h'][i]]**2))

def Pfunc(i,simgas1,simgas,simstar,m_gas,m_star):
    # Distance to other gas particles in disk plane:
    dist1       =   rad((simgas1[posxy]-simgas.loc[i][posxy]).astype(np.float64),posxy).values
    m_gas1      =   m_gas[dist1 < simgas['h'][i]]
    p,surf_gas,surf_star,sigma_gas,sigma_star,vel_disp_gas = [0 for j in range(0,6)]
    if len(m_gas1) >= 1:
        surf_gas    =   sum(m_gas1)/(np.pi*simgas['h'][i]**2.)
        sigma_gas   =   np.std(simgas1.loc[dist1 < simgas['h'][i]]['vz'])
        # Distance to other star particles in disk plane:
        dist2       =   rad((simstar[posxy]-simgas.loc[i][posxy]).astype(np.float64),posxy).values
        m_star1     =   m_star[dist2 < simgas['h'][i]]
        if len(m_star1) >= 1:
            surf_star   =   sum(m_star1)/(np.pi*simgas['h'][i]**2.)
            sigma_star  =   np.std(simstar.loc[dist2 < simgas['h'][i]]['vz'])
        # Total velocity dispersion of gas
        vel_disp_gas   =   np.std(np.sqrt((simgas1.loc[dist1 < simgas['h'][i]]['vx'].values)**2+\
                            (simgas1.loc[dist1 < simgas['h'][i]]['vy'].values)**2+\
                            (simgas1.loc[dist1 < simgas['h'][i]]['vz'].values)**2))
        if len(simstar.loc[dist2 < simgas['h'][i]]) == 0: sigma_star = 0
        if sigma_star != 0: p = np.pi/2.*G_grav*surf_gas*(surf_gas+(sigma_gas/sigma_star)*surf_star)/1.65
        if sigma_star == 0: p = np.pi/2.*G_grav*surf_gas*(surf_gas)/1.65

    else:
        if simgas['SFR'][i] > 0:
            surf_gas    =   simgas['m'][i]/(np.pi*simgas['h'][i]**2.)
            m_star1     =   m_star[dist2 < simgas['h'][i]]
            if len(m_star1) >= 1:
                surf_star   =   sum(m_star1)/(np.pi*simgas['h'][i]**2.)

    return i, p,surf_gas,surf_star,sigma_gas,sigma_star,vel_disp_gas

#===========================================================================
""" For interpolation step"""
#---------------------------------------------------------------------------

def interpolate_in_GMC_models(GMCgas,cloudy_grid_param,cloudy_grid):

    # Parameters used in the interpolation
    int_parameters      =   ['m','FUV','Z','P_ext']

    # Looking at GMC properties in log space
    # GMCgas              =   GMCgas[0:144]
    N_GMCs              =   len(GMCgas)
    GMCgas1             =   np.log10(GMCgas[int_parameters])
    # print('Interpolating for %s GMCs' % N_GMCs)

    # Make sure we don't go outside of grid:
    for _ in int_parameters:
        GMCgas1.ix[GMCgas1[_] < min(cloudy_grid_param[_+'s']),_] = min(cloudy_grid_param[_+'s'])
        GMCgas1.ix[GMCgas1[_] > max(cloudy_grid_param[_+'s']),_] = max(cloudy_grid_param[_+'s'])

    # Values used for interpolation in cloudy models:
    GMCs                        =   np.column_stack((GMCgas1['m'].values,GMCgas1['FUV'].values,GMCgas1['Z'].values,GMCgas1['P_ext'].values))

    # List of target items that we are interpolating for:
    target_list                 =   ['Mgmc_fit','FUV_fit','Z_fit','P_ext_fit',\
                                    'f_H2','f_HI','m_dust','m_H','m_H2','m_HI','m_HII',\
                                    'm_C','m_CII','m_CIII','m_CIV','m_CO',\
                                    'Tkmw','nHmw','nHmin','nHmax']

    for line in lines: target_list.append('L_'+line)

    cloudy_grid['Mgmc_fit']     =   cloudy_grid['Mgmc']
    cloudy_grid['FUV_fit']      =   cloudy_grid['FUV']
    cloudy_grid['Z_fit']        =   cloudy_grid['Z']
    cloudy_grid['P_ext_fit']    =   cloudy_grid['P_ext']

    # Make grid of model results corresponding target grid values and interpolate in it
    for target in target_list:
        target_grid                 =   np.zeros([len(cloudy_grid_param['Mgmcs']),len(cloudy_grid_param['FUVs']),len(cloudy_grid_param['Zs']),len(cloudy_grid_param['P_exts'])])
        i                           =   0
        for i1 in range(0,len(cloudy_grid_param['Mgmcs'])):
            for i2 in range(0,len(cloudy_grid_param['FUVs'])):
                for i3 in range(0,len(cloudy_grid_param['Zs'])):
                    for i4 in range(0,len(cloudy_grid_param['P_exts'])):
                        target_grid[i1,i2,i3,i4]    =   cloudy_grid[target][i]
                        i                           +=  1

        # Make function that will do the interpolation:
        interp                      =   RegularGridInterpolator((cloudy_grid_param['Mgmcs'],cloudy_grid_param['FUVs'],cloudy_grid_param['Zs'],cloudy_grid_param['P_exts']), target_grid)
        # And interpolate:
        GMCgas[target]              =   interp(GMCs)

    GMCgas['f_H2']              =   GMCgas['m_H2']/GMCgas['m_H']
    GMCgas['f_HI']              =   GMCgas['m_HI']/GMCgas['m_H']
    GMCgas['f_HII']             =   GMCgas['m_HII'] / GMCgas['m_H']

    # Find index of closest models!
    model_numbers               =   np.zeros([len(cloudy_grid_param['Mgmcs']),len(cloudy_grid_param['FUVs']),len(cloudy_grid_param['Zs']),len(cloudy_grid_param['P_exts'])])
    j                           =   0
    for i1 in range(0,len(cloudy_grid_param['Mgmcs'])):
        for i2 in range(0,len(cloudy_grid_param['FUVs'])):
            for i3 in range(0,len(cloudy_grid_param['Zs'])):
                for i4 in range(0,len(cloudy_grid_param['P_exts'])):
                    model_numbers[i1,i2,i3,i4]      =   j
                    j                               +=  1
    closest_model_i             =   np.zeros(N_GMCs)
    for i in range(0,N_GMCs):
        Mgmc                        =   GMCgas['m'][i]
        FUV                         =   GMCgas['FUV'][i]
        Z                           =   GMCgas['Z'][i]
        P_ext                       =   GMCgas['P_ext'][i]
        i_Mgmc                      =   np.argmin(abs(10.**cloudy_grid_param['Mgmcs']-Mgmc))
        i_FUV                       =   np.argmin(abs(10.**cloudy_grid_param['FUVs']-FUV))
        i_Z                         =   np.argmin(abs(10.**cloudy_grid_param['Zs']-Z))
        i_P_ext                     =   np.argmin(abs(10.**cloudy_grid_param['P_exts']-P_ext))
        closest_model_i[i]          =   int(model_numbers[i_Mgmc,i_FUV,i_Z,i_P_ext])
    GMCgas['closest_model_i']   =   closest_model_i

    # Check sum of L_[CII] using closest model number
    L_tot               =   0.
    for i in range(0,N_GMCs):
        L_tot               +=   cloudy_grid['L_CII'][GMCgas['closest_model_i'][i]]
    print('Sum of L_[CII] using closest model number: %s Lsun' % L_tot)
    print('Sum of L_[CII] using interpolation: %s Lsun' % np.sum(GMCgas['L_CII']))


    print(str.format("{0:.2f}",sum(GMCgas['f_H2']*GMCgas['m_H'])/sum(GMCgas['m_H'])*100.)+' % of Hydrogen is molecular Hydrogen (excluding H in other H-bond molecules)')
    print(str.format("{0:.2f}",sum(GMCgas['f_HI']*GMCgas['m_H'])/sum(GMCgas['m_H'])*100.)+' % of Hydrogen is atomic Hydrogen (excluding H in other H-bond molecules)')
    print(str.format("{0:.2f}",sum(GMCgas['f_HII']*GMCgas['m_H'])/sum(GMCgas['m_H'])*100.)+' % of Hydrogen is ionized Hydrogen (excluding H in other H-bond molecules)')

    print(str.format("{0:.2f}",sum(GMCgas['m_H2'])/sum(GMCgas['m_H'])*100.)+' % of Hydrogen is molecular hydrogen (excluding H in other H-bond molecules)')
    print(str.format("{0:.2f}",sum(GMCgas['m_HI'])/sum(GMCgas['m_H'])*100.)+' % of Hydrogen is atomic hydrogen (excluding H in other H-bond molecules)')
    print(str.format("{0:.2f}",sum(GMCgas['m_HII'])/sum(GMCgas['m_H'])*100.)+' % of Hydrogen is ionized hydrogen (excluding H in other H-bond molecules)')

    print('Line emission: (Some might be nan if not enough line emission was found for this phase)')
    for line in lines:
        print('Total L_'+line+': %.2e L_sun' % (sum(GMCgas['L_'+line])))

    # Add some attributes to GMC dataframe
    GMCgas.M_dust               =   np.sum(GMCgas['m_dust'])
    GMCgas.M_H2                 =   np.sum(GMCgas['m_H2'])
    GMCgas.M_HI                 =   np.sum(GMCgas['m_HI'])
    GMCgas.M_HII                =   np.sum(GMCgas['m_HII'])
    GMCgas.M_C                  =   np.sum(GMCgas['m_C'])
    GMCgas.M_CII                =   np.sum(GMCgas['m_CII'])
    GMCgas.M_CIII               =   np.sum(GMCgas['m_CIII'])
    GMCgas.M_CO                 =   np.sum(GMCgas['m_CO'])

    return(GMCgas)

def interpolate_in_dif_models(difgas,cloudy_grid_param,cloudy_grid):

    # Parameters used in the interpolation
    int_parameters      =   ['nH','R','Z','Tk']

    # Looking at dif properties in log space
    N_difs              =   len(difgas)
    difgas1             =   np.log10(difgas[int_parameters])

    # Make sure we don't go outside of grid:
    for _ in int_parameters:
        difgas1.ix[difgas1[_] < min(cloudy_grid_param[_+'s']),_] = min(cloudy_grid_param[_+'s'])
        difgas1.ix[difgas1[_] > max(cloudy_grid_param[_+'s']),_] = max(cloudy_grid_param[_+'s'])

    # Values used for interpolation in cloudy models:
    difs                        =   np.column_stack((difgas1['nH'].values,difgas1['R'].values,difgas1['Z'].values,difgas1['Tk'].values))

    # List of target items that we are interpolating for:
    target_list                 =   ['index','m_dust','Tk_DNG','Tk_DIG',\
                                    'fm_DNG','fm_H_DNG','fm_HI_DNG','fm_HII_DNG','fm_H2_DNG',\
                                    'fm_C_DNG','fm_CII_DNG','fm_CIII_DNG','fm_CIV_DNG','fm_CO_DNG',\
                                    'fm_H_DIG','fm_HI_DIG','fm_HII_DIG','fm_H2_DIG',\
                                    'fm_C_DIG','fm_CII_DIG','fm_CIII_DIG','fm_CIV_DIG','fm_CO_DIG']

    for line in lines: target_list.append('L_'+line)
    for line in lines: target_list.append('L_'+line+'_int')
    for line in lines: target_list.append('f_'+line+'_DNG')

    cloudy_grid['nH_fit']       =   cloudy_grid['nH']
    cloudy_grid['R_fit']        =   cloudy_grid['R']
    cloudy_grid['Z_fit']        =   cloudy_grid['Z']
    cloudy_grid['Tk_fit']       =   cloudy_grid['Tk']

    # Make grid of model results corresponding target grid values and interpolate in it
    for target in target_list:
        target_grid                 =   np.zeros([len(cloudy_grid_param['nHs']),len(cloudy_grid_param['Rs']),len(cloudy_grid_param['Zs']),len(cloudy_grid_param['Tks'])])
        i                           =   0
        for i1 in range(0,len(cloudy_grid_param['nHs'])):
            for i2 in range(0,len(cloudy_grid_param['Rs'])):
                for i3 in range(0,len(cloudy_grid_param['Zs'])):
                    for i4 in range(0,len(cloudy_grid_param['Tks'])):
                        target_grid[i1,i2,i3,i4]    =   cloudy_grid[target][i]
                        i                           +=  1

        # Make function that will do the interpolation:
        interp                      =   RegularGridInterpolator((cloudy_grid_param['nHs'],cloudy_grid_param['Rs'],cloudy_grid_param['Zs'],cloudy_grid_param['Tks']), target_grid, method = 'linear')
        # And interpolate:
        difgas[target]              =   interp(difs)

    # Find index of closest models!
    model_numbers               =   np.zeros([len(cloudy_grid_param['nHs']),len(cloudy_grid_param['Rs']),len(cloudy_grid_param['Zs']),len(cloudy_grid_param['Tks'])])
    j                           =   0
    for i1 in range(0,len(cloudy_grid_param['nHs'])):
        for i2 in range(0,len(cloudy_grid_param['Rs'])):
            for i3 in range(0,len(cloudy_grid_param['Zs'])):
                for i4 in range(0,len(cloudy_grid_param['Tks'])):
                    model_numbers[i1,i2,i3,i4]      =   j
                    j                               +=  1
    closest_model_i             =   np.zeros(N_difs)
    for i in range(0,N_difs):
        nH                          =   difgas['nH'][i]
        R                           =   difgas['R'][i]
        Z                           =   difgas['Z'][i]
        Tk                          =   difgas['Tk'][i]
        i_nH                        =   np.argmin(abs(10.**cloudy_grid_param['nHs']-nH))
        i_R                         =   np.argmin(abs(10.**cloudy_grid_param['Rs']-R))
        i_Z                         =   np.argmin(abs(10.**cloudy_grid_param['Zs']-Z))
        i_Tk                        =   np.argmin(abs(10.**cloudy_grid_param['Tks']-Tk))
        closest_model_i[i]          =   int(model_numbers[i_nH,i_R,i_Z,i_Tk])
    difgas['closest_model_i']   =   closest_model_i

    # Check sum of L_[CII] using closest model number
    L_tot               =   0.
    for i in range(0,N_difs):
        L_tot               +=   cloudy_grid['L_CII'][difgas['closest_model_i'][i]]
    print('Sum of L_[CII] using closest model number: %s Lsun' % L_tot)
    print('Sum of L_[CII] using interpolation: %s Lsun' % np.sum(difgas['L_CII']))


    # Calculate mass fractions in ionized vs neutral diffuse gas:
    difgas['m_DNG']         =   difgas['m']*difgas['fm_DNG']
    difgas['m_DIG']         =   difgas['m']*(1-difgas['fm_DNG'])
    for mass in ['H','HI','H2','HII','C','CII','CIII','CIV','CO']:
        difgas['m_'+mass+'_DNG']       =   difgas['m_DNG']*difgas['fm_'+mass+'_DNG']
        difgas['m_'+mass+'_DIG']       =   difgas['m_DIG']*difgas['fm_'+mass+'_DIG']

    # Check masses
    print('Total diffuse gas mass: %s Msun' % (sum(difgas['m'])))
    print('Total diffuse neutral gas (DNG) mass: %s or %s %%' % (np.sum(difgas['m_DNG']),np.sum(difgas['m_DNG'])/np.sum(difgas['m'])*100.))
    print('Total diffuse ionized gas (DIG) mass: %s or %s %%' % (np.sum(difgas['m_DIG']),np.sum(difgas['m_DIG'])/np.sum(difgas['m'])*100.))
    print('Check sum: %s %%' % (sum(difgas['m'])/(sum(difgas['m_DNG'])+sum(difgas['m_DIG']))*100.))

    # Prune the dataframe a bit:
    difgas1                 =   difgas[['x','y','z','vx','vy','vz','vel_disp_gas',\
                                'm','nH','R','Z','Tk','FUV',\
                                'm_DNG','m_DIG','m_H_DNG','m_H_DIG','m_HI_DNG','m_HI_DIG',\
                                'm_H2_DNG','m_H2_DIG','m_HII_DNG','m_HII_DIG',\
                                'm_C_DNG','m_C_DIG','m_CII_DNG','m_CII_DIG','m_CIV_DNG','m_CIV_DIG','m_CIII_DNG','m_CIII_DIG','m_CO_DNG','m_CO_DIG',\
                                'm_dust','Tk_DNG','Tk_DIG','closest_model_i']].copy()

    # Store line emission from ionized vs neutral diffuse gas
    print('Line emission: (Some might be nan if not enough line emission was found for this phase)')
    for line in lines:
        difgas1['L_'+line]              =   difgas['L_'+line]
        difgas1['L_'+line+'_int']       =   difgas['L_'+line+'_int']
        difgas1['L_'+line+'_DNG']       =   difgas['L_'+line]*difgas['f_'+line+'_DNG']
        difgas1['L_'+line+'_DNG_int']   =   difgas['L_'+line+'_int']*difgas['f_'+line+'_DNG']
        difgas1['L_'+line+'_DIG']       =   difgas['L_'+line]*(1.-difgas['f_'+line+'_DNG'])
        difgas1['L_'+line+'_DIG_int']   =   difgas['L_'+line+'_int']*(1.-difgas['f_'+line+'_DNG'])
        print('Total L_'+line+': %.2e L_sun' % (sum(difgas1['L_'+line])))

    # Dust masses:
    difgas1['m_dust_DNG']       =   difgas1['m_dust']/difgas1['m']*difgas1['m_DNG']
    difgas1['m_dust_DIG']       =   difgas1['m_dust']/difgas1['m']*difgas1['m_DIG']

    # Gas phases within diffuse gas:
    print('In DNG:')
    print('%.2f %% of hydrogen is molecular' % (np.sum(difgas1['m_H2_DNG'])/np.sum(difgas1['m_H_DNG'])*100.))
    print('%.2f %% of hydrogen is atomic' % (np.sum(difgas1['m_HI_DNG'])/np.sum(difgas1['m_H_DNG'])*100.))
    print('%.2f %% of hydrogen is ionized' % (np.sum(difgas1['m_HII_DNG'])/np.sum(difgas1['m_H_DNG'])*100.))
    print('In DIG:')
    print('%.2f %% of hydrogen is molecular' % (np.sum(difgas1['m_H2_DIG'])/np.sum(difgas1['m_H_DIG'])*100.))
    print('%.2f %% of hydrogen is atomic' % (np.sum(difgas1['m_HI_DIG'])/np.sum(difgas1['m_H_DIG'])*100.))
    print('%.2f %% of hydrogen is ionized' % (np.sum(difgas1['m_HII_DIG'])/np.sum(difgas1['m_H_DIG'])*100.))

    # Add some attributes to dif dataframe
    difgas1.M_DNG               =   np.sum(difgas1['m_DNG'])
    difgas1.M_DIG               =   np.sum(difgas1['m_DIG'])
    difgas1.M_dust_DNG          =   np.sum(difgas1['m_dust_DNG'])
    difgas1.M_dust_DIG          =   np.sum(difgas1['m_dust_DIG'])

    return(difgas1)

#===========================================================================
""" For datacube step"""
#---------------------------------------------------------------------------

def get_v_axis():
    """Returns velocity axis.
    """
    v_axis              =   np.arange(-v_max,v_max+v_max/1e6,v_res)
    dv                  =   v_axis[1]-v_axis[0]
    v_axis              =   v_axis[0:len(v_axis)-1]+v_res/2.
    return v_axis

def get_x_axis_kpc():
    """Returns position (x or y) axis in kpc.
    """
    x_axis              =   np.arange(-x_max_pc,x_max_pc+x_max_pc/1e6,x_res_pc)
    dx                  =   x_axis[1]-x_axis[0]
    x_axis_kpc          =   (x_axis[0:len(x_axis)-1]+x_res_pc/2.)/1000.
    return x_axis_kpc

def mk_datacube(gal_ob,dataframe,ISM_dc_phase='GMC'):
    """
    Creates a datacube for specific galaxy, ISM phase and target

    Parameters
    ----------
    gal_ob : class object
        Galaxy object
    dataframe : pandas dataframe
        Dataframe containing particle data for this ISM phase
    ISM_dc_phase : str
        ISM phase, default: 'GMC'
    """

    print('\nNOW CREATING DATACUBES OF %s FOR ISM PHASE %s' % (target,ISM_dc_phase))

    # Derive an extension for the file names
    if ISM_dc_phase == 'GMC': ISM_ext = ISM_dc_phase
    if ISM_dc_phase != 'GMC': ISM_ext = ISM_dc_phase + '_' + gal_ob.UV_str + 'UV'
    ext                 =   '%s_%s_%s' % (target,z1,ISM_ext)
    if target == 'Z': ext = ext.replace('Z_','m_') # for metallicity, we just need the mass profiles

    # Create velocity and position axes
    v_axis              =   get_v_axis()
    x_axis_kpc          =   get_x_axis_kpc()

    print(' 1) Create (if not there already) radial profiles of all model clouds')
    rad_prof_path = d_cloud_profiles+'rad_profs_%s.npy' % ext
    print('Looking for: %s ' % rad_prof_path)
    if not os.path.exists(rad_prof_path):
        mk_cloud_rad_profiles(ISM_dc_phase=ISM_dc_phase,target=target,FUV=gal_ob.UV_str,rad_prof_path=rad_prof_path)
        # mk_cloud_rad_profiles(ISM_dc_phase=ISM_dc_phase,target=target,FUV=gal_ob.UV_str,rad_prof_path=rad_prof_path)

    print(' 2) Load clouds in galaxy')
    t1 = time.clock()
    global clouds
    clouds                                      =   load_clouds(dataframe,target,ISM_dc_phase)
    print('Total number of clouds to be drizzled: %s' % len(clouds))

    print('3) Check how many models are available')
    model_rad_profs         =   np.load(rad_prof_path, allow_pickle=True)
    models_r_pc             =   model_rad_profs[1,:,:]  # pc
    model_index             =   clouds['closest_model_i'].values
    clouds_r_pc             =   [models_r_pc[int(i)] for i in model_index]
    clouds_R_pc             =   np.max(clouds_r_pc,axis=1)
    print('%s corresponding models did not finish ' % len(clouds_R_pc[clouds_R_pc == 0]))
    print('out of %s clouds' % len(clouds))

    print('4) Drizzle onto datacube')
    if N_cores == 1:
        print('(Not using multiprocessing - 1 core in use)')
        # Make one for loop
        start_end           =   [0,len(clouds)]
        dc                  =   drizzle(start_end,v_axis,x_axis_kpc,rad_prof_path,ISM_dc_phase,target,gal_ob.UV_str,gal_ob.zred)

    if N_cores > 1:
        # Set up a pool of workers to run more for loops in parallel
        pool                =   mp.Pool(processes = N_cores)        # 8 processors on my Mac Pro, 16 on Betty
        n_clouds            =   500

        if len(clouds) < n_clouds: n_clouds = len(clouds)
        work_division       =   [(_*n_clouds,(_+1)*n_clouds) for _ in range(0,int(np.floor(len(clouds)/n_clouds)))]

        if (work_division[-1][1] - len(clouds)) > 0:
            work_division.append((work_division[-1][1],len(clouds)))
        # result = drizzle(work_division[0],v_axis,x_axis_kpc,rad_prof_path,ISM_dc_phase,target,gal_ob.UV_str,gal_ob.zred) # for testing
        # a = asdaf
        results             =   [pool.apply_async(drizzle,args=(start_end,v_axis,x_axis_kpc,rad_prof_path,ISM_dc_phase,target,gal_ob.UV_str,gal_ob.zred)) for start_end in work_division]
        print('(Using multiprocessing on %s clouds at a time! %s cores in use)' % (n_clouds,N_cores))
        pool.close()
        pool.join()
        sub_ims             =   [p.get() for p in results]
        dc                  =   sub_ims[0]
        for _ in range(1,len(sub_ims)):
            dc                  +=      sub_ims[_]
        t2 = time.clock()
        dt = t2-t1
        if dt < 60: print('Time it took to do this ISM phase: %.2f s' % (t2-t1))
        if dt > 60: print('Time it took to do this ISM phase: %.2f min' % ((t2-t1)/60.))

    dc                  =   np.nan_to_num(dc)
    dc_sum              =   np.sum(dc)
    if target in ['L_' + l for l in lines]:
        print(target + ' = %.2e Lsun from interpolated clouds' % (np.sum(clouds[target][0:len(clouds)])))
        print(target + ' = %.2e Lsun from datacube (will be smaller due to missing radial profiles for models that did not run)' % np.sum(dc_sum))
    else:
        print(target + ' = %.2e [unit] from interpolated clouds' % (np.sum(clouds[target][0:len(clouds)])))
        print(target + ' = %.2e [unit] from datacube (will be smaller due to missing radial profiles for models that did not run)' % np.sum(dc_sum))
    return(dc,dc_sum)

def drizzle(start_end,v_axis,x_axis_kpc,rad_prof_path,ISM_dc_phase,target,FUV,zred,plotting=True,verbose=False,checkplot=False):

    '''
    Purpose
    ---------
    Drizzle *all* clouds onto galaxy grid in velocity and space

    '''
    # Only look at these clouds:
    clouds1                  =   clouds[start_end[0]:start_end[1]]
    N_clouds                =   len(clouds1)
    print('Now drizzling %s %s clouds (# %s to %s) onto galaxy grid at %s x MW FUV...' % (N_clouds,ISM_dc_phase,start_end[0],start_end[1]-1,FUV))

    # Empty numpy array to hold result
    lenv,lenx               =   len(v_axis),len(x_axis_kpc)
    result                  =   np.zeros([lenv,lenx,lenx])

    if verbose:
        if ISM_dc_phase == 'GMC':
            model_path      =   d_cloud_models+'GMC/output/GMC_'
            models          =   pd.read_pickle(d_cloud_models+'cloud_models/GMCgrid'+ ext_DENSE + '_' + z1+'_em.models')

        if ISM_dc_phase in ['DNG','DIG']:
            model_path      =   d_cloud_models+'dif/output/dif_'
            models          =   pd.read_pickle(d_cloud_models+'cloud_models/dif/grids/difgrid_'+FUV+'UV'+ ext_DIFFUSE + '_' + z1+'_em.models')

    # For reference, get total values of "target" from interpolation
    if target == 'm':
        interpolation_result =  clouds1[target].values
    if target == 'Z':
        interpolation_result =  clouds1['m'].values*clouds1['Z'].values # Do a mass-weighted mean of metallicities
    else:
        # target = 'L_' + target
        if ISM_dc_phase in ['GMC']:        interpolation_result    =   clouds1[target].values
        if ISM_dc_phase in ['DNG','DIG']:  interpolation_result    =   clouds1[target+'_'+ISM_dc_phase].values

    # LOADING
    model_rad_profs         =   np.load(rad_prof_path, allow_pickle=True)
    all_inner_r_pc = model_rad_profs[0,:,0]
    # print('%s models did not finish ' % len(all_inner_r_pc[all_inner_r_pc == 0]))

    models_r_pc             =   model_rad_profs[1,:,:]  # pc
    models_SB               =   model_rad_profs[0,:,:]  # Lsun/pc^2
    # Assign these models to clouds in the galaxy:
    model_index             =   clouds1['closest_model_i'].values
    clouds_r_pc             =   [models_r_pc[int(i)] for i in model_index]
    clouds_SB               =   [models_SB[int(i)] for i in model_index]
    clouds_R_pc             =   np.max(clouds_r_pc,axis=1)
    vel_disp_gas            =   clouds1['vel_disp_gas'].values
    v_proj                  =   clouds1['v_proj'].values
    x_cloud                 =   clouds1['x'].values
    y_cloud                 =   clouds1['y'].values

    # SETUP
    # Some things we will need
    v_max, v_res            =   max(v_axis),v_axis[1]-v_axis[0]
    fine_v_axis             =   np.arange(-v_max,v_max+v_max/1e6,v_res/8.)
    x_res_kpc               =   x_axis_kpc[1]-x_axis_kpc[0]
    x_res_pc                =   x_res_kpc*1000.
    npix_highres_def        =   9# highres pixels per galaxy pixel, giving 3 pixels on either side of central pixel (of resolution x_res_pc)
    x_res_kpc_highres       =   x_res_kpc/npix_highres_def # size of high resolution pixel
    x_res_pc_highres        =   x_res_kpc_highres*1000. # size of high resolution pixel
    pix_area_highres_kpc    =   x_res_kpc_highres**2. # area of high resolution pixel
    pix_area_highres_pc     =   x_res_pc_highres**2. # area of high resolution pixel

    # What galaxy pixel center comes closest to this cloud center?
    min_x                   =   np.min(x_axis_kpc)
    range_x                 =   np.max(x_axis_kpc) - min_x
    x_index                 =   np.round((x_cloud-min_x)/range_x*(lenx-1)).astype(int)
    y_index                 =   np.round((y_cloud-min_x)/range_x*(lenx-1)).astype(int)


    # ----------------------------------------------
    # SMALL CLOUDS
    # Clouds that are "very" small compared to pixel size in galaxy image
    # give all their luminosity to one galaxy pixel
    small_cloud_index       =   [(0 < clouds_R_pc) & (clouds_R_pc <= x_res_kpc*1000./8.)][0]
    print('%s small clouds, unresolved by galaxy pixels' % (len(small_cloud_index[small_cloud_index == True])))
    small_cloud_x_index     =   np.extract(small_cloud_index,x_index)
    small_cloud_y_index     =   np.extract(small_cloud_index,y_index)
    small_cloud_targets     =   np.extract(small_cloud_index,interpolation_result)
    small_cloud_vdisp_gas   =   np.extract(small_cloud_index,vel_disp_gas)
    small_cloud_v_proj      =   np.extract(small_cloud_index,v_proj)
    pixels_outside          =   0
    for target1,vel_disp_gas1,v_proj1,i_x,i_y in zip(small_cloud_targets,small_cloud_vdisp_gas,small_cloud_v_proj,small_cloud_x_index,small_cloud_y_index):
        vel_prof                =       mk_cloud_vel_profile(v_proj1,vel_disp_gas1,fine_v_axis,v_axis)
        try:
            result[:,i_x,i_y]       +=      vel_prof*target1
        except:
            pixels_outside      +=  1

    # ----------------------------------------------
    # LARGER CLOUDS
    # Cloud is NOT very small compared to pixel size in galaxy image
    # => resolve cloud and split into nearby pixels
    large_cloud_index       =   [(0 < clouds_R_pc) & (clouds_R_pc > x_res_pc/8.)][0] # boolean array
    print('%s large clouds, resolved by galaxy pixels' % (len(large_cloud_index[large_cloud_index == True])))

    # Total number of highres pixels to (more than) cover this cloud
    Npix_highres            =   2.2*np.extract(large_cloud_index,clouds_R_pc)/(x_res_pc_highres)
    Npix_highres            =   np.ceil(Npix_highres).astype(int)
    # Count highres cloud pixels in surrounding galaxy pixels
    max_pix_dif             =   np.extract(large_cloud_index,clouds_R_pc)/x_res_pc
    max_pix_dif             =   np.ceil(max_pix_dif).astype(int)
    highres_axis_max        =   (np.array(Npix_highres)*x_res_pc_highres)/2.
    large_cloud_interpol    =   np.extract(large_cloud_index,interpolation_result)
    large_cloud_model_index =   np.extract(large_cloud_index,model_index)
    large_cloud_x_index     =   np.extract(large_cloud_index,x_index)
    large_cloud_y_index     =   np.extract(large_cloud_index,y_index)
    large_cloud_targets     =   np.extract(large_cloud_index,clouds1[target].values)
    large_cloud_vdisp_gas   =   np.extract(large_cloud_index,vel_disp_gas)
    large_cloud_v_proj      =   np.extract(large_cloud_index,v_proj)
    large_cloud_R_pc        =   np.extract(large_cloud_index,clouds_R_pc)
    large_models_r_pc       =   [models_r_pc[int(_)] for _ in large_cloud_model_index]
    large_models_SB         =   [models_SB[int(_)] for _ in large_cloud_model_index]
    i                       =   0
    pixels_outside          =   0
    # overwrite_me            =   np.zeros([lenv,lenx,lenx])

    for target1,vel_disp_gas1,v_proj1,i_x,i_y,npix_highres in zip(large_cloud_targets,large_cloud_vdisp_gas,large_cloud_v_proj,large_cloud_x_index,large_cloud_y_index,Npix_highres):
        if np.sum(large_models_SB[i]) > 0:
            # overwrite_me        =   overwrite_me*0.
            vel_prof            =   mk_cloud_vel_profile(v_proj1,vel_disp_gas1,fine_v_axis,v_axis)
            # create grid of coordinates for high-res image of cloud:
            # npix_highres = npix_highres*10. # CHECK
            # x_res_pc_highres = x_res_pc_highres/10. # CHECK
            x_highres_axis      =   np.linspace(-highres_axis_max[i],highres_axis_max[i],npix_highres)
            x_highres_mesh, y_highres_mesh = np.mgrid[slice(-max(x_highres_axis) + x_res_pc_highres/2., max(x_highres_axis) - x_res_pc_highres/2. + x_res_pc_highres/1e6, x_res_pc_highres),\
                            slice(-max(x_highres_axis) + x_res_pc_highres/2., max(x_highres_axis) - x_res_pc_highres/2. + x_res_pc_highres/1e6, x_res_pc_highres)]
            radius              =   np.sqrt(x_highres_mesh**2+y_highres_mesh**2)
            # create high-res image of this cloud, evaluating surface brightness at the center of each high-res pixel
            interp_func_r       =   interp1d(large_models_r_pc[i],large_models_SB[i],fill_value=large_models_SB[i][-1],bounds_error=False)
            im_cloud            =   interp_func_r(radius)
            im_cloud[radius > large_cloud_R_pc[i]]  =   0.

            # Remove "per area" from image units [Lsun/pc^2 -> Lsun or Msun/pc^2 -> Msun]
            im_cloud            =   im_cloud*pix_area_highres_pc

            # CHECK plot
            # R_max = highres_axis_max[i]
            # plt.close('all')
            # plot.simple_plot(figsize=(6, 6),xr=[-R_max,R_max],yr=[-R_max,R_max],aspect='equal',\
            #     x1=x_highres_axis,y1=x_highres_axis,col1=im_cloud,\
            #     contour_type1='mesh',xlab='x [pc]',ylab='y [pc]',\
            #     colorbar1=True,lab_colorbar1='L$_{\odot}$ per cell')
            # plt.show(block=False)


            # Normalize to total luminosity of this cloud:
            im_cloud            =   im_cloud*large_cloud_interpol[i]/np.sum(im_cloud)

            if verbose: print('check\n%.2e Msun from cloud image' % (np.sum(im_cloud)))

            # Identify low rew pixels that we will be filling up
            x_indices           =   [large_cloud_x_index[i] + _ for _ in np.arange(-max_pix_dif[i],max_pix_dif[i]+1)]
            y_indices           =   [large_cloud_y_index[i] + _ for _ in np.arange(-max_pix_dif[i],max_pix_dif[i]+1)]
            x_index_mesh, y_index_mesh = np.mgrid[slice(x_indices[0], x_indices[-1] + 1./1e6, 1),\
                slice(y_indices[0], y_indices[-1] + 1./1e6, 1)]
            x_index_array       =   x_index_mesh.reshape(len(x_indices)**2)
            y_index_array       =   y_index_mesh.reshape(len(y_indices)**2)
            # Center in x and y direction for highres cloud image:
            i_highres_center    =   float(npix_highres)/2.-0.5
            check               =   np.zeros([lenv,lenx,lenx])

            for x_i,y_i in zip(x_index_array,y_index_array):
                xdist_highres_from_cloud_center         =   (int(x_i) - large_cloud_x_index[i]) * npix_highres_def
                ydist_highres_from_cloud_center         =   (int(y_i) - large_cloud_y_index[i]) * npix_highres_def
                x_i_highres         =   int(i_highres_center + xdist_highres_from_cloud_center)# - (npix_highres-1)/2.)
                y_i_highres         =   int(i_highres_center + ydist_highres_from_cloud_center)# - (npix_highres-1)/2.)
                try:
                    # result[:,int(x_i),int(y_i)]             +=  vel_prof*np.sum(im_cloud[x_i_highres:x_i_highres+npix_highres_def+1, y_i_highres:y_i_highres+npix_highres_def+1])
                    check[:,int(x_i),int(y_i)]      +=  vel_prof*np.sum(im_cloud[x_i_highres:x_i_highres+npix_highres_def+1, y_i_highres:y_i_highres+npix_highres_def+1])
                    # overwrite_me                            +=  vel_prof*np.sum(im_cloud[x_i_highres:x_i_highres+npix_highres_def+1, y_i_highres:y_i_highres+npix_highres_def+1])
                except:
                    pixels_outside                  +=1
                    pass

            if np.sum(check) != 0:
                check              =   check*large_cloud_interpol[i]/np.sum(check)
                result             +=   check

            if verbose:
                if target == 'm':
                    print('%.2e Msun from drizzled image of cloud' % (np.sum(check)))
                    print('%.2e Msun from cloudy model result grid' % (large_cloud_interpol[i]))
                else:
                    print('%.2e Lsun from drizzled image of cloud' % (np.sum(check)))
                    if ISM_dc_phase == 'GMC':
                        print('%.2e Lsun from cloudy grid' % (models[target][large_cloud_model_index[i]]))
                    if ISM_dc_phase in ['DNG','DIG']:
                        if ISM_dc_phase == 'DNG':
                            nearest = models[target][large_cloud_model_index[i]]*models['f_'+line+'_DNG'][large_cloud_model_index[i]]
                            print('%.2e Lsun from cloudy grid nearest model' % nearest)
                            print('%.2e Lsun from interpolation' % (large_cloud_interpol[i]))
                        if ISM_dc_phase == 'DIG':
                            print('%.2e Lsun from cloudy grid nearest model' % (models[target][large_cloud_model_index[i]]*(1-models['f_'+line+'_DNG'][large_cloud_model_index[i]])))

        i                   +=  1

    print('%s highres cloud pixels went outside galaxy image' % (pixels_outside))

    return(result)

def load_clouds(dataframe,target,ISM_dc_phase):
    '''
    Purpose
    ---------
    Load clouds from saved galaxy files, and convert to a similar format.

    '''

    clouds1             =   dataframe.copy()
    print('Total number of clouds loaded: %s' % len(clouds1))

    if ISM_dc_phase in ['DNG','DIG']:
        if target == 'Z': clouds1[target]     =   dataframe['Z'].values
        if target != 'Z':
            if target in ['L_' + l for l in lines]:
                clouds1[target]     =   dataframe[target+'_'+ISM_dc_phase].values
            else:
                clouds1[target]     =   dataframe[target+'_'+ISM_dc_phase].values

    if ISM_dc_phase == 'GMC':
        clouds1['R']        =   dataframe['Rgmc'].values

    # TEST
    # clouds1             =   clouds1[0:30000]


    clouds1['x']        =   dataframe['x'] # kpc
    clouds1['y']        =   dataframe['y'] # kpc
    clouds1['z']        =   dataframe['z'] # kpc

    # TEST
    # clouds1['vx']         =   clouds1['vx']*0.+100.
    # clouds1['vx'][clouds['x'] < 0] =  -100.
    # clouds1['vy']         =   clouds1['vy']*0.
    # clouds1['vz']         =   clouds1['vz']*0.

    # Rotate cloud positions and velocities around y axis
    inc_rad             =   2.*np.pi*float(inc_dc)/360.
    coord               =   np.array([clouds1['x'],clouds1['y'],clouds1['z']])
    coord_rot           =   np.dot(rotmatrix(inc_rad,axis='y'),coord)
    clouds1['x'] = coord_rot[0]
    clouds1['y'] = coord_rot[1]
    clouds1['z'] = coord_rot[2]
    vel                 =   np.array([clouds1['vx'],clouds1['vy'],clouds1['vz']])
    vel_rot             =   np.dot(rotmatrix(inc_rad,axis='y'),vel)
    clouds1['vx'] = vel_rot[0]; clouds1['vy'] = vel_rot[1]; clouds1['vz'] = vel_rot[2]
    clouds1['v_proj']   =   clouds1['vz'] # km/s
    # pdb.set_trace()

    # Cut out only what's inside image:
    radius_pc           =   np.sqrt(clouds1['x']**2 + clouds1['y']**2)
    clouds1             =   clouds1[radius_pc < x_max_pc/1000.]
    clouds1             =   clouds1.reset_index(drop=True)

    # TEST!!
    # clouds1           =   clouds1.iloc[0:2]
    # clouds1['x'][0]   =   0
    # clouds1['y'][0]   =   0

    return(clouds1)

def mk_cloud_vel_profile(v_proj,vel_disp_gas,fine_v_axis,v_axis,plotting=False):
    '''
    Purpose
    ---------
    Make the velocity profile for *one* cloud

    What this function does
    ---------
    Calculates the fraction of total flux [Jy] going into the different velocity bins

    Arguments
    ---------
    v_proj: projected line-of-sight velocity of the cloud

    vel_disp_gas: velocity dispersion of the cloud

    v_axis: larger velocity axis to project clouds onto
    '''

    if vel_disp_gas > 0:

        # Evaluate Gaussian on fine velocity axis
        Gaussian                =   1./np.sqrt(2*np.pi*vel_disp_gas**2) * np.exp( -(fine_v_axis-v_proj)**2 / (2*vel_disp_gas**2) )

        # Numerical integration over velocity axis bins
        v_res                   =   (v_axis[1]-v_axis[0])/2.

        vel_prof                =   abs(np.array([integrate.trapz(fine_v_axis[(fine_v_axis >= v-v_res) & (fine_v_axis < v+v_res)], Gaussian[(fine_v_axis >= v-v_res) & (fine_v_axis < v+v_res)]) for v in v_axis]))

        if plotting:
            plot.simple_plot(fig=2,
                fontsize=12,xlab='v [km/s]',ylab='F [proportional to Jy]',\
                x1=fine_v_axis,y1=Gaussian,col1='r',ls1='--',\
                x2=v_axis,y2=vel_prof,col2='b',ls2='-.')

            plt.show(block=False)

        # Normalize that profile
        vel_prof                =   vel_prof*1./np.sum(vel_prof)

        # If vel_disp_gas is very small, vel_prof sometimes comes out with only nans, use a kind of Dirac delta function:
        if np.isnan(np.max(vel_prof)):
            vel_prof                =   v_axis*0.
            vel_prof[find_nearest(v_axis,v_proj,find='index')] = 1.

    else:
        # If vel_disp_gas = 0 km/s, use a kind of Dirac delta function:
        vel_prof                =   v_axis*0.
        vel_prof[find_nearest(v_axis,v_proj,find='index')] = 1.

    return(vel_prof)

#===============================================================================
""" Cosmology """
#-------------------------------------------------------------------------------

def get_lum_dist(zred):
    '''
    Purpose
    ---------
    Calculate luminosity distance for a certain redshift

    returns D_L in Mpc

    '''

    from astropy.cosmology import FlatLambdaCDM
    cosmo               =   FlatLambdaCDM(H0=hubble*100., Om0=omega_m, Ob0=1-omega_m-omega_lambda)

    if len(zred) > 1:
        D_L                 =   cosmo.luminosity_distance(zred).value
        zred_0              =   zred[zred == 0]
        if len(zred_0) > 0:
            D_L[zred == 0]      =   3+27.*np.random.rand(len(zred_0)) # Mpc (see Herrera-Camus+16)

    if len(zred) == 1:
        D_L                 =   cosmo.luminosity_distance(zred).value

    # ( Andromeda is rougly 0.78 Mpc from us )

    return(D_L)

#===========================================================================
""" Some arithmetics """
#---------------------------------------------------------------------------

def rad(foo,labels):
    # Calculate distance from [0,0,0] in 3D for DataFrames!
    if len(labels)==3: return np.sqrt(foo[labels[0]]**2+foo[labels[1]]**2+foo[labels[2]]**2)
    if len(labels)==2: return np.sqrt(foo[labels[0]]**2+foo[labels[1]]**2)

def find_nearest(array,value,find='value'):
    idx = (np.abs(array-value)).argmin()
    if find == 'value': return array[idx]
    if find == 'index': return idx

def lin_reg_boot(x,y,n_boot=5000,plotting=False):
    print('Fit power law and get errors with bootstrapping!')

    n_boot                  =   5000
    nGal                    =   len(x)

    def calc_slope(i, x1, y1, nGal):

        random_i                        =   (np.random.rand(nGal)*nGal).astype(int)

        slope1,inter1,foo1,foo2,foo3    =   stats.linregress(x1[random_i],y1[random_i])
        return slope1,inter1

    slope,inter,x1,x2,x3    =   stats.linregress(x,y)
    boot_results            =   [[calc_slope(i,x1=x,y1=y,nGal=nGal)] for i in range(0,n_boot)]
    slope_boot              =   [boot_results[i][0][0] for i in range(0,n_boot)]
    inter_boot              =   [boot_results[i][0][1] for i in range(0,n_boot)]

    # if plotting:
    #     # Make plot of distribution!
    #     yr = [0,240]
    #     plot.simple_plot(fig=j+1,xlab = ' slope from bootstrapping %s times' % n_boot,ylab='Number',\
    #         xr=plot.axis_range(slope_boot,log=False),yr=yr,legloc=[0.05,0.8],\
    #         histo1='y',histo_real1=True,x1=slope_boot,bins1=n_boot/50.,\
    #         x2=[slope,slope],y2=yr,lw2=1,ls2='--',col2='blue',lab2='Slope from fit to models',\
    #         x3=[np.mean(slope_boot),np.mean(slope_boot)],y3=yr,lw3=1,ls3='--',col3='green',lab3='Mean slope from bootstrapping')
    #         # figname='plots/line_SFR_relations/bootstrap_results/'+line+'slope_boot.png',figtype='png')
    #     plt.show(block=False)

    print('Slope from fit to models: %s' % np.mean(slope))
    print('Bootstrap mean: %s' % np.mean(slope_boot))
    print('Bootstrap std dev: %s' % np.std(slope_boot))

    print('Intercept from fit to models: %s' % np.mean(inter))
    print('Bootstrap mean: %s' % np.mean(inter_boot))
    print('Bootstrap std dev: %s' % np.std(inter_boot))

    return(slope,inter,np.std(slope_boot),np.std(inter_boot))

def rotmatrix(angle,axis='x'):

    cos         =   np.cos(angle)
    sin         =   np.sin(angle)

    if axis == 'x':
        rotmatrix       =   np.array([[1,0,0],[0,cos,-sin],[0,sin,cos]])

    if axis == 'y':
        rotmatrix       =   np.array([[cos,0,sin],[0,1,0],[-sin,0,cos]])

    if axis == 'z':
        rotmatrix       =   np.array([[cos,-sin,0],[sin,cos,0],[0,0,1]])

    return rotmatrix

def annulus_area(radius,x0,y0,dr,dx):

    # get upper integration limit
    xf  =   lambda r: min([np.sqrt(r**2 - y0**2), x0+dx])
    rf  =   radius+dr

    # make lambda function for the area under the curve where the annulus intercepts a smal CC region
    f   =   lambda x, r: min([0.5*x*np.sqrt(r**2 - x**2) + np.arctan(x/np.sqrt(r**2-x**2))*r**2, y0+dx])

    # get the area where r = radius+dr
    A1  =   f(xf(rf),rf) - f(x0,rf)

    # get the area where r = radius
    A2  =   f(xf(radius),radius) - f(x0,radius)

    return abs(A1-A2)

def gauss(center,FWHM,x):
    """
    Return value(s) of Gaussian, given the

    .. note:: See https://en.wikipedia.org/wiki/Gaussian_function
    """

    std_dev     =   FWHM_to_stddev(FWHM)

    A           =   1/(std_dev*np.sqrt(2*np.pi))

    return A*np.exp(-4*np.log(2)*(x-center)**2/FWHM**2)

def FWHM_to_stddev(FWHM):

    return(FWHM/(2.*np.sqrt(2*np.log(2))))

#===============================================================================
""" Conversions """
#-------------------------------------------------------------------------------

def LsuntoJydv(Lsun,zred=7,d_L=69727,nu_rest=1900.5369):
    """ Converts luminosity (in Lsun) to velocity-integrated flux (in Jy*km/s)

    args
    ----
    Lsun: numpy array
    solar luminosity (Lsun)

    zred: scalar
    redshift z (num)

    d_L: scalar
    luminosity distance (Mpc)

    nu_rest: scalar
    rest frequency of fine emission line (GHz)

    returns
    Jy*km/s array
    ------
    """

    return Lsun * (1+zred)/(1.04e-3 * nu_rest * d_L**2)

def solLum2Jy(Lsunkms, zred, d_L, nu_rest):
    """ Converts solar luminosity/(km/s) to milli-jansky/(km/s)

    args
    ----
    Lsunkms: numpy array
    solar luminosity / vel bin ( Lsun/(km/s) )

    zred: scalar
    redshift z (num)

    d_L: scalar
    luminosity distance (Mpc)

    nu_rest: scalar
    rest frequency of fine emission line (GHz)

    returns
    Jy/(km/s) array
    ------
    """

    return Lsunkms * (1+zred)/(1.04e-3 * nu_rest * d_L**2)

def Jykm_s_to_ergs_s(Jykms, line, zred, lum_dist):
    """Converts Jy*km/s to ergs/s
    """

    L_ergs              =   Jykm_s_to_L_sun(Jykms, line, zred, lum_dist)*Lsun*1e7 # ergs/s/kpc^2

    return(L_ergs)

def Jykm_s_to_L_sun(Jykms, line, zred, lum_dist):
    """Converts Jy*km/s to solar luminosity

    Parameters
    ----------
    Jykms : scalar
        total in Jy*km/s (10^-26*W/Hz/m^2*km/s)

    line : str
        Line ID

    zred: scalar
        redshift

    lum_dist: scalar
        luminosity distance (Mpc)

    """

    f_line          =   params['f_' + line]

    # Solomon+97 eq. 1:
    return 1.04e-3 * Jykms * (f_line/(1+zred)) * lum_dist**2

def disp2FWHM(sig):
    return 2*np.sqrt(2*np.log(2)) * sig

def W_m2_to_Jykm_s(line,zred,I):
    """Converts flux in W/m^2 to velocity integrated flux [Jy km/s]
    See: https://github.com/aconley/ALMAzsearch/blob/master/ALMAzsearch/radio_units.py

    Parameters
    ----------
    line : str
        line ID
    zred: scalar
        redshift

    """

    f_line          =   params['f_' + line]*1e9
    return I * clight/1e3 / (1e-26*f_line/(1+zred))

def Jykm_s_to_W_m2(line,zred,I):
    """Converts velocity integrated flux [Jy km/s] to W/m^2
    See: https://github.com/aconley/ALMAzsearch/blob/master/ALMAzsearch/radio_units.py

    Parameters
    ----------
    line : str
        line ID
    zred: scalar
        redshift
    """

    f_line          =   params['f_' + line]*1e9
    return I * 1e-26 * f_line/(1+zred) / (clight/1e3)

def arcsec2_to_sr(arcsec2):
    """Converts area on sky in arcsec^2 to steradians
    See: http://cosmos.phy.tufts.edu/cosmicfrontier/quants.html
    """

    return(arcsec2/4.25e10)

#===============================================================================
""" Other functions """
#-------------------------------------------------------------------------------

def diff_percent(x1,x2):
    '''
    Purpose
    -------
    Return difference in percent relative to x1: (x1-x2)/x1


    '''

    diff            =   (x1-x2)/x1*100.

    return(diff)

def fullprint(*args, **kwargs):
  from pprint import pprint
  import numpy
  opt = numpy.get_printoptions()
  numpy.set_printoptions(threshold='nan')
  pprint(*args, **kwargs)
  numpy.set_printoptions(**opt)

def line_name(line,latex=False):
    '''
    Purpose

    Get name for a line in pretty writing
    '''

    line_dictionary = {\
        'CII':'[CII]',\
        'OI':'[OI]',\
        'OIII':'[OIII]',\
        'NII_122':'[NII]122',\
        'NII_205':'[NII]205',\
        'CI_609':'CI(1-0)609',\
        'CI_369':'CI(2-1)369',\
        'CO32':'CO(3-2)',\
        'CO21':'CO(2-1)',\
        'CO10':'CO(1-0)'}

    return(line_dictionary[line])

def directory_checker(dirname):
    """ if a directory doesn't exist, then creates it """
    dirname =   str(dirname)
    if not os.path.exists(dirname):
        print("creating directory: %s" % dirname)
        try:
            os.mkdir(dirname)
        except:
            os.stat(dirname)

def directory_path_checker(pathway):
    """ checks that all the directories in a pathway exist; if they don't exist,
    then they are created."""

    # create and initialize list of indexes
    indexes =   []
    indexes.append( pathway.find('/') )
    index   =   indexes[0]

    # append index which marks the beginning of a new subdirectory
    while index >= 0:
        index   =   pathway.find('/',indexes[-1]+1)
        if index > 0: indexes.append(index)

    if indexes[0] == 0: indexes = indexes[1:]
    # run directory_checker for each directory in pathway
    for index in indexes:   directory_checker( pathway[:index] )

def check_version(module,version_required):

    version         =   module.__version__

    for i,subversion in enumerate(version.split('.')):
        if int(subversion) < version_required[i]:
            print('\nActive version of module %s might cause problems...' % module.__name__)
            print('version detected: %s' % version)
            print('version required: %s.%s.%s' % (version_required[0],version_required[1],version_required[2]))
            break
        if i == len(version.split('.'))-1:
            print('\nNo version problems for %s module expected!' % module.__name__)
