# coding=utf-8
###     Module: analysis.py of SIGAME             	###

import pandas as pd
import numpy as np
import pdb as pdb
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
import scipy as scipy
import classes as cl
from scipy.interpolate import interp1d
import scipy.stats as stats
import os.path
import pickle
import plot as plot
import aux as aux
import re as re
import time as time

params                      =   np.load('sigame/temp/params.npy').item()
for key,val in params.items():
    exec(key + '=val')


#===============================================================================
""" Global line luminosities """
#-------------------------------------------------------------------------------

def ISM_line_contributions(line='CII'):
    '''
    Purpose
    ---------
    Splits line luminosity into contributions from different ISM phases

    Arguments
    ---------
    line: which line to plot - str
    default = 'CII'
    '''

    print('')
    plt.close('all')        # close all windows

    # Load model results
    models      =   aux.find_model_results()
    for key in models.keys():
        exec(key + '=models[key].values')
    nGal                =   len(models)

    # SFR and Zsfr of models (chosen for x-axis here)
    SFR_sort            =   np.sort(SFR)
    Zsfr_sort           =   np.sort(Zsfr)
    x1,x2               =   SFR,Zsfr

    # Line luminosity and mass of ISM phases in models
    L_GMC,L_DNG,L_DIG   =   models['L_'+line+'_GMC'].values,models['L_'+line+'_DNG'].values,models['L_'+line+'_DIG'].values
    L_tot               =   L_GMC+L_DNG+L_DIG
    f_L                 =   np.array([L_GMC/L_tot,L_DNG/L_tot,L_DIG/L_tot])*100.
    M_tot               =   M_GMC+M_DNG+M_DIG
    f_M                 =   np.array([M_GMC/M_tot,M_DNG/M_tot,M_DIG/M_tot])*100.

    # Make plot (takes phases in the order GMCs, DNG, DIG)
    xr1                 =   np.array([2,24])
    yr1                 =   np.array([1,100])
    xr2                 =   np.array([0.1,0.5])
    fig                 =   plot.comp_ISM_phases(x1=x1,x2=x2,y1=f_L,xr1=xr1,yr1=yr1,xr2=xr2,ylab='Fraction of total L$_{[\mathrm{CII}]}$ [%]')

    # Add ISM phases for MW
    L_CII_MW_phases     =   np.array([0.55,0.25,0.2])*100.          # [ionized gas, HI, H2+PDRs] Pineda+14
    j                   =   0
    colors              =   ['r','orange','b']
    xrs                 =   [xr1,xr2]
    for L,color in zip(L_CII_MW_phases,colors):
        for i in [1,2]:
            ax1                 =   fig.add_subplot(3,2,j+i)
            ax1.plot(xrs[i-1],[L,L],ls='--',color=color,lw=1.5,label='Ionized gas in MW$_{}$')
            ax1.text(max(xrs[i-1])*0.95,L+5, 'MW', size=8, zorder=1, color=color)
        j                   +=  2

    plt.savefig('plots/line_emission/ISM_contributions/['+line+']_lum_fractions_'+z1+ext_DENSE+ext_DIFFUSE+'.pdf', format='pdf') # .eps for paper!

def ISM_line_efficiency(line='CII'):
    '''
    Purpose
    ---------
    Plots the line efficiency (line luminosity / gas mass) in different ISM phases

    Arguments
    ---------
    line: which line to plot - str
    default = 'CII'
    '''

    plt.close('all')        # close all windows

    # Load model results
    models      =   aux.find_model_results()
    for key in models.keys():
        exec(key + '=models[key].values')
    nGal                =   len(models)

    # SFR and Zsfr of models (chosen for x-axis here)
    SFR                 =   models['SFR'].values
    SFR_sort            =   np.sort(SFR)
    Zsfr                =   models['Zsfr'].values
    Zsfr_sort           =   np.sort(Zsfr)
    x1,x2               =   SFR,Zsfr

    # Line luminosity and mass of ISM phases in models
    L_GMC,L_DNG,L_DIG   =   models['L_'+line+'_GMC'].values,models['L_'+line+'_DNG'].values,models['L_'+line+'_DIG'].values
    L_tot               =   L_GMC+L_DNG+L_DIG
    f_L                 =   np.array([L_GMC/L_tot,L_DNG/L_tot,L_DIG/L_tot])*100.
    M_GMC,M_DNG,M_DIG   =   models['M_GMC'].values,models['M_DNG'].values,models['M_DIG'].values
    f_LM                =   np.array([L_GMC/M_GMC,L_DNG/M_DNG,L_DIG/M_DIG])

    # Make plot
    xr1                 =   np.array([2,24])
    yr1                 =   np.array([0.0004,0.015])
    xr2                 =   np.array([0.1,0.5])
    fig                 =   plot.comp_ISM_phases(x1=x1,x2=x2,y1=f_LM,xr1=xr1,yr1=yr1,xr2=xr2,ylab='L$_{\mathrm{[CII]}}$ / gas mass [L$_{\odot}/$M$_{\odot}$]')

    # Add ISM phases for MW
    # ax1                 =   plt.gca()
    L_CII_MW_phases     =   np.array([0.55,0.25,0.2])*100.          # [ionized gas, HI, H2+PDRs] Pineda+14
    j                   =   0
    colors              =   ['r','orange','b']
    xrs                 =   [xr1,xr2]
    for L,color in zip(L_CII_MW_phases,colors):
        for i in [1,2]:
            ax1                 =   fig.add_subplot(3,2,j+i)
            ax1.plot(xrs[i-1],[L,L],ls='--',color=color,lw=1.5,label='Ionized gas in MW$_{}$')
            ax1.text(max(xrs[i-1])*0.95,L+5, 'MW', size=8, zorder=1, color=color)
        j                   +=  2

    plt.savefig('plots/line_emission/line_efficiency/CII_ISM_eff_'+z1+ext_DENSE+ext_DIFFUSE+'.pdf', format='pdf') # .eps for paper!

    print('GMCs has ' + str(np.mean(f_LM[0]/f_LM[2])) + ' times higher [CII] efficiency than DIG (on average)')
    print('GMCs has ' + str(np.mean(f_LM[0]/f_LM[1])) + ' times higher [CII] efficiency than DNG (on average)')
    print('DNG has ' + str(np.mean(f_LM[1]/f_LM[2])) + ' times higher [CII] efficiency than DIG (on average)')

    print('\n rough comparison with z=2 results')

    print('Mean of GMC gas : %s ' % np.mean(f_LM[0]))
    print('Mean of DNG gas : %s ' % np.mean(f_LM[1]))
    print('Mean of DIG gas : %s ' % np.mean(f_LM[2]))

    L_CII_z2_H2         =   np.array([15,100,40,300,300,800,1000])*1e6
    L_CII_z2_PDR        =   np.array([130,20,100,80,100,90,110])*1e6
    L_CII_z2_HII        =   np.array([0.6,3,0.4,0.9,2,2,1.5])*1e6
    M_z2_H2             =   np.array([4,10,10,20,30,200,200])*1e7
    M_z2_PDR            =   np.array([4,7,8,15,20,25,30])*1e8
    M_z2_HII            =   np.array([3,5,10,11,12,20,22])*1e9

    print('Mean of H2 gas : %s ' % np.mean(L_CII_z2_H2/M_z2_H2))
    print('Mean of PDR gas : %s ' % np.mean(L_CII_z2_PDR/M_z2_PDR))
    print('Mean of HII gas : %s ' % np.mean(L_CII_z2_HII/M_z2_HII))

#===============================================================================
""" Other """
#-------------------------------------------------------------------------------

def ISM_mass_contributions():
    '''
    Purpose
    ---------
    Plots the mass contribution from different ISM phases

    '''

    plt.close('all')        # close all windows

    # Load model results
    models                      =   aux.find_model_results()
    for key in models.keys():
        exec(key + '=models[key].values')
    nGal                =   len(models)

    # SFR and Zsfr of models (chosen for x-axis here)
    SFR_sort            =   np.sort(SFR)
    Zsfr_sort           =   np.sort(Zsfr)
    x1,x2               =   SFR,Zsfr

    # Line luminosity and mass of ISM phases in models
    M_tot               =   M_GMC+M_DNG+M_DIG
    f_M                 =   np.array([M_GMC/M_tot,M_DNG/M_tot,M_DIG/M_tot])*100.

    # Make plot
    xr1                 =   np.array([2,24])
    yr1                 =   np.array([1,130])
    xr2                 =   np.array([0.1,0.5])
    yr2                 =   np.array([1,130])
    fig                 =   plot.comp_ISM_phases(x1=x1,x2=x2,xr2=xr2,y1=f_M,xr1=xr1,yr1=yr1,ylab='Fraction of total ISM mass [%]')

    # Add ISM phases for MW
    M_MW_phases         =   np.array([0.17,0.60,0.23])*100.          # [ionized gas, HI, H2+PDRs] Pineda+14
    j                   =   0
    colors              =   ['r','orange','b']
    xrs                 =   [xr1,xr2]
    for L,color in zip(M_MW_phases,colors):
        for i in [1,2]:
            ax1                 =   fig.add_subplot(3,2,j+i)
            ax1.plot(xrs[i-1],[L,L],ls='--',color=color,lw=1.5,label='Ionized gas in MW$_{}$')
            ax1.text(max(xrs[i-1])*0.95,L+5, 'MW', size=8, zorder=1, color=color)
        j                   +=  2


    plt.savefig('plots/mass_fractions/ISM_phases/CII_ISM_mass_'+z1+ext_DENSE+ext_DIFFUSE+'.pdf', format='pdf') # .eps for paper!



