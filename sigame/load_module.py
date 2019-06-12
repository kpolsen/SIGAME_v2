###     Module: load_module.py of SIGAME            ###
###     - loads SPH data                            ###
###     - calculates additional gas properties      ###

import re # to replace parts of string
import numpy as np
import itertools
import linecache as lc
import pdb as pdb
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import yt as yt
from yt.units import parsec, Msun, eV
import caesar
from caesar import utils
import h5py
from yt.units import Mpc,kpc
from yt.data_objects.particle_filters import add_particle_filter
import scipy as scipy
import pandas as pd
from scipy.integrate import simps
from aux import *
import os
import cPickle as cPickle
from statsmodels.nonparametric.smoothers_lowess import lowess
import math
import plot as plot
import aux as aux

params                      =   np.load('sigame/temp/params.npy', allow_pickle=True).item()
for key,val in params.items():
    exec(key + '=val')

def extract_galaxy(verbose=True):

    params                      =   np.load('sigame/temp/params.npy', allow_pickle=True).item()
    for key,val in params.items():
        exec(key + '=val')

    # Read galaxy from simulation snapshot file and/or center it!
    plt.close('all')        # close all windows

    print('\n ** Extract data for galaxy and save as DataFrame **')
    if simtype == 'mufasa':             galnames_unsorted,zreds_unsorted = load_mufasa(snaps,halos)
    if simtype == 'test':               galnames_unsorted,zreds_unsorted = create_test_galaxy()
    models              =   {'galnames_unsorted':galnames_unsorted,'zreds_unsorted':zreds_unsorted}
    cPickle.dump(models,open(d_temp+'global_results/galnames_'+simtype+'_'+z1,'wb'))
    print('Number of galaxies in entire sample: '+str(len(galnames_unsorted)))

    return galnames_unsorted,zreds_unsorted

def create_test_galaxy(galname='G1_test',zred=6,R_gal=5):
    """
    Creates a test galaxy with fixed, simplified characteristics.

    -------------------------
    Attributes: name, type, definition
    -------------------------
    index           int     galaxy index from redshift sample of galaxies
    name            str     galaxy name
    radius          float   galaxy radius
    zred            float   galaxy redshift
    lum_dist        float   luminosity distance at redshift
    N_radial_bins   int     number of bins in radius for plots

    -------------------------
    Optional keyword argument(s) and default value(s) to __init__(**kwargs)
    -------------------------
    index           0
    N_radial_bins   30


    """

    plt.close('all')

    # Default gas parameters
    Ngas                =   3000
    x,y,z,vx,vy,vz      =   get_rotating_disk(Ngas,R_gal,400)
    SFR                 =   np.zeros(Ngas)+1e-3
    Z                   =   np.zeros(Ngas)+1.
    nH                  =   np.zeros(Ngas)+1e-5
    Tk                  =   np.zeros(Ngas)+1e3
    h                   =   np.zeros(Ngas)+10**(-0.5)
    np.random.seed(Ngas+1)
    f_H21               =   np.random.rand(Ngas)*1.
    m                   =   np.zeros(Ngas)+10**(5.45)
    # Solar abundances used by Gizmo (email from RD), must be mass-fractions?:
    solar               =   [0.28,3.26e-3,1.32e-3,8.65e-3,2.22e-3,9.31e-4,1.08e-3,6.44e-4,1.01e-4,1.73e-3]
    a_He,a_C,a_N,a_O,a_Ne,a_Mg,a_Si,a_S,a_Ca,a_Fe   = [solar[i]*Z for i in range(len(solar))]
    gas_params          =   {'x':x,'y':y,'z':z,'vx':vx,'vy':vy,'vz':vz,'SFR':SFR,'Z':Z,'nH':nH,'Tk':Tk,'h':h,'f_H21':f_H21,\
                            'm':m,'a_He':a_He,'a_C':a_C,'a_N':a_N,'a_O':a_O,'a_Ne':a_Ne,'a_Mg':a_Mg,'a_Si':a_Si,'a_S':a_S,'a_Ca':a_Ca,'a_Fe':a_Fe}
    simgas              =   pd.DataFrame()
    for key,val in gas_params.items():
        simgas[key]         =   val
    simgas.to_pickle(d_temp+'sim/z'+'{:.2f}'.format(float(zred))+'_'+galname+'_sim0.gas')

    # Default stellar parameters
    Nstars              =   1000
    x,y,z,vx,vy,vz      =   get_rotating_disk(Nstars,R_gal,400)
    SFR                 =   np.zeros(Nstars)+10**(5.45)
    Z                   =   np.zeros(Nstars)+0.1
    m                   =   np.zeros(Nstars)+10**(6)
    age                 =   np.zeros(Nstars)+50.
    star_params         =   {'x':x,'y':y,'z':z,'vx':vx,'vy':vy,'vz':vz,'SFR':SFR,'Z':Z,'m':m,'age':age}
    simstar             =   pd.DataFrame()
    for key,val in star_params.items():
        simstar[key]         =   val
    simstar.to_pickle(d_temp+'sim/z'+'{:.2f}'.format(float(zred))+'_'+galname+'_sim0.star')

    # Default dark matter parameters
    Ndm                 =   30000
    x,y,z,vx,vy,vz      =   get_rotating_disk(Ndm,R_gal,400)
    m                   =   np.zeros(Ndm)+10**(5.45)
    dm_params           =   {'x':x,'y':y,'z':z,'vx':vx,'vy':vy,'vz':vz,'m':m}
    simdm               =   pd.DataFrame()
    for key,val in dm_params.items():
        simdm[key]         =   val
    simdm.to_pickle(d_temp+'sim/z'+'{:.2f}'.format(float(zred))+'_'+galname+'_sim0.dm')

    galnames_unsorted   =   [galname]
    zreds_unsorted      =   [zred]

    return galnames_unsorted,zreds_unsorted

def get_rotating_disk(N,R_gal,max,plotting=False):
    np.random.seed(N)
    ra                  =   np.random.rand(N)*R_gal  # draw random numbers between 0 and 1
    np.random.seed(N+1)
    phi                 =   np.random.rand(N)*360.  # draw random numbers between 0 and 1
    x                   =   ra*np.cos(phi)
    y                   =   ra*np.sin(phi)
    z                   =   np.zeros(N)
    radius              =   np.sqrt(x**2+y**2+z**2)
    speed               =   radius*max/np.max(radius)
    coord               =   np.array([x,y,z])
    coord_rot           =   np.dot(aux.rotmatrix(np.pi/2.,axis='z'),coord)
    length              =   np.sqrt(coord_rot[0,:]**2+coord_rot[1,:]**2+coord_rot[2,:]**2)
    vx                  =   coord_rot[0,:]*1./length*speed
    vy                  =   coord_rot[1,:]*1./length*speed
    vz                  =   coord_rot[2,:]*1./length*speed

    if plotting:

        # position check plot
        plot.simple_plot(x1=x,y1=y,ma1='x',fill1='n')
        plt.show(block=False)

        # velocity check plot
        plot.simple_plot()
        ax1                 =   plt.gca()
        ax1.quiver(x, y, vx, vy, units='width')
        ax1.set_xlabel('x [kpc]')
        ax1.set_ylabel('y [kpc]')
        plt.show(block=False)

    return x,y,z,vx,vy,vz

def load_mufasa(snaps,halos,member_search=True,verbose=True):
    print('\n * Simulation data is in GIZMO-MUFASA format *')

    # Level of written output from yt
    yt.funcs.mylog.setLevel(0)
    if verbose: yt.funcs.mylog.setLevel(1)

    snapshots           =   pd.read_table(params['d_t']+'snapshots.txt',names=['snaps','zs','times','D_L'],skiprows=1,sep='\t',engine='python')
    zs_table            =   snapshots['zs'].values
    snaps_table         =   snapshots['snaps'].values

    # Save final galaxy names here:
    galnames_selected   =   []
    R_gals              =   np.array([])
    zreds_selected      =   np.array([])

    print('Looking in these snapshots: ')
    print(snaps)

    Ngalaxies           =   0
    Ngalaxies_max       =   30

    snap_name   =   'snapshot_'

    # Extract all possible SFRs
    galnames            =   pd.DataFrame({'halo':[],'snap':[],'GAL':[],'SFR':[]})
    i                   =   0
    for halo in halos:

        print('Now looking at file with extension _h'+str(int(halo)))

        snap_dir    =   'sim/'+z1+'/mufasa/snapshots/h'+str(int(halo))+''

        run_caesar  =   True
        for snap in snaps:
            if os.path.isfile(snap_dir+'/Groups/caesar_00'+str(int(snap))+'_z'+'{:.3f}'.format(float(zs_table[snaps_table == snap][0]))+'.hdf5'):
                run_caesar  =   False
            if os.path.isfile(snap_dir+'/Groups/caesar_0'+str(int(snap))+'_z'+'{:.3f}'.format(float(zs_table[snaps_table == snap][0]))+'.hdf5'):
                run_caesar  =   False
        if run_caesar:
            print('Running caesar drive on *all* snapshots for this halo file')
            caesar.drive(snap_dir, snap_name, snaps, skipran=True, progen=True) # will only run caesar if member file cannot be found for one or more of snap_nums
        else:
            print('Already ran caesar on all snapshots for this halo file')

        for snap in reversed(snaps):
            zred                            =   '{:.3f}'.format(float(zs_table[snaps_table == snap][0]))
            # if snap < max(snaps): haloID = progen_ID # use progenitor from snapshot at higher z
            print('\nSnapshot # %s' % int(snap))
            print('from '+snap_dir+'/'+snap_name+'0'+str(int(snap))+'.hdf5')
            if snap < 100: ds              =   yt.load(snap_dir+'/'+snap_name+'0'+str(int(snap))+'.hdf5',over_refine_factor = 1) # raw snapshot
            if snap > 100: ds              =   yt.load(snap_dir+'/'+snap_name+str(int(snap))+'.hdf5',over_refine_factor = 1) # raw snapshot
            # Extract data within this radius from location of galaxy:
            dd                             =   ds.all_data()
            print('Simulation type: '+ds.dataset_type)

            if snap < 100: obj             =   caesar.load(snap_dir+'/Groups/caesar_00'+str(int(snap))+'_z'+zred+'.hdf5') # caesar member file
            if snap > 100: obj             =   caesar.load(snap_dir+'/Groups/caesar_0'+str(int(snap))+'_z'+zred+'.hdf5') # caesar member file
            print 'Total number of galaxies found: '+str(obj.ngalaxies)
            Ngal            =   obj.ngalaxies
            print('Info for this snapshot:')
            for key in ds.parameters.keys():
                print('%s = %s' % (key,ds.parameters[key]))
            omega_baryon    =   obj.simulation.omega_baryon
            omega_matter    =   obj.simulation.omega_matter
            hubble_constant =   obj.simulation.hubble_constant
            print('Omega baryon: %s' % omega_baryon)
            print('Omega matter: %s' % omega_matter)
            print('Hubble constant: %s' % hubble_constant)
            print('XH: %s' % obj.simulation.XH)
            # Optional: plot position of halos in this object
            # obj.yt_dataset  =   ds
            # obj.vtk_vis(annotate_galaxies=True,galaxy_only=True,draw_spheres='virial')

            # TEST!! Save data for entire halo (communication with Caitlin Doughty)
            # rr              =   halo_now.radius
            # pos             =   halo_now.pos
            # sphere          =   ds.sphere(pos,rr)
            # gas_f_HI,gas_f_H2 = hydrogen_mass_calc(obj,sphere)
            # gas_densities   =   sphere['PartType0','Density'].in_cgs()
            # gas_f_HI        =   np.ones(len(gas_densities))
            # gas_f_H2        =   np.ones(len(gas_densities))
            # simgas          =   pd.DataFrame({'nH':gas_densities,'f_H2':gas_f_H2})

            for GAL in range(200):

                try:
                    add_this                    =   pd.DataFrame({'halo':[int(halo)],'snap':[snap],'GAL':[GAL],'SFR':[float(obj.galaxies[GAL].sfr.d)]})
                    galnames                    =   galnames.append(add_this,ignore_index=True)
                    galnames[['halo','snap','GAL']] = galnames[['halo','snap','GAL']].astype(int)
                    i           +=   1
                except:
                    break

    # Select the Ngalaxies_max galaxies with highest SFRs
    galnames            =   galnames.sort_values(['SFR'],ascending=False).reset_index(drop=True)
    galnames            =   galnames[:Ngalaxies_max]


    print('Save %s most star forming galaxies!' % Ngalaxies_max)

    SFRg,SFRh           =   np.zeros(Ngalaxies_max),np.zeros(Ngalaxies_max)
    i                   =   0
    for halo,snap,GAL,sfr in zip(galnames['halo'].values,galnames['snap'].values,galnames['GAL'].values,galnames['SFR'].values):

        halo,snap,GAL,sfr               =   0,48,0,14.84
        zred                            =   '{:.3f}'.format(float(zs_table[snaps_table == snap][0]))

        print('\nNow looking at galaxy # %s with parent halo ID %s in snapshot %s at z = %s' % (int(GAL),int(halo),int(snap),zred))

        print('Creating galaxy name:')
        galname             =   'h'+str(int(halo))+'_s'+str(int(snap))+'_G'+str(int(GAL))
        print(galname)

        snap_dir    =   'sim/'+z1+'/mufasa/snapshots/h'+str(int(halo))+''
        if snap < 100: ds              =   yt.load(snap_dir+'/'+snap_name+'0'+str(int(snap))+'.hdf5',over_refine_factor = 1) # raw snapshot
        if snap > 100: ds              =   yt.load(snap_dir+'/'+snap_name+str(int(snap))+'.hdf5',over_refine_factor = 1) # raw snapshot
        if snap < 100: obj             =   caesar.load(snap_dir+'/Groups/caesar_00'+str(int(snap))+'_z'+zred+'.hdf5') # caesar member file
        if snap > 100: obj             =   caesar.load(snap_dir+'/Groups/caesar_0'+str(int(snap))+'_z'+zred+'.hdf5') # caesar member file

        galaxy              =   obj.galaxies[GAL]

        # Check contamination
        # if snap < 100: contamination_check(snap_dir+'/Groups/caesar_00'+str(int(snap))+'_z'+zreds[str(int(snap))]+'.hdf5',\
        #     snap_dir+'/'+snap_name+'0'+str(int(snap))+'.hdf5',haloID)
        # if snap > 100: contamination_check(snap_dir+'/Groups/caesar_0'+str(int(snap))+'_z'+zreds[str(int(snap))]+'.hdf5',\
        #     snap_dir+'/'+snap_name+str(int(snap))+'.hdf5',haloID)

        # if snap > min(snaps):
        #     print('Snapshot is not the first: Find progenitor ID for this halo to use at higher redshift:')
        #     progen_ID           =   halo_now.progen_index
        #     print(progen_ID)

        # Get location and radius for each galaxy belonging to this haloID:
        loc                 =   galaxy.pos
        R_gal               =   galaxy.radius
        print('Cut out a sphere with radius %s ' % R_gal)
        sphere              =   ds.sphere(loc,R_gal)

        print('Extracting all gas particle properties...')

        gas_pos             =   sphere['PartType0','Coordinates'].in_units('kpc')
        print('%s SPH particles' % len(gas_pos))

        if len(gas_pos) == 0: break

        gas_pos             =   gas_pos-loc
        gas_pos             =   caesar.utils.rotator(gas_pos, galaxy.rotation_angles['ALPHA'], galaxy.rotation_angles['BETA'])
        gas_posx,gas_posy,gas_posz = gas_pos[:,0].d,gas_pos[:,1].d,gas_pos[:,2].d
        gas_densities       =   sphere['PartType0','Density'].in_cgs()
        gas_vel             =   sphere['PartType0','Velocities'].in_cgs()/1e5
        gas_f_H21           =   sphere['PartType0','FractionH2']
        gas_f_neu           =   sphere['PartType0','NeutralHydrogenAbundance']
        gas_vel             =   caesar.utils.rotator(gas_vel, galaxy.rotation_angles['ALPHA'], galaxy.rotation_angles['BETA'])
        gas_velx,gas_vely,gas_velz = gas_vel[:,0].d,gas_vel[:,1].d,gas_vel[:,2].d
        gas_m               =   sphere['PartType0','Masses'].in_units('Msun')               # Msun
        gas_x_e             =   sphere['PartType0','ElectronAbundance']                     # electrons per Hydrogen atom (max: 1.15)
        gas_f_ion           =   gas_x_e/max(gas_x_e)                                    # ionized gas mass fraction (because we don't trust NeutralHydrogenAbundance)
        gas_f_HI1           =   1-gas_f_ion
        gas_f_HI,gas_f_H2   =   hydrogen_mass_calc(obj,sphere)
        print('Molecular gas mass fraction is %s %% (KMT)' % (np.sum(gas_f_H2*gas_m)/np.sum(gas_m)*100.))
        print('Molecular gas mass fraction is %s %% (simulation)' % (np.sum(gas_f_H21*gas_m)/np.sum(gas_m)*100.))
        gas_Tk              =   sphere['PartType0','Temperature']                           # Tk
        gas_h               =   sphere['PartType0','SmoothingLength'].in_units('kpc')       # Tk
        gas_SFR             =   sphere['PartType0','StarFormationRate']                     # Msun/yr
        gas_Z               =   sphere['PartType0','Metallicity_00'].d/0.0134               # from RD
        gas_a_He            =   sphere['PartType0','Metallicity_01'].d
        gas_a_C             =   sphere['PartType0','Metallicity_02'].d
        gas_a_N             =   sphere['PartType0','Metallicity_03'].d
        gas_a_O             =   sphere['PartType0','Metallicity_04'].d
        gas_a_Ne            =   sphere['PartType0','Metallicity_05'].d
        gas_a_Mg            =   sphere['PartType0','Metallicity_06'].d
        gas_a_Si            =   sphere['PartType0','Metallicity_07'].d
        gas_a_S             =   sphere['PartType0','Metallicity_08'].d
        gas_a_Ca            =   sphere['PartType0','Metallicity_09'].d
        gas_a_Fe            =   sphere['PartType0','Metallicity_10'].d

        print('\nChecking molecular gas mass fraction:')
        print('%.3s %% using RT prescription' % (np.sum(gas_m*gas_f_H2)/np.sum(gas_m)*100.))
        print('%.3s %% using simulation\n' % (np.sum(gas_m*gas_f_H21)/np.sum(gas_m)*100.))

        print('Extracting all star particle properties...')
        star_pos_all        =   sphere['PartType4','Coordinates'].in_units('kpc')
        star_pos            =   star_pos_all-loc
        star_pos            =   caesar.utils.rotator(star_pos, galaxy.rotation_angles['ALPHA'], galaxy.rotation_angles['BETA'])
        star_posx,star_posy,star_posz = star_pos[:,0].d,star_pos[:,1].d,star_pos[:,2].d
        star_vel            =   sphere['PartType4','Velocities'].in_cgs()/1e5
        star_vel            =   caesar.utils.rotator(star_vel, galaxy.rotation_angles['ALPHA'], galaxy.rotation_angles['BETA'])
        star_velx,star_vely,star_velz = star_vel[:,0].d,star_vel[:,1].d,star_vel[:,2].d
        star_m              =   sphere['PartType4','Masses'].in_units('Msun')
        star_a_C            =   sphere['PartType4','Metallicity_02'].d
        star_a_O            =   sphere['PartType4','Metallicity_04'].d
        star_a_Si           =   sphere['PartType4','Metallicity_07'].d
        star_a_Fe           =   sphere['PartType4','Metallicity_10'].d
        star_Z              =   sphere['PartType4','Metallicity_00'].d/0.0134               # from RD
        current_time        =   ds.current_time.in_units('yr')/1e6 # Myr
        star_formation_a    =   sphere['PartType4','StellarFormationTime'].d               # in scale factors, do as with Illustris
        star_formation_z    =   1./star_formation_a-1
        # Code from yt project (yt.utilities.cosmology)
        star_formation_t    =   2.0/3.0/np.sqrt(1-omega_matter)*np.arcsinh(np.sqrt((1-omega_matter)/omega_matter)/ np.power(1+star_formation_z, 1.5))/(hubble_constant) # Mpc*s/(100*km)
        star_formation_t    =   star_formation_t*kpc2m/100./(1e6*365.25*86400) # Myr
        star_age            =   current_time.d-star_formation_t

        print('Extracting all DM particle properties...')
        dm_pos_all          =   sphere['PartType1','Coordinates'].in_units('kpc')
        dm_pos              =   dm_pos_all-loc
        dm_pos              =   caesar.utils.rotator(dm_pos, galaxy.rotation_angles['ALPHA'], galaxy.rotation_angles['BETA'])
        dm_posx,dm_posy,dm_posz = dm_pos[:,0].d,dm_pos[:,1].d,dm_pos[:,2].d
        dm_vel              =   sphere['PartType1','Velocities'].in_cgs()/1e5
        dm_vel              =   caesar.utils.rotator(dm_vel, galaxy.rotation_angles['ALPHA'], galaxy.rotation_angles['BETA'])
        dm_velx,dm_vely,dm_velz = dm_vel[:,0].d,dm_vel[:,1].d,dm_vel[:,2].d
        dm_m                =   sphere['PartType1','Masses'].in_units('Msun')


        # Put into dataframes:
        simgas              =   pd.DataFrame({'x':gas_posx,'y':gas_posy,'z':gas_posz,\
            'vx':gas_velx,'vy':gas_vely,'vz':gas_velz,\
            'SFR':gas_SFR,'Z':gas_Z,'nH':gas_densities,'Tk':gas_Tk,'h':gas_h,\
            'f_HI':gas_f_HI,'f_H2':gas_f_H2,'f_neu':gas_f_neu,'f_HI1':gas_f_HI1,'f_H21':gas_f_H21,'m':gas_m,\
            'a_He':gas_a_He,'a_C':gas_a_C,'a_N':gas_a_N,'a_O':gas_a_O,'a_Ne':gas_a_Ne,'a_Mg':gas_a_Mg,\
            'a_Si':gas_a_Si,'a_S':gas_a_S,'a_Ca':gas_a_Ca,'a_Fe':gas_a_Fe})
        simstar             =   pd.DataFrame({'x':star_posx,'y':star_posy,'z':star_posz,\
            'vx':star_velx,'vy':star_vely,'vz':star_velz,\
            'Z':star_Z,'m':star_m,'age':star_age})
        simdm               =   pd.DataFrame({'x':dm_posx,'y':dm_posy,'z':dm_posz,\
            'vx':dm_velx,'vy':dm_vely,'vz':dm_velz,'m':dm_m})

        # Center in position and velocity
        simgas,simstar,simdm =   center_cut_galaxy(simgas,simstar,simdm,plot=False)

        # Calculate approximate 100 Myr average of SFR for this galaxy
        m_star              =   simstar['m'].values
        age_star            =   simstar['age'].values
        SFR                 =   sum(m_star[age_star < 100])/100e6
        print('Estimated SFR for this galaxy (<30 kpc) averaged over past 100 Myr: '+str(SFR))
        print('SFR in simulation: %s' % sfr)
        print('SFR in simulation: %s' % galaxy.sfr)
        print('For parent halo: %s' % galaxy.halo.sfr)
        SFRg[i]             =   SFR
        SFRh[i]             =   galaxy.halo.sfr.d
        i                   +=  1
        print('Stellar mass: %s Msun' % np.sum(m_star))
        print('Dark matter mass: %s ' % np.sum(dm_m))

        pdb.set_trace()
        print('Save galaxy data as DataFrame in sigame/temp/sim/...')

        simgas.to_pickle('sigame/temp/sim/z'+'{:.2f}'.format(float(zred))+'_'+galname+'_sim0.gas')
        simstar.to_pickle('sigame/temp/sim/z'+'{:.2f}'.format(float(zred))+'_'+galname+'_sim0.star')
        simdm.to_pickle('sigame/temp/sim/z'+'{:.2f}'.format(float(zred))+'_'+galname+'_sim0.dm')

        galnames_selected.append(galname)
        zreds_selected          =   np.append(zreds_selected,float(zred))

    # Check contamination at first snapshot, then compare following progen index to that snapshot number:
    # print('Progren index:')
    # haloID              =   obj.galaxies[GAL].parent_halo_index
    # print(obj.halo[haloID].progen_index)
    # contamination_check(MEMBER,SNAPSHOT,0) # insert halo ID for this galaxy

    # halo_list           =   obj.galaxies[GAL].halo.glist
    # x_e_halo            =   dd['PartType0','ElectronAbundance'][halo_list].d
    # m_halo              =   dd['PartType0','Masses'][halo_list].in_units('Msun').d
    # halo_data           =   pd.DataFrame({'x_e':x_e_halo,'m':m_halo})
    # halo_data.to_pickle('sigame/temp/halo_x_e_z'+str(int(zred))+'_h'+str(int(halo))+'_s'+str(int(snap))+'_G'+str(GAL))

    pdb.set_trace()
    return galnames_selected,zreds_selected

def time_machine():
    print('\n * Look at history of one galaxy *')

    halo                =   5
    snaps               =   [77,78,79,80,81,82,83,84]
    GAL                 =   1
    haloID              =   0
    zreds               =   {'74':'2.600','77':'2.400','78':'2.350','79':'2.300','80':'2.250','81':'2.200','82':'2.150','83':'2.100','84':'2.050','85':'2.000',\
                            '48':'5.750','47':'5.875','46':'6.000','45':'6.125','44':'6.250','43':'6.375','42':'6.500','41':'6.625',\
                            '40':'6.750','39':'6.875','38':'7.000','37':'7.125','36':'7.250','35':'7.375','34':'7.500'} # redshifts associated with these snapshots
    snap_dir            =   'SPH/'+z1+'/mufasa/snapshots/h'+str(int(halo))+''
    snap_name           =   'snapshot_'

    M_stars             =   np.zeros(len(snaps))
    SFRs                =   np.zeros(len(snaps))
    times_for_snap      =   {'77':2.682,'78':2.741,'79':2.803,'80':2.866,'81':2.932,'82':3.001,'83':3.072,'84':3.146}
    i                   =   0

    # caesar.drive(snap_dir, snap_name, snaps, skipran=True, progen=True)
    for snap in reversed(snaps):
        zred = zreds[str(int(snap))]
        if snap < max(snaps): haloID = progen_halo # use progenitor from snapshot at higher z
        if snap < max(snaps): GAL = progen_GAL # use progenitor from snapshot at higher z
        print('\nSnapshot # %s' % int(snap))
        ds                      =   yt.load(snap_dir+'/'+snap_name+'0'+str(int(snap))+'.hdf5',over_refine_factor = 1) # raw snapshot
        dd                      =   ds.all_data()
        obj                     =   caesar.load(snap_dir+'/Groups/caesar_00'+str(int(snap))+'_z'+zred+'.hdf5') # caesar member file
        halo_now                =   obj.halos[haloID]
        if snap > min(snaps): progen_halo = halo_now.progen_index
        print('Progen halo ID: %s' % progen_halo)
        gal                     =   obj.galaxies[GAL]
        if snap > min(snaps): progen_GAL = gal.progen_index
        print('Progen gal ID: %s' % progen_GAL)
        R_gal                   =   gal.radius.in_units('kpc').d
        print('Radius is: '+str(R_gal)+' kpc')
        gal                     =   halo_now.galaxies[GAL]
        loc                     =   gal.pos
        R_gal                   =   gal.radius
        print('Cut out a sphere with radius %s ' % R_gal)
        sphere                  =   ds.sphere(loc,R_gal)
        print('Extracting all gas particle properties...')
        # gas_m                   =   sphere['PartType0','Masses'].in_units('Msun')               # Msun
        gas_SFR                 =   sphere['PartType0','StarFormationRate']
        SFRs[i]                 =   np.sum(gas_SFR)
        star_m                  =   sphere['PartType4','Masses'].in_units('Msun')
        M_stars[i]              =   np.sum(star_m)
        # Do a projection plot
        plt.close('all')        # close all windows
        new_box_size            =   ds.quan(2.*R_gal,'kpc')
        left_edge               =   gal.pos - new_box_size/2
        right_edge              =   gal.pos + new_box_size/2
        p                       =   yt.SlicePlot(ds, normal=[0,1,0], fields=('gas','density'), center=gal.pos, width=new_box_size)
        p.set_font_size(12)
        fig             =   plt.figure()
        grid            =   AxesGrid(fig, (0.075,0.075,0.85,0.85),
                            nrows_ncols = (1,1),
                            axes_pad = 0.15,
                            label_mode = "L",
                            aspect = True,
                            share_all = False,
                            cbar_location="right",
                            cbar_mode="single",
                            cbar_size="3%",
                            cbar_pad="0%")
        plot                    =   p.plots[('gas','density')]
        plot.figure             =   fig
        plot.axes               =   grid[0].axes
        plot.cax                =   grid.cbar_axes[0]
        grid[0].set_xlim([-R_gal,R_gal])#[-2.*R_gal,2.*R_gal])
        grid[0].set_ylim([-R_gal,R_gal])#[-2.*R_gal,2.*R_gal])
        # px.set_cmap('density','gnuplot2')
        p.set_xlabel('x [kpc]')
        p.set_ylabel('y [kpc]')
        p._setup_plots()
        galname                 =   'h'+str(int(halo))+'_s'+str(int(snap))+'_G'+str(int(GAL))
        plt.savefig('plots/sim/time_machine/'+galname+'_z'+zred+'.png')
        i                       +=  1
    history             =   {'M_stars':M_stars,'SFRs':SFRs,'times_for_snap':times_for_snap}
    np.save('sigame/temp/history/history_h'+str(int(halo))+'.npy',history)

def hydrogen_mass_calc(obj,dd):
    print('Calculate HI & H2 mass for each valid gas particle')

    redshift    = obj.simulation.redshift

    from .treecool_data import UVB
    uvb = UVB['FG11']

    # sim = obj.simulation

    if np.log10(redshift + 1.0) > uvb['logz'][len(uvb['logz'])-1]:
        gamma_HI = 0.0
    else:
        gamma_HI = np.interp(np.log10(redshift + 1.0),
                             uvb['logz'],uvb['gH0'])

    ## density thresholds in atoms/cm^3
    low_rho_thresh  =    0.       # atoms/cm^3
    rho_thresh      =    1000        # atoms/cm^3

    XH              =   obj.simulation.XH
    proton_mass     =   1.67262178e-24 # g
    FSHIELD         =   0.99

    # ## Leroy et al 2008, Fig17 (THINGS) Table 6 ##
    P0BLITZ         =   1.7e4
    ALPHA0BLITZ     =   0.8

    # ## Poppin+09 constants (equation 7)
    a               =   7.982e-11                  # cm^3/s
    b               =   0.7480
    T0              =   3.148                      # K
    T1              =   7.036e5                    # K

    sigHI           =   3.27e-18 * (1.0+redshift)**(-0.2)
    fbaryon         =   obj.simulation.omega_baryon / obj.simulation.omega_matter
    nHss_part       =   6.73e-3 * (sigHI/2.49e-18)**(-2./3.) * (fbaryon / 0.17)**(-1./3.)

    ## global lists
    halo_glist      =   np.array(obj.global_particle_lists.halo_glist,dtype=np.int32)
    galaxy_glist    =   np.array(obj.global_particle_lists.galaxy_glist,dtype=np.int32)

    print('extract gas properties')
    from caesar.property_manager import get_property, has_property
    # gpos            = obj.data_manager.pos[obj.data_manager.glist]
    # gmass           = obj.data_manager.mass[obj.data_manager.glist]
    # grhoH           = get_property(obj, 'rho', 'gas').in_cgs().d * XH / proton_mass
    # gtemp           = obj.data_manager.gT.to('K').d
    # gsfr            = obj.data_manager.gsfr.d
    # gnh             = get_property(obj, 'nh', 'gas').d
    gpos            =   dd['PartType0','Coordinates']
    gmass           =   dd['PartType0','Masses'].in_units('Msun')
    grhoH           =   dd['PartType0','Density'].in_cgs().d * XH / proton_mass
    gtemp           =   dd['PartType0','Temperature'].in_units('K').d
    gsfr            =   dd['PartType0','StarFormationRate'].d
    gnh             =   dd['PartType0','NeutralHydrogenAbundance'].d

    nhalos          =   len(obj.halos)
    ngalaxies       =   len(obj.galaxies)
    ngas            =   len(gmass)

    # H2_data_present = 0
    # if has_property(obj, 'gas', 'fH2'):
    #     gfH2  = get_property(obj, 'fH2', 'gas').d
    #     H2_data_present = 1
    # else:
    #     mylog.warning('Could not locate molecular fraction data. Estimating via Leroy+08')
    gfH2            =   dd['PartType0','FractionH2'].d
    H2_data_present     =   1


    f_HI            =   np.zeros(ngas)
    f_H2            =   np.zeros(ngas)

    print('and do the calculation!')
    part            =   0.1
    for i in range(0,ngas):

        ## skip if density is too low (continue will skip the rest of the code)
        if grhoH[i] < low_rho_thresh:
            continue

        fHI     =   gnh[i]
        fH2     =   0.0

        ## low density non-self shielded gas
        if grhoH[i] < rho_thresh:
            ### Popping+09 equations 3, 7, 4
            #xi       = fHI
            beta     = a / (np.sqrt(gtemp[i]/T0) *
                            (1.0 + np.sqrt(gtemp[i]/T0))**(1.0-b) *
                            (1.0 + np.sqrt(gtemp[i]/T1))**(1.0+b))   # cm^3/s
            #gamma_HI = (1.0-xi)*(1.0-xi) * grhoH[i] * beta / xi   # 1/s

            ## Rahmati+13 equations 2, 1
            nHss      = nHss_part * (gtemp[i] * 1.0e-4)**0.17 * (gamma_HI * 1.0e12)**(2./3.)
            fgamma_HI = 0.98 * (1.0 + (grhoH[i] / nHss)**(1.64))**(-2.28) + 0.02 * (1.0 + grhoH[i] / nHss)**(-0.84)

            ## Popping+09 equations 6, 5
            C = grhoH[i] * beta / (gamma_HI * fgamma_HI)
            fHI = (2.0 * C + 1.0 - np.sqrt((2.0*C+1.0)*(2.0*C+1.0) - 4.0 * C * C)) / (2.0*C)

        ## high density gas when no H2 data is present
        ## estimate H2 via Leroy+08
        elif not H2_data_present:
            cold_phase_massfrac = (1.0e8 - gtemp[i])/1.0e8
            Rmol = (grhoH[i] * gtemp[i] / P0BLITZ)**ALPHA0BLITZ
            fHI  = FSHIELD * cold_phase_massfrac / (1.0 + Rmol)
            fH2  = FSHIELD - fHI


        ## high density gas when H2 data is present
        else:
            fH2 = gfH2[i]
            fHI = 1.0 - fH2

            if fHI < 0.0:
                fHI = 0.0

        # HImass[i]    =  fHI * gmass[i]
        # H2mass[i]    =  fH2 * gmass[i]

        f_HI[i]      =  fHI
        f_H2[i]      =  fH2

        if i*1./ngas > part:
            print('%s %% done!' % (part*100.))
            part        +=   0.1

    return f_HI,f_H2

def load_gadget(GAL,snap,member_search=False,verbose=True):
    print('\n * Simulation data is in gadget format *')

    # query all available members
    MEMBER      =   d_sph+'gadget-3/members/members.'+str(int(snap))+'.hdf5'
    SNAPSHOT    =   d_sph+'gadget-3/snapshots/snapshot.'+str(int(snap))+'.hdf5'

    # Level of written output from yt
    yt.funcs.mylog.setLevel(0)
    if verbose: yt.funcs.mylog.setLevel(1)

    # load the raw dataset into yt
    ds          =   yt.load(SNAPSHOT,over_refine_factor = 1)        # Set over_refine_factor to a higher value if your want nicer images (will take time!)
    print('Simulation type: '+ds.dataset_type)

    # Make member search on snapshot?
    if member_search:
        i           =   0
        while os.path.isfile(d_sph+'gadget-3/snapshots/snapshot.'+str(int(i))+'.hdf5'):
            ds          =   yt.load(d_sph+'gadget-3/snapshots/snapshot.'+str(int(i))+'.hdf5',over_refine_factor = 1)        # Set over_refine_factor to a higher value if your want nicer images (will take time!)
            obj         =   caesar.CAESAR(ds)
            obj.member_search()
            obj.save(d_sph+'gadget-3/members/members.'+str(int(i))+'.hdf5')
            i           +=  1

    # load the current member file
    obj         =   caesar.load(MEMBER)

    print 'Number of galaxies found: '+str(obj.ngalaxies)

    # Print Ngal biggest galaxies (10 galaxies by default)
    Ngal        =   obj.ngalaxies
    if verbose: print(obj.galinfo(top=Ngal))

    pdb.set_trace()
    # Select gas/star particles in ONE galaxy
    gal                 =   obj.galaxies[GAL]
    galaxy_glist        =   gal.glist
    galaxy_slist        =   gal.slist

    # Rotate coordinates + velocities to be aligned with xy-plane
    dd                  =   ds.all_data()     # to get abundances and ages out...
    gas_pos             =   dd['PartType0','Coordinates'].in_units('kpc')
    star_pos            =   dd['PartType4','Coordinates'].in_units('kpc')
    # DM_pos              =   caesar.utils.rotator(DM_pos, gal.rotation_angles['ALPHA'], gal.rotation_angles['BETA'])
    # DM_vel              =   caesar.utils.rotator(DM_vel, gal.rotation_angles['ALPHA'], gal.rotation_angles['BETA'])

    R_gal               =   gal.radius.in_units('kpc').d
    if verbose: print('Radius from caesar: '+str(R_gal)+' kpc')

    # Cut out everything within R_gal
    if verbose: print('Loading all gas particle properties...')
    gas_pos             =   gas_pos-obj.galaxies[GAL].pos.in_units('kpc')
    gas_pos             =   caesar.utils.rotator(gas_pos, gal.rotation_angles['ALPHA'], gal.rotation_angles['BETA'])
    radius              =   np.sqrt(gas_pos[:,0].d**2.+gas_pos[:,1].d**2.+gas_pos[:,2].d**2.)
    gas_vel             =   dd['PartType0','Velocities'].in_cgs()/1e5
    gas_vel             =   caesar.utils.rotator(gas_vel, gal.rotation_angles['ALPHA'], gal.rotation_angles['BETA'])
    gas_velx,gas_vely,gas_velz            =   gas_vel[:,0].d[radius < R_gal],gas_vel[:,1].d[radius < R_gal],gas_vel[:,2].d[radius < R_gal]
    gas_densities       =   dd['PartType0','Density'].in_cgs()
    gas_densities       =   gas_densities[radius < R_gal]
    gas_m               =   dd['PartType0','Masses'].in_units('Msun')             # Msun
    gas_m               =   gas_m.d[radius < R_gal]
    gas_x_e             =   dd['PartType0','ElectronAbundance']                 # electrons per Hydrogen atom (max: 1.15)
    gas_x_e             =   gas_x_e.d[radius < R_gal]
    gas_f_ion           =   gas_x_e/max(gas_x_e)                                                # ionized gas mass fraction (because we don't trust NeutralHydrogenAbundance)
    gas_f_HI            =   1.-gas_x_e/max(gas_x_e)
    gas_Tk              =   dd['PartType0','Temperature']                         # Tk
    gas_Tk              =   gas_Tk.d[radius < R_gal]
    gas_h               =   dd['PartType0','SmoothingLength'].in_units('kpc')     # Tk
    gas_h               =   gas_h.d[radius < R_gal]
    gas_SFR             =   dd['PartType0','StarFormationRate']                   # Msun/yr
    gas_SFR             =   gas_SFR.d[radius < R_gal]
    gas_a_C             =   dd['PartType0','Metallicity_00'].d
    gas_a_O             =   dd['PartType0','Metallicity_01'].d
    gas_a_Si            =   dd['PartType0','Metallicity_02'].d
    gas_a_Fe            =   dd['PartType0','Metallicity_03'].d
    gas_Z               =   (gas_a_C+gas_a_O+gas_a_Si+gas_a_Fe) * 0.0189/0.0147 / 0.02          # from RT
    gas_Z               =   gas_Z[radius < R_gal]
    gas_posx,gas_posy,gas_posz            =   gas_pos[:,0].d[radius < R_gal],gas_pos[:,1].d[radius < R_gal],gas_pos[:,2].d[radius < R_gal]

    if verbose: print('Loading all star particle properties...')
    star_pos            =   star_pos-obj.galaxies[GAL].pos.in_units('kpc')
    radius              =   np.sqrt(star_pos[:,0].d**2.+star_pos[:,1].d**2.+star_pos[:,2].d**2.)
    star_pos            =   caesar.utils.rotator(star_pos, gal.rotation_angles['ALPHA'], gal.rotation_angles['BETA'])
    star_vel            =   dd['PartType4','Velocities'].in_cgs()/1e5
    star_vel            =   caesar.utils.rotator(star_vel, gal.rotation_angles['ALPHA'], gal.rotation_angles['BETA'])
    star_velx,star_vely,star_velz            =   star_vel[:,0].d[radius < R_gal],star_vel[:,1].d[radius < R_gal],star_vel[:,2].d[radius < R_gal]
    star_m              =   dd['PartType4','Masses'].in_units('Msun')
    star_m              =   star_m.d[radius < R_gal]
    star_a_C            =   dd['PartType4','Metallicity_00'].d
    star_a_O            =   dd['PartType4','Metallicity_01'].d
    star_a_Si           =   dd['PartType4','Metallicity_02'].d
    star_a_Fe           =   dd['PartType4','Metallicity_03'].d
    star_Z              =   (star_a_C+star_a_O+star_a_Si+star_a_Fe) * 0.0189/0.0147 / 0.02       # from RT
    star_Z              =   star_Z[radius < R_gal]
    current_time        =   ds.current_time.in_units('yr')
    star_age            =   (current_time.d-dd['PartType4','StellarFormationTime'].d)/1e6
    star_age            =   star_age[radius < R_gal]
    star_posx,star_posy,star_posz            =   star_pos[:,0].d[radius < R_gal],star_pos[:,1].d[radius < R_gal],star_pos[:,2].d[radius < R_gal]
    # Check contamination
    # gal.contamination_check()
    # pdb.set_trace()

    # if verbose: print('Loading all gas particle properties...')

    # dd                  =   ds.all_data()     # to get abundances and ages out...
    # gas_f_HI            =   dd['PartType0','NeutralHydrogenAbundance'][galaxy_glist].d        # fraction of Hydrogen atoms that are NOT ionized
    # gas_x_e             =   dd['PartType0','ElectronAbundance'][galaxy_glist].d                 # electrons per Hydrogen atom (max: 1.15)
    # gas_f_ion           =   gas_x_e/max(gas_x_e)                                                # ionized gas mass fraction (because we don't trust NeutralHydrogenAbundance)
    # gas_f_HI            =   1.-gas_x_e/max(gas_x_e)                                             # assuming that only H is ionized?
    # gas_f_HI            =   dd['PartType0','NeutralHydrogenAbundance'][galaxy_glist].d
    # gas_pos             =   dd['PartType0','Coordinates'].in_units('kpc')[galaxy_glist]         # kpc
    # gas_vel             =   dd['PartType0','Velocities'][galaxy_glist].in_cgs()/1e5             # km/s
    # gas_densities       =   dd['PartType0','Density'][galaxy_glist].in_cgs()                    # g/cm^-3
    # gas_m               =   dd['PartType0','Masses'][galaxy_glist].in_units('Msun')             # Msun
    # gas_Tk              =   dd['PartType0','Temperature'][galaxy_glist].d                       # Tk
    # gas_h               =   dd['PartType0','SmoothingLength'][galaxy_glist].in_units('kpc')     # Tk
    # gas_SFR             =   dd['PartType0','StarFormationRate'][galaxy_glist].d                 # Msun/yr
    # gas_a_C             =   dd['PartType0','Metallicity_00'][galaxy_glist].d
    # gas_a_O             =   dd['PartType0','Metallicity_01'][galaxy_glist].d
    # gas_a_Si            =   dd['PartType0','Metallicity_02'][galaxy_glist].d
    # gas_a_Fe            =   dd['PartType0','Metallicity_03'][galaxy_glist].d
    # gas_Z               =   (gas_a_C+gas_a_O+gas_a_Si+gas_a_Fe) * 0.0189/0.0147 / 0.02          # from RT

    # if verbose: print('Loading all star particle properties...')
    # position, velocity, mass, metallicity
    # star_pos            =   dd['PartType4','Coordinates'][galaxy_slist].in_units('kpc')
    # star_vel            =   dd['PartType4','Velocities'][galaxy_slist].in_cgs()/1e5
    # star_m              =   dd['PartType4','Masses'][galaxy_slist].in_units('Msun')
    # star_a_C            =   dd['PartType4','Metallicity_00'][galaxy_slist].d
    # star_a_O            =   dd['PartType4','Metallicity_01'][galaxy_slist].d
    # star_a_Si           =   dd['PartType4','Metallicity_02'][galaxy_slist].d
    # star_a_Fe           =   dd['PartType4','Metallicity_03'][galaxy_slist].d
    # star_Z              =   (star_a_C+star_a_O+star_a_Si+star_a_Fe) * 0.0189/0.0147 / 0.02       # from RT
    # current_time        =   ds.current_time.in_units('yr')
    # star_age            =   (current_time.d-dd['PartType4','StellarFormationTime'][galaxy_slist].d)/1e6

    # if verbose: print('Loading all DM particle properties...')
    # DM_pos              =   dd['PartType1','Coordinates'][galaxy_slist].in_units('kpc')
    # DM_vel              =   dd['PartType1','Velocities'][galaxy_slist].in_cgs()/1e5
    # DM_m                =   dd['PartType1','Masses'][galaxy_slist].in_units('Msun')

    # Center coordinates on galaxy
    # gas_pos             =   gas_pos-obj.galaxies[GAL].pos.in_units('kpc')
    # star_pos            =   star_pos-obj.galaxies[GAL].pos.in_units('kpc')
    # DM_pos              =   DM_pos-obj.galaxies[GAL].pos.in_units('kpc')

    # Test what smoothing kernel was used
    # r           =   np.arange(1e-6,3.*gas_h[0].d,3.*gas_h[0].d/100.)                    # kpc
    # rho_r1      =   gas_m[0].d*1/gas_h[0].d**3*kernel(r/gas_h[0].d,'quintic')           # Msun/kpc^3
    # M1          =   scipy.integrate.simps(rho_r1*4*np.pi*r**2.,r)
    # r           =   np.arange(1e-6,3.*gas_h[0].d,2.*gas_h[0].d/100.)                    # kpc
    # rho_r2      =   gas_m[0].d*1/gas_h[0].d**3*kernel(r/gas_h[0].d,'cubic')             # Msun/kpc^3
    # M2          =   scipy.integrate.simps(rho_r2*4*np.pi*r**2.,r)
    # kernel_test =   np.array([M1,M2])
    # index       =   np.argmin(abs(kernel_test-gas_m[0].d))
    # if index == 0: kernel_type = 'quintic'
    # if index == 1: kernel_type = 'cubic'
    # if index == 2: kernel_type = 'simple'
    # print('Kernel seems to be '+kernel_type)
    # if verbose: print('Actual mass of particle 0: ',gas_m[0].d)
    # if verbose: print('Integrated mass of particle 0: ',kernel_test[index])

    gas_data   =   pd.DataFrame({'x':gas_posx,'y':gas_posy,'z':gas_posz,\
        'vx':gas_velx,'vy':gas_vely,'vz':gas_velz,\
        'SFR':gas_SFR,'Z':gas_Z,'nH':gas_densities,'Tk':gas_Tk,'h':gas_h,\
        'f_HI':gas_f_HI,'f_ion':gas_f_ion,'m':gas_m})
    star_data   =   pd.DataFrame({'x':star_posx,'y':star_posy,'z':star_posz,\
        'vx':star_velx,'vy':star_vely,'vz':star_velz,\
        'Z':star_Z,'m':star_m,'age':star_age})
    # DM_data     =   pd.DataFrame({'x':DM_pos[:,[0]].d.T[0],'y':DM_pos[:,[1]].d.T[0],'z':DM_pos[:,[2]].d.T[0],\
        # 'vx':DM_vel[:,[0]].d.T[0],'vy':DM_vel[:,[1]].d.T[0],'vz':DM_vel[:,[2]].d.T[0],\
        # 'm':DM_m.d})
    return star_data,gas_data

def load_gadget_caesar(GAL,snap,member_search=False,verbose=True):
    print('\n * Simulation data is in gadget format *')

    # query all available members
    MEMBER      =   d_sph+'gadget-3/members/members.'+str(int(snap))+'.hdf5'
    SNAPSHOT    =   d_sph+'gadget-3/snapshots/snapshot.'+str(int(snap))+'.hdf5'

    # Level of written output from yt
    yt.funcs.mylog.setLevel(0)
    if verbose: yt.funcs.mylog.setLevel(1)

    # load the raw dataset into yt
    ds          =   yt.load(SNAPSHOT,over_refine_factor = 1)        # Set over_refine_factor to a higher value if your want nicer images (will take time!)
    print('Simulation type: '+ds.dataset_type)

    # Make member search on snapshot?
    if member_search:
        i           =   0
        while os.path.isfile(d_sph+'gadget-3/snapshots/snapshot.'+str(int(i))+'.hdf5'):
            ds          =   yt.load(d_sph+'gadget-3/snapshots/snapshot.'+str(int(i))+'.hdf5',over_refine_factor = 1)        # Set over_refine_factor to a higher value if your want nicer images (will take time!)
            obj         =   caesar.CAESAR(ds)
            obj.member_search()
            obj.save(d_sph+'gadget-3/members/members.'+str(int(i))+'.hdf5')
            i           +=  1

    # load the current member file
    obj         =   caesar.load(MEMBER)

    print 'Number of galaxies found: '+str(obj.ngalaxies)

    # Print Ngal biggest galaxies (10 galaxies by default)
    Ngal        =   obj.ngalaxies
    if verbose: print(obj.galinfo(top=Ngal))

    # Select gas/star particles in ONE galaxy
    gal                 =   obj.galaxies[GAL]
    galaxy_glist        =   gal.glist
    galaxy_slist        =   gal.slist

    # Check contamination
    # gal.contamination_check()
    # pdb.set_trace()

    if verbose: print('Loading all gas particle properties...')

    dd                  =   ds.all_data()     # to get abundances and ages out...
    # gas_f_HI            =   dd['PartType0','NeutralHydrogenAbundance'][galaxy_glist].d          # fraction of Hydrogen atoms that are NOT ionized
    gas_x_e             =   dd['PartType0','ElectronAbundance'][galaxy_glist].d                 # electrons per Hydrogen atom (max: 1.15)
    gas_f_ion           =   gas_x_e/max(gas_x_e)                                                # ionized gas mass fraction
    gas_f_HI            =   1.-gas_x_e/max(gas_x_e)                                             # because we don't trust NeutralHydrogenAbundance
    # gas_f_HI            =   dd['PartType0','NeutralHydrogenAbundance'][galaxy_glist].d
    gas_pos             =   dd['PartType0','Coordinates'].in_units('kpc')[galaxy_glist]         # kpc
    gas_vel             =   dd['PartType0','Velocities'][galaxy_glist].in_cgs()/1e5             # km/s
    gas_densities       =   dd['PartType0','Density'][galaxy_glist].in_cgs()/(mH*1.e3)          # cm^-3
    gas_m               =   dd['PartType0','Masses'][galaxy_glist].in_units('Msun')             # Msun
    pdb.set_trace()

    gas_Tk              =   dd['PartType0','Temperature'][galaxy_glist].d                       # Tk
    gas_h               =   dd['PartType0','SmoothingLength'][galaxy_glist].in_units('kpc')     # Tk
    gas_SFR             =   dd['PartType0','StarFormationRate'][galaxy_glist].d                 # Msun/yr
    gas_a_C             =   dd['PartType0','Metallicity_00'][galaxy_glist].d
    gas_a_O             =   dd['PartType0','Metallicity_01'][galaxy_glist].d
    gas_a_Si            =   dd['PartType0','Metallicity_02'][galaxy_glist].d
    gas_a_Fe            =   dd['PartType0','Metallicity_03'][galaxy_glist].d
    gas_Z               =   (gas_a_C+gas_a_O+gas_a_Si+gas_a_Fe) * 0.0189/0.0147 / 0.02          # from RT

    if verbose: print('Loading all star particle properties...')
    # position, velocity, mass, metallicity
    star_pos            =   dd['PartType4','Coordinates'][galaxy_slist].in_units('kpc')
    star_vel            =   dd['PartType4','Velocities'][galaxy_slist].in_cgs()/1e5
    star_m              =   dd['PartType4','Masses'][galaxy_slist].in_units('Msun')
    star_a_C            =   dd['PartType4','Metallicity_00'][galaxy_slist].d
    star_a_O            =   dd['PartType4','Metallicity_01'][galaxy_slist].d
    star_a_Si           =   dd['PartType4','Metallicity_02'][galaxy_slist].d
    star_a_Fe           =   dd['PartType4','Metallicity_03'][galaxy_slist].d
    star_Z              =   (star_a_C+star_a_O+star_a_Si+star_a_Fe) * 0.0189/0.0147 / 0.02       # from RT
    current_time        =   ds.current_time.in_units('yr')
    star_age            =   (current_time.d-dd['PartType4','StellarFormationTime'][galaxy_slist].d)/1e6

    # if verbose: print('Loading all DM particle properties...')
    # DM_pos              =   dd['PartType1','Coordinates'][galaxy_slist].in_units('kpc')
    # DM_vel              =   dd['PartType1','Velocities'][galaxy_slist].in_cgs()/1e5
    # DM_m                =   dd['PartType1','Masses'][galaxy_slist].in_units('Msun')

    # Center coordinates on galaxy
    gas_pos             =   gas_pos-obj.galaxies[GAL].pos.in_units('kpc')
    star_pos            =   star_pos-obj.galaxies[GAL].pos.in_units('kpc')
    # DM_pos              =   DM_pos-obj.galaxies[GAL].pos.in_units('kpc')

    # Rotate coordinates + velocities to be aligned with xy-plane
    gas_pos             =   caesar.utils.rotator(gas_pos, gal.rotation_angles['ALPHA'], gal.rotation_angles['BETA'])
    gas_vel             =   caesar.utils.rotator(gas_vel, gal.rotation_angles['ALPHA'], gal.rotation_angles['BETA'])
    star_pos            =   caesar.utils.rotator(star_pos, gal.rotation_angles['ALPHA'], gal.rotation_angles['BETA'])
    star_vel            =   caesar.utils.rotator(star_vel, gal.rotation_angles['ALPHA'], gal.rotation_angles['BETA'])
    # DM_pos              =   caesar.utils.rotator(DM_pos, gal.rotation_angles['ALPHA'], gal.rotation_angles['BETA'])
    # DM_vel              =   caesar.utils.rotator(DM_vel, gal.rotation_angles['ALPHA'], gal.rotation_angles['BETA'])

    gas_data   =   pd.DataFrame({'x':gas_pos[:,[0]].d.T[0],'y':gas_pos[:,[1]].d.T[0],'z':gas_pos[:,[2]].d.T[0],\
        'vx':gas_vel[:,[0]].d.T[0],'vy':gas_vel[:,[1]].d.T[0],'vz':gas_vel[:,[2]].d.T[0],\
        'SFR':gas_SFR,'Z':gas_Z,'nH':gas_densities.d,'Tk':gas_Tk,'h':gas_h.d,\
        'f_HI':gas_f_HI,'f_ion':gas_f_ion,'m':gas_m.d,\
        'aC':gas_a_C,'aO':gas_a_O,'aSi':gas_a_Si,'aFe':gas_a_Fe})
    star_data   =   pd.DataFrame({'x':star_pos[:,[0]].d.T[0],'y':star_pos[:,[1]].d.T[0],'z':star_pos[:,[2]].d.T[0],\
        'vx':star_vel[:,[0]].d.T[0],'vy':star_vel[:,[1]].d.T[0],'vz':star_vel[:,[2]].d.T[0],\
        'Z':star_Z,'m':star_m.d,'age':star_age})
    # DM_data     =   pd.DataFrame({'x':DM_pos[:,[0]].d.T[0],'y':DM_pos[:,[1]].d.T[0],'z':DM_pos[:,[2]].d.T[0],\
        # 'vx':DM_vel[:,[0]].d.T[0],'vy':DM_vel[:,[1]].d.T[0],'vz':DM_vel[:,[2]].d.T[0],\
        # 'm':DM_m.d})
    return star_data,gas_data

def load_gizmo(GAL,snap,verbose=True):
    print('\n * Simulation data is in gadget format but from gizmo *')

    # query all available members
    MEMBER      =   d_sph+'gadget-3/members/sphgr_snapshot.'+str(int(snap))+'.hdf5'
    SNAPSHOT    =   d_sph+'gadget-3/snapshots/snapshot.'+str(int(snap))+'.hdf5'

    # load the raw dataset into yt
    ds          =   yt.load(SNAPSHOT,over_refine_factor = 1)        # Set over_refine_factor to a higher value if your want nicer images (will take time!)

    print('Simulation type')
    print(ds.dataset_type)

    # load the current member file
    obj         =   sphgr.load_sphgr_data(MEMBER,ds)

    print 'Number of galaxies found: '+str(obj.ngalaxies)

    # Print Ngal biggest galaxies (10 galaxies by default)
    Ngal        =   obj.ngalaxies
    print(obj.galinfo(top=Ngal))

    # Select gas/star particles in ONE galaxy
    gal                 =   obj.galaxies[GAL]
    galaxy_glist        =   gal.glist
    galaxy_slist        =   gal.slist
    dd                  =   ds.all_data()     # to get abundances and ages out...

    # Check contamination
    gal.contamination_check()

    print('Loading all gas particle properties...')
    # position, velocity, mass, density, Tk, SFR, x_e, h
    gas_pos             =   obj.particle_data['gpos'][galaxy_glist].in_units('kpc')
    gas_vel             =   obj.particle_data['gvel'][galaxy_glist].in_cgs()/1e5
    gas_m               =   obj.particle_data['gmass'][galaxy_glist].in_units('Msun')
    gas_densities       =   obj.particle_data['grho'][galaxy_glist].in_cgs()            #  cgs is g/cm^3 !!
    gas_Tk              =   obj.particle_data['gtemp'][galaxy_glist]
    gas_sfr             =   obj.particle_data['gsfr'][galaxy_glist]
    gas_xe              =   obj.particle_data['gne'][galaxy_glist]
    gas_h               =   obj.particle_data['ghsml'][galaxy_glist].in_units('kpc')
    # atomic and molecular gas mass (FIRE simulations don't have H2 fractions, but Robert added a function to calculate it for them)
    all_gas_mHI,all_gas_mH2 =   obj.get_hydrogen_masses()
    gas_mHI,gas_mH2         =   all_gas_mHI[galaxy_glist],all_gas_mH2[galaxy_glist]
    # metallicity
    a_He                =   dd['PartType0','Metallicity_01'][galaxy_glist].d     # He
    a_C                 =   dd['PartType0','Metallicity_02'][galaxy_glist].d
    a_N                 =   dd['PartType0','Metallicity_03'][galaxy_glist].d
    a_O                 =   dd['PartType0','Metallicity_04'][galaxy_glist].d
    a_Ne                =   dd['PartType0','Metallicity_05'][galaxy_glist].d
    a_Mg                =   dd['PartType0','Metallicity_06'][galaxy_glist].d
    a_Si                =   dd['PartType0','Metallicity_07'][galaxy_glist].d
    a_S                 =   dd['PartType0','Metallicity_08'][galaxy_glist].d
    a_Ca                =   dd['PartType0','Metallicity_09'][galaxy_glist].d
    a_Fe                =   dd['PartType0','Metallicity_10'][galaxy_glist].d
    gas_Z               =   obj.particle_data['gmetallicity'][galaxy_glist] / 0.02             # from DN
    gas_Z               =   (a_C+a_O+a_Si+a_Fe) * 0.0189/0.0147 / 0.02           # from RT
    print('Loading all star particle properties...')
    # position, velocity, mass, metallicity
    star_pos            =   obj.particle_data['spos'][galaxy_slist].in_units('kpc')
    star_vel            =   obj.particle_data['svel'][galaxy_slist].in_cgs()/1e5
    star_m              =   obj.particle_data['smass'][galaxy_slist].in_units('Msun')
    star_Z              =   obj.particle_data['smetallicity'][galaxy_slist]
    current_time        =   ds.current_time.in_units('yr')
    all_star_time       =   dd['PartType4','StellarFormationTime']
    all_star_redshift   =   1./all_star_time-1.
    all_star_age        =   ds.cosmology.t_from_z(zred).to('Myr')-ds.cosmology.t_from_z(all_star_redshift).to('Myr')  # properly takes into account cosmology etc
    star_age            =   all_star_age[galaxy_slist]
    print('Loading all DM particle properties...')
    # DM_pos              =   obj.particle_data['dmpos'][galaxy_slist].in_units('kpc')
    # DM_m                =   obj.particle_data['dmmass'][galaxy_slist].in_units('Msun')
    # DM_vel              =   obj.particle_data['dmvel'][galaxy_slist].in_cgs()/1e5

    # Center coordinates on galaxy
    gas_pos             =   gas_pos-obj.galaxies[GAL].pos.in_units('kpc')
    star_pos            =   star_pos-obj.galaxies[GAL].pos.in_units('kpc')
    # DM_pos              =   DM_pos-obj.galaxies[GAL].pos.in_units('kpc')

    # Rotate coordinates + velocities to be aligned with xy-plane
    gas_pos             =   gf.rotator_3d(gas_pos, gal.rotation_angles['ALPHA'], gal.rotation_angles['BETA'])
    gas_vel             =   gf.rotator_3d(gas_vel, gal.rotation_angles['ALPHA'], gal.rotation_angles['BETA'])
    star_pos            =   gf.rotator_3d(star_pos, gal.rotation_angles['ALPHA'], gal.rotation_angles['BETA'])
    star_vel            =   gf.rotator_3d(star_vel, gal.rotation_angles['ALPHA'], gal.rotation_angles['BETA'])
    # DM_pos              =   gf.rotator_3d(DM_pos, gal.rotation_angles['ALPHA'], gal.rotation_angles['BETA'])
    # DM_vel              =   gf.rotator_3d(DM_vel, gal.rotation_angles['ALPHA'], gal.rotation_angles['BETA'])

    # Test what smoothing kernel was used
    r           =   np.arange(1e-6,3.*gas_h[0].d,3.*gas_h[0].d/100.)                    # kpc
    rho_r1      =   gas_m[0].d*1/gas_h[0].d**3*kernel(r/gas_h[0].d,'quintic')           # Msun/kpc^3
    M1          =   scipy.integrate.simps(rho_r1*4*np.pi*r**2.,r)
    r           =   np.arange(1e-6,3.*gas_h[0].d,2.*gas_h[0].d/100.)                    # kpc
    rho_r2      =   gas_m[0].d*1/gas_h[0].d**3*kernel(r/gas_h[0].d,'cubic')           # Msun/kpc^3
    M2          =   scipy.integrate.simps(rho_r2*4*np.pi*r**2.,r)
    kernel_test =   np.array([M1,M2])
    index       =   np.argmin(abs(kernel_test-gas_m[0].d))
    if index == 0: kernel_type = 'quintic'
    if index == 1: kernel_type = 'cubic'
    if index == 2: kernel_type = 'simple'
    print('Kernel seems to be '+kernel_type)
    print('Actual mass of particle 0: ',gas_m[0].d)
    print('Integrated mass of particle 0: ',kernel_test[index])

    gas_data    =   pd.DataFrame({'x':gas_pos[:,[0]].d.T[0],'y':gas_pos[:,[1]].d.T[0],'z':gas_pos[:,[2]].d.T[0],\
        'vx':gas_vel[:,[0]].d.T[0],'vy':gas_vel[:,[1]].d.T[0],'vz':gas_vel[:,[2]].d.T[0],\
        'SFR':gas_sfr.d,'Z':gas_Z.d,'nH':gas_densities.d,'Tk':gas_Tk.d,'h':gas_h.d,'x_e':gas_xe.d,\
        'f_HI':gas_mHI/gas_m.d,'f_H2':gas_mH2/gas_m.d,'m':gas_m.d,\
        'aHe':a_He,'aC':a_C,'aN':a_N,'aO':a_O,'aNe':a_Ne,'aMg':a_Mg,'aSi':a_Si,'aSi':a_S,'aCa':a_Ca,'aFe':a_Fe})
    star_data   =   pd.DataFrame({'x':star_pos[:,[0]].d.T[0],'y':star_pos[:,[1]].d.T[0],'z':star_pos[:,[2]].d.T[0],\
        'vx':star_vel[:,[0]].d.T[0],'vy':star_vel[:,[1]].d.T[0],'vz':star_vel[:,[2]].d.T[0],\
        'Z':star_Z.d,'m':star_m.d,'age':star_age.d})
    # DM_data     =   pd.DataFrame({'x':DM_pos[:,[0]].d.T[0],'y':DM_pos[:,[1]].d.T[0],'z':DM_pos[:,[2]].d.T[0],\
        # 'vx':DM_vel[:,[0]].d.T[0],'vy':DM_vel[:,[1]].d.T[0],'vz':DM_vel[:,[2]].d.T[0],\
        # 'm':DM_m.d})

    return star_data,gas_data

def load_JSL(GAL,snap,member_search=True,verbose=True):

    print('\n * Simulation data is from Jesper Sommer-Larsen (JSL) *')

    columns             =   ['x','y','z','vx','vy','vz','lognH','logTk','logfHI','logh','logSFR','m','logZ']
    gas                 =   pd.read_table(d_sph+'JSL/NC'+str(snap)+'_2.gas',names=columns,sep='\s*',engine='python')
    columns             =   ['x','y','z','vx','vy','vz','m','age','logZ']
    star                =   pd.read_table(d_sph+'JSL/NC'+str(snap)+'_2.stars',names=columns,sep='\s*',engine='python')
    columns             =   ['x','y','z','vx','vy','vz','m']
    # DM                  =   pd.read_table(d_sph+'JSL/NC'+str(snap)+'_2.DM',names=columns,sep='\s*',engine='python')
    gas['f_HI']         =   10.**gas['logfHI']
    gas['f_ion']        =   1.-gas['f_HI']
    gas['nH']           =   10.**gas['lognH']
    gas['Tk']           =   10.**gas['logTk']
    gas['h']            =   10.**gas['logh']
    gas['Z']            =   10.**gas['logZ']
    gas['SFR']          =   10.**gas['logSFR']


    pdb.set_trace()

    # rename columns accordingly:
    gas_data            =   gas[['x','y','z','vx','vy','vz','nH','Tk','h','Z','SFR','m','f_HI','f_ion']]
    gas_data.columns    =   ['x','y','z','vx','vy','vz','nH','Tk','h','Z','SFR','m','f_HI','f_ion']      # not sure about x_e !!!
    star['Z']           =   10.**star['logZ']
    star['age']         =   star['age']         # [Myr]
    star_data           =   star[['x','y','z','vx','vy','vz','m','age','Z']]
    star_data.columns   =   ['x','y','z','vx','vy','vz','m','age','Z']      # not sure about x_e !!!
    # DM_data             =   DM[['x','y','z','vx','vy','vz','m']]
    # DM_data.columns     =   ['x','y','z','vx','vy','vz','m']      # not sure about x_e !!!
    # Test what smoothing kernel was used
    r                   =   np.arange(1e-6,3.*gas_data['h'][0],3.*gas_data['h'][0]/100.)                    # kpc
    rho_r1              =   gas_data['m'][0]*1/gas_data['h'][0]**3*kernel(r/gas_data['h'][0],'quintic')           # Msun/kpc^3
    M1                  =   scipy.integrate.simps(rho_r1*4*np.pi*r**2.,r)
    r                   =   np.arange(1e-6,3.*gas_data['h'][0],2.*gas_data['h'][0]/100.)                    # kpc
    rho_r2              =   gas_data['m'][0]*1/gas_data['h'][0]**3*kernel(r/gas_data['h'][0],'cubic')           # Msun/kpc^3
    M2                  =   scipy.integrate.simps(rho_r2*4*np.pi*r**2.,r)
    kernel_test         =   np.array([M1,M2])
    index               =   np.argmin(abs(kernel_test-gas_data['m'][0]))
    if index == 0: kernel_type = 'quintic'
    if index == 1: kernel_type = 'cubic'
    print('Kernel seems to be '+kernel_type)
    print('Actual mass of particle 0: ',gas_data['m'][0])
    print('Integrated mass of particle 0: ',kernel_test[index])
    # quick check of rotation
    fig                 =   plt.figure(0)
    ax1                 =   fig.add_axes([0.15,0.1,0.75,0.8])
    # ax1.plot(gas_data['y'],gas_data['z'],'.',ms=2)
    ax1.plot(star_data['x'],star_data['y'],'.',ms=2)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_xlim([-20,20])
    ax1.set_ylim([-20,20])
    plt.show(block=False)

    # TEST: impose a density cut at 0.13cm^-3 as in CAESAR
    print(len(gas_data))
    gas_data            =   gas_data[gas_data['nH'] >  0.13]
    print(len(gas_data))

    return star_data,gas_data

def load_gadget_txt(GAL,snap,galname,verbose=True):

    print('\n* Simulation data is from Gadget but in text format *')

    galname             =   's'+str(int(snap))+'_G'+str(int(GAL))
    columns             =   ['x','y','z','vx','vy','vz','density','f_HI','fH2','sigmav','Tk','Z','SFR','h','m','Ne1']
    gas                 =   pd.read_table(d_sph+'gadget-3/'+galname+'_SPH.gas',names=columns,sep='\s*',engine='python',skiprows=3)
    # Convert from co-moving to physical scales
    gas[pos]            =   gas[pos]/(1.+zred)
    # Positions have h divided out, but smoothing lengths not
    gas['h']            =   gas['h']*0.7
    gas['f_ion']        =   gas['Ne1']/max(gas['Ne1'])
    columns             =   ['aC','aO','aSi','aFe']
    gasZ                =   pd.read_table(d_sph+'gadget-3/'+galname+'_SPH.gasZ',names=columns,sep='\s*',engine='python',skiprows=2)

    gas_Z               =   (gasZ['aC']+gasZ['aO']+gasZ['aSi']+gasZ['aFe']) * 0.0189/0.0147 / 0.02          # from RT

    columns             =   ['x','y','z','vx','vy','vz','m','age']
    star                =   pd.read_table(d_sph+'gadget-3/'+galname+'_SPH.star',names=columns,sep='\s*',engine='python',skiprows=3)
    star[pos]           =   star[pos]/(1.+zred)
    columns             =   ['a_C','a_O','a_Si','a_Fe']
    starZ               =   pd.read_table(d_sph+'gadget-3/'+galname+'_SPH.starZ',names=columns,sep='\s*',engine='python',skiprows=3)

    # rename columns accordingly:
    gas['nH']           =   gas['density'] # g/cm^3
    radius              =   np.sqrt(gas['x'].values**2.+gas['y'].values**2.+gas['z'].values**2.)
    gas                 =   gas[radius < 10.] # making a rough cutout as in Olsen+15
    gas_data            =   gas[['x','y','z','vx','vy','vz','nH','Tk','h','Z','SFR','m','f_ion','f_HI']]
    gas_data.columns    =   ['x','y','z','vx','vy','vz','nH','Tk','h','Z','SFR','m','f_ion','f_HI']
    star['Z']           =   (starZ['a_C']+starZ['a_O']+starZ['a_Si']+starZ['a_Fe'])*0.0189/0.0147 # mail from Robert
    if zred == 2: star['age']         =   3.223e3-star['age']/1e6                # [Myr] star['age'] = age of Universe [yr] when star particle was spawned, age @ z=2: 3.223e3 Myr with cosm. parameters used (h = 0.7, Omega_m = 0.3?)
    star_data           =   star[['x','y','z','vx','vy','vz','m','age','Z']]
    star_data.columns   =   ['x','y','z','vx','vy','vz','m','age','Z']      # not sure about x_e !!!

    # DM_data             =   pd.DataFrame({'x':[],'y':[],'z':[],'vx':[],'vy':[],'vz':[],'m':[]})     # empty DM frame

    return star_data,gas_data

def load_illustris(GAL,snap,verbose=True):
    print('\n * Simulation data is from Illustris *')
    # Scripts here: http://www.illustris-project.org/data/docs/scripts/ (but can't make them work on cutouts from catalog)

    data                    =   h5py.File(d_sph + 'illustris/cutouts/cutout_s'+str(int(snap))+'_G'+str(int(GAL))+'.hdf5')

    # Meta-data from raw snapshots downloaded (does not include cosmological parameters)
    # print(data['Header'].attrs.get()

    Omega_lambda, Omega_m, hubble_constant  =   0.7274, 0.2726, 0.704   # from reading raw snapshot
    a                       =   1./(1+zred)                             # scale factor at this z

    gas                     =   data.get('PartType0')
    star                    =   data.get('PartType4')
    # DM                      =   data.get('PartType1')

    gas_pos                 =   np.array(gas.get('Coordinates'))*hubble_constant/(1+zred) # co-moving kpc/h
    gas_vel                 =   np.array(gas.get('Velocities'))*np.sqrt(a) # km*sqrt(a)/s
    cgs_unit                =   1e10*Msun*1000./hubble_constant/(1./(1+zred)/hubble_constant*kpc2cm)**3.
    gas_density             =   np.array(gas.get('Density'))*cgs_unit # g/cm^3
    gas_ne                  =   np.array(gas.get('ElectronAbundance'))
    gas_m                   =   np.array(gas.get('Masses'))*1e10/hubble_constant
    gas_f_HI                =   np.array(gas.get('NeutralHydrogenAbundance'))/max(np.array(gas.get('NeutralHydrogenAbundance'))) # mass fraction of hydrogen that is neutral
    gas_f_ion               =   1.-gas_f_HI # mass fraction of hydrogen that is ionized
    gas_h                   =   np.array(gas.get('SmoothingLength'))*hubble_constant/(1+zred) # co-moving kpc/h
    # OBS on h: Twice the maximum radius of all Delaunay tetrahedra that have this cell at a vertex in comoving units (s_i from Springel et al. 2010).
    gas_SFR                 =   np.array(gas.get('StarFormationRate'))
    gas_Z                   =   np.array(gas.get('GFM_Metallicity'))/0.0127 # Solar metallicity units
    gas_E                   =   np.array(gas.get('InternalEnergy')) # (km/s)^2, internal (thermal) energy per unit mass

    # Calculate temperature (email from Martin, see also http://www.illustris-project.org/data/docs/faq/#snap1)
    g_gamma                 =   5.0/3.0
    g_minus_1               =   g_gamma-1.0
    XH                      =   0.76                # abundance of Hydrogen
    yhelium                 =   (1-XH)/(4*XH)
    mu                      =   (1+4*yhelium)/(1+yhelium+gas_ne)
    MeanWeight              =   mu*m_p
    gas_Tk                  =   MeanWeight/kB_ergs * g_minus_1 * gas_E * 1e10

    star_pos                =   np.array(star.get('Coordinates'))*hubble_constant/(1+zred) # co-moving kpc/h
    star_vel                =   np.array(star.get('Velocities'))*np.sqrt(a) # km*sqrt(a)/s
    star_m                  =   np.array(star.get('Masses'))*1e10/hubble_constant
    star_formation_a        =   np.array(star.get('GFM_StellarFormationTime')) # formation time as scale factor, a = 1/(1+z)
    star_formation_z        =   1./star_formation_a-1
    # Code from yt project (yt.utilities.cosmology)
    star_formation_t        =   2.0/3.0/np.sqrt(1-Omega_m)*np.arcsinh(np.sqrt((1-Omega_m)/Omega_m)/ np.power(1+star_formation_z, 1.5))/(hubble_constant) # Mpc*s/(100*km)
    time_z                  =   2.0/3.0/np.sqrt(1-Omega_m)*np.arcsinh(np.sqrt((1-Omega_m)/Omega_m)/ np.power(1+zred, 1.5))/(hubble_constant) # Mpc*s/(100*km)
    star_formation_t        =   star_formation_t*kpc2m/100./(1e6*365.25*86400) # Myr
    time_z                  =   time_z*kpc2m/100./(1e6*365.25*86400) # Myr
    star_age                =   time_z-star_formation_t # Myr
    star_Z                  =   np.array(star.get('GFM_Metallicity'))/0.0127 # Solar metallicity units

    # DM_pos                  =   np.array(DM.get('Coordinates'))*hubble_constant/(1+zred) # co-moving kpc/h
    # DM_vel                  =   np.array(DM.get('Velocities'))*np.sqrt(a) # km*sqrt(a)/s
    # DM_m                    =   np.zeros(len(DM_vel)) # can't find it...


    # Making a rouch translation in position, velocity:
    gas_x                   =   gas_pos[:,[0]].squeeze()-np.median(gas_pos[:,[0]].squeeze())
    gas_y                   =   gas_pos[:,[1]].squeeze()-np.median(gas_pos[:,[1]].squeeze())
    gas_z                   =   gas_pos[:,[2]].squeeze()-np.median(gas_pos[:,[2]].squeeze())
    gas_vel_x               =   gas_vel[:,[0]].squeeze()-np.median(gas_vel[:,[0]].squeeze())
    gas_vel_y               =   gas_vel[:,[1]].squeeze()-np.median(gas_vel[:,[1]].squeeze())
    gas_vel_z               =   gas_vel[:,[2]].squeeze()-np.median(gas_vel[:,[2]].squeeze())
    star_x                  =   star_pos[:,[0]].squeeze()-np.median(star_pos[:,[0]].squeeze())
    star_y                  =   star_pos[:,[1]].squeeze()-np.median(star_pos[:,[1]].squeeze())
    star_z                  =   star_pos[:,[2]].squeeze()-np.median(star_pos[:,[2]].squeeze())
    star_vel_x              =   star_vel[:,[0]].squeeze()-np.median(star_vel[:,[0]].squeeze())
    star_vel_y              =   star_vel[:,[1]].squeeze()-np.median(star_vel[:,[1]].squeeze())
    star_vel_z              =   star_vel[:,[2]].squeeze()-np.median(star_vel[:,[2]].squeeze())

    gas_data   =   pd.DataFrame({'x':gas_x,'y':gas_y,'z':gas_z,\
        'vx':gas_vel_x,'vy':gas_vel_y,'vz':gas_vel_z,\
        'SFR':gas_SFR,'Z':gas_Z,'nH':gas_density,'Tk':gas_Tk,'h':gas_h,\
        'f_HI':gas_f_HI,'f_ion':gas_f_ion,'m':gas_m})

    star_data   =   pd.DataFrame({'x':star_x,'y':star_y,'z':star_z,\
        'vx':star_vel_x,'vy':star_vel_y,'vz':star_vel_z,\
        'Z':star_Z,'m':star_m,'age':star_age})
    star_data   =   star_data[star_formation_a > 0] # the rest are wind particles!

    # DM_data   =   pd.DataFrame({'x':DM_pos[:,[0]].squeeze(),'y':DM_pos[:,[1]].squeeze(),'z':DM_pos[:,[2]].squeeze(),\
        # 'vx':DM_vel[:,[0]].squeeze(),'vy':DM_vel[:,[1]].squeeze(),'vz':DM_vel[:,[2]].squeeze(),'m':DM_m})

    # pdb.set_trace()

    print('Done!')

    return star_data,gas_data

def center_cut_galaxy(simgas,simstar,simdm,plot=False):
    print(' ** Center galaxy in spatial and velocity coordinates, gas+stars, and cut out! **')
    plt.close('all')        # close all windows

    print('Number of gas particles: '+  str(len(simgas)))
    print('Number of star particles: '+ str(len(simstar)))
    print('Number of DM particles: '+   str(len(simdm)))

    # Define R_gal from simulation cutout
    R_gal           =   max(np.sqrt(simgas['x']**2.+simgas['y']**2.+simgas['z']**2.))
    print('R_gal = ' + str.format("{0:.2f}",R_gal) + ' kpc (from simulation)')

    print('Center all in x,y,z spatial coordinates (according to stellar distribution)')
    dat             =   simstar
    bins            =   200.
    r               =   np.sqrt(dat['x'].values**2.+dat['y'].values**2.+dat['z'].values**2.)
    # r_bin           =   np.arange(-max(r),max(r),2.*max(r)/bins)
    r_bin           =   np.arange(-2,2,4./bins)
    # Average mass surface density in radial bins:
    m_binned_x      =   np.array([sum(dat['m'][(dat['x'] >= r_bin[i]) & (dat['x'] < r_bin[i+1])]) for i in range(0,len(r_bin)-1)])
    m_binned_y      =   np.array([sum(dat['m'][(dat['y'] >= r_bin[i]) & (dat['y'] < r_bin[i+1])]) for i in range(0,len(r_bin)-1)])
    m_binned_z      =   np.array([sum(dat['m'][(dat['z'] >= r_bin[i]) & (dat['z'] < r_bin[i+1])]) for i in range(0,len(r_bin)-1)])
    # Smooth out profiles a bit:
    m_binned_x1       =   lowess(m_binned_x,r_bin[0:len(r_bin)-1],frac=0.1,is_sorted=True,it=0)
    m_binned_y1       =   lowess(m_binned_y,r_bin[0:len(r_bin)-1],frac=0.1,is_sorted=True,it=0)
    m_binned_z1       =   lowess(m_binned_z,r_bin[0:len(r_bin)-1],frac=0.1,is_sorted=True,it=0)
    # find max of distribution:
    xpos            =   r_bin[np.argmax(m_binned_x1[:,1])]
    ypos            =   r_bin[np.argmax(m_binned_y1[:,1])]
    zpos            =   r_bin[np.argmax(m_binned_z1[:,1])]
    print('corrections: ',xpos,ypos,zpos)
    # move original coordinates
    simgas[pos]     =   simgas[pos]-[xpos,ypos,zpos]
    simstar[pos]    =   simstar[pos]-[xpos,ypos,zpos]
    simdm[pos]      =   simdm[pos]-[xpos,ypos,zpos]
    r_bin           =   r_bin[0:len(r_bin)-1]
    if plot:
        plt.figure(0)
        plt.plot(r_bin,m_binned_x,'r')
        plt.plot(r_bin,m_binned_y,'g')
        plt.plot(r_bin,m_binned_z,'b')
        plt.plot(r_bin,m_binned_x1[:,1],'--r')
        plt.plot(r_bin,m_binned_y1[:,1],'--g')
        plt.plot(r_bin,m_binned_z1[:,1],'--b')
        # plt.plot([xpos,xpos],[0,1e10],'--r')
        # plt.plot([ypos,ypos],[0,1e10],'--g')
        # plt.plot([zpos,zpos],[0,1e10],'--b')
        plt.xlabel('x (r) y (g) z (b) [kpc]')
        plt.ylabel('accumulated stellar mass [M$_{\odot}$]')
        plt.title('Centering in [x,y,z]')
        plt.show(block=False)

    print('Center all in velocity space (vx,vy,vz) (according to gas distribution)')
    dat             =   simgas
    # use gas to center galaxy in velocity space (as if observing)
    ngrid           =   1000
    grid,vxd,vyd,vzd    =   ([0]*ngrid for i in range(4))
    grid[0]        =    -600.
    for i in range(1,len(grid)):
        grid[i]         =   grid[i-1]+2*(-grid[0])/ngrid
        vxd[i]          =   sum(dat.loc[dat.loc[:,'vx']<grid[i],'m'])
        vyd[i]          =   sum(dat.loc[dat.loc[:,'vy']<grid[i],'m'])
        vzd[i]          =   sum(dat.loc[dat.loc[:,'vz']<grid[i],'m'])
    # find the position where half of the mass has accumulated
    vxpos           =   max(np.array(grid)[np.array(vxd)<max(vxd)/2])
    vypos           =   max(np.array(grid)[np.array(vyd)<max(vyd)/2])
    vzpos           =   max(np.array(grid)[np.array(vzd)<max(vzd)/2])
    # correct velocities
    simgas[vpos]     =   simgas[vpos]-[vxpos,vypos,vzpos]
    simstar[vpos]    =   simstar[vpos]-[vxpos,vypos,vzpos]
    # simdm[vpos]      =   simdm[vpos]-[vxpos,vypos,vzpos]
    print('corrections: ',vxpos,vypos,vzpos)
    # if plot:
    #     plt.figure(1)
    #     plt.title('Centering in [vx,vy,vz]')
    #     plt.plot(grid,vxd,'r')
    #     plt.plot(grid,vyd,'g')
    #     plt.plot(grid,vzd,'b')
    #     plt.plot([0]*ngrid+vxpos,vxd,'--r')
    #     plt.plot([0]*ngrid+vypos,vxd,'--g')
    #     plt.plot([0]*ngrid+vzpos,vxd,'--b')
    #     plt.xlabel('vx,vy,vz position [km/s]')
    #     plt.ylabel('accumulated SPH gas mass [M$_{\odot}$]')
    #     plt.show(block=False)

    # Calculate more precise galactic radius (for SFR surface density etc)
    # R               =   np.arange(1000)/999.*R_cut
    # dr              =   R[1]-R[0]
    # r_gal           =   np.sqrt(simstar['x'].values**2+simstar['y'].values**2)
    # mass_profile    =   []
    # for r1 in R:
    #     mass_profile         =   np.append(mass_profile,sum(simstar['m'][r_gal < r1]))
    # m_star          =   simstar['m'].values
    # R_gal           =   min(R[mass_profile > f_R_gal*sum(m_star[r_gal < R_cut])])
    # print('R_gal = ' + str.format("{0:.2f}",R_gal) + ' kpc (contains 90% of stellar mass)')
    # if plot:
    #     plt.figure(2)
    #     plt.title('Finding R_gal')
    #     plt.plot(R,mass_profile,'r')
    #     plt.plot([R_gal,R_gal],[min(mass_profile),max(mass_profile)],'b--')
    #     plt.show(block=False)

    # if plot:
    #     fig                 =   plt.figure(4)
    #     ax1                 =   fig.add_axes([0.15,0.1,0.75,0.8])
    #     ax1.set_title('Gas distribution < R_cut')
    #     ax1.plot(simgas['x'],simgas['y'],'.',ms=2)
    #     ax1.set_xlabel('x')
    #     ax1.set_ylabel('y')
    #     ax1.set_xlim([-R_cut,R_cut])
    #     ax1.set_ylim([-R_cut,R_cut])
    #     plt.show(block=False)
    # cut out galaxy < R_gal
    r               =   rad(simgas[pos],pos)
    simgas          =   simgas[r < R_gal]
    simgas          =   simgas.reset_index(drop=True)
    r               =   rad(simstar[pos],pos)
    simstar         =   simstar[r < R_gal]
    simstar         =   simstar.reset_index(drop=True)
    r               =   rad(simdm[pos],pos)
    simdm           =   simdm[r < R_gal]
    simdm           =   simdm.reset_index(drop=True)

    # Make dataframe with global properties
    M_star          =   sum(simstar['m'])
    M_gas           =   sum(simgas['m'])
    M_dm            =   sum(simdm['m'])
    SFR             =   sum(simstar['m'].values[simstar['age'].values < 100])/100e6           # Msun/yr
    SFRsd           =   SFR/(np.pi*R_gal**2.)
    # For JSL galaxies: Force SFR to that derived by JSL (including stars that went back to gas stage)
    # if galname == 's6_G0': SFR = 11.3
    # if galname == 's7_G0': SFR = 5.23
    # Zmw             =   sum(simgas['Z']*simgas['m'])/sum(simgas['m'])
    Zsfr            =   sum(simgas['Z']*simgas['SFR'])/sum(simgas['SFR'])

    # Print properties
    sep         =   ['30','10','20','40']
    sep1        =   '+%'+sep[0]+'s+%'+sep[1]+'s+%'+sep[2]+'s+%'+sep[3]+'s+'
    sep2        =   '|%'+sep[0]+'s|%'+sep[1]+'s|%'+sep[2]+'s|%'+sep[3]+'s|'
    print(sep1 % ((int(sep[0])*'-'), (int(sep[1])*'-'), (int(sep[2])*'-'), (int(sep[3])*'-')))
    print(sep2 % ('Parameter'.center(int(sep[0])), 'Value'.center(int(sep[1])), 'Name in code'.center(int(sep[2])), 'Explanation'.center(int(sep[3]))))
    print(sep1 % ((int(sep[0])*'-'), (int(sep[1])*'-'), (int(sep[2])*'-'), (int(sep[3])*'-')))
    print(sep2 % ('Stellar mass [1e9 M_sun]'.center(int(sep[0])), str.format("{0:.3f}",M_star/1e9), "prop['M_star']".center(int(sep[2])),'Mass of all stellar particles'.center(int(sep[3]))))
    print(sep2 % ('Gas mass [1e9 M_sun]'.center(int(sep[0])), str.format("{0:.3f}",M_gas/1e9), "prop['M_gas']".center(int(sep[2])),'Mass of all gas particles'.center(int(sep[3]))))
    # print(sep2 % ('DM mass [1e9 M_sun]'.center(int(sep[0])), str.format("{0:.3f}",M_DM/1e9), "prop['M_DM']".center(int(sep[2])),'Mass of all gas particles'.center(int(sep[3]))))
    print(sep2 % ('SFR [M_sun/yr]'.center(int(sep[0])), str.format("{0:.3f}",SFR), "prop['SFR']".center(int(sep[2])),'SFR averaged over past 100 Myr'.center(int(sep[3]))))
    print(sep2 % ('SFRd [M_sun/yr/kpc^2]'.center(int(sep[0])), str.format("{0:.4f}",SFRsd), "prop['SFRsd']".center(int(sep[2])),'Surface density of SFR'.center(int(sep[3]))))
    print(sep2 % ('Z [Z_sun]'.center(int(sep[0])), str.format("{0:.3f}",Zsfr), "prop['Z']".center(int(sep[2])),'Mass-weighted metallicity'.center(int(sep[3]))))
    print(sep1 % ((int(sep[0])*'-'), (int(sep[1])*'-'), (int(sep[2])*'-'), (int(sep[3])*'-')))

    # pdb.set_trace()
    return simgas,simstar,simdm

def kernel(q,type):
    if type == 'simple':
        rho         =   1.-q
        rho[q>1]    =   0
    # if type == 'cubic':
    #     rho         =   ((2-q)**3.-(1-q)**3)/(4.*np.pi)
    #     rho[q>1]    =   (2-q[q>1])**3./(4.*np.pi)
    #     rho[q>2]    =   0.
    if type == 'cubic':
        rho         =   (1-(3/2.)*q**2.+(3/4.)*q**3.)/np.pi
        rho[q>1]    =   (1/4.)*(2-q[q>1])**3./np.pi
        rho[q>2]    =   0.
    if type == 'quintic':
        rho         =   ((3-q)**5.-6.*(2-q)**5+15.*(1-q)**5)/(120.*np.pi)
        rho[q>1]    =   ((3-q[q>1])**5.-6.*(2-q[q>1])**5)/(120.*np.pi)
        rho[q>2]    =   (3-q[q>2])**5./(120.*np.pi)
        rho[q>3]    =   0.
    return rho

def contamination_check(caesarfile,snapshot,halonum):

    obj = caesar.load(caesarfile)
    ds = yt.load(snapshot)

    ad = ds.all_data()

    halo_center = obj.halos[halonum].pos.in_units('code_length')
    halo_radius = obj.halos[halonum].radii['virial'].in_units('code_length')
    #get lowres particle positions
    lowres_posx = []
    lowres_posy = []
    lowres_posz = []

    print '----------------------------------------'
    print 'assigning the low resolution particles'
    if ('PartType2', 'particle_position_x') in ds.derived_field_list:
        print 'Found Part Type 2 particles in your datset'
        lowres_posx.append(ad[('PartType2', 'particle_position_x')].in_units('code_length'))
        lowres_posy.append(ad[('PartType2', 'particle_position_y')].in_units('code_length'))
        lowres_posz.append(ad[('PartType2', 'particle_position_z')].in_units('code_length'))

    if ('PartType3', 'particle_position_x') in ds.derived_field_list:
        print 'Found Part Type 3 particles in your dataset'
        lowres_posx.append(ad[('PartType3', 'particle_position_x')].in_units('code_length'))
        lowres_posy.append(ad[('PartType3', 'particle_position_y')].in_units('code_length'))
        lowres_posz.append(ad[('PartType3', 'particle_position_z')].in_units('code_length'))

    if ('PartType5', 'particle_position_x') in ds.derived_field_list:
        print 'Found Part Type 5 particles in your dataset'
        lowres_posx.append(ad[('PartType5', 'particle_position_x')].in_units('code_length'))
        lowres_posy.append(ad[('PartType5', 'particle_position_y')].in_units('code_length'))
        lowres_posz.append(ad[('PartType5', 'particle_position_z')].in_units('code_length'))



    #get the number of lowres particle types
    n_lowres_particle_types = len(lowres_posx)

    #find number of contaminating partlces
    contaminating_particles =  []
    contaminating_particles_3virialradii = []

    for i in range(n_lowres_particle_types):
        particles_x = lowres_posx[i]
        particles_y = lowres_posy[i]
        particles_z = lowres_posz[i]

        contaminating_particles.append(np.where( (particles_x < halo_center[0]+halo_radius) & (particles_x > halo_center[0]-halo_radius) & \
                                                     (particles_y < halo_center[1]+halo_radius) & (particles_y > halo_center[1]-halo_radius) & \
                                                     (particles_z < halo_center[2]+halo_radius) & (particles_z > halo_center[2]-halo_radius)))

        contaminating_particles_3virialradii.append(np.where( (particles_x < halo_center[0]+(3.*halo_radius)) & (particles_x > halo_center[0]-(3.*halo_radius)) & \
                                                                  (particles_y < halo_center[1]+(3.*halo_radius)) & (particles_y > halo_center[1]-(3.*halo_radius)) & \
                                                                  (particles_z < halo_center[2]+(3.*halo_radius)) & (particles_z > halo_center[2]-(3.*halo_radius))))

    #print stats
    print '----------------------------------------'
    if len(contaminating_particles) == 1:
        print 'congratulations! you have 0 contaminating_particles in your main halo'
    if len(contaminating_particles) > 1:
        fraction_of_contaminating_particles = len(contaminating_particles)/len(obj.halos[halonum].glist)
        print 'fraction of contaminating particles in main halo is: %e'%fraction_of_contaminating_particles

    if len(contaminating_particles_3virialradii) == 1:
        print 'congratulations! you have 0 contaminating_particles within 3x the virial radius of your main halo'
    if len(contaminating_particles_3virialradii) > 1:
        fraction_of_contaminating_particles_3virialradii = len(contaminating_particles_3virialradii)/len(obj.halos[halonum].glist)
        print 'fraction of contaminating particles within 3x the virial radius of the main halo is: %e'%fraction_of_contaminating_particles_3virialradii


    print '----------------------------------------'
    print '----------------------------------------'
