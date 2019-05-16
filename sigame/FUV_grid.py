###     Module: FUV_grid.py of SIGAME         		###

import numpy as np
import pandas as pd
import pickle
import pdb
import scipy
from scipy.interpolate import InterpolatedUnivariateSpline,interp1d,interp2d
import time
import multiprocessing as mp
import subprocess as sub
import aux as aux
import matplotlib.pyplot as plt

# From SIGAME submodules:
# import sigame.plot as plot

params                      =   np.load('sigame/temp/params.npy').item()
for key,val in params.items():
    exec(key + '=val')

def grid_radiation():
    print('** Get FUV flux from stellar population **')

    # Ages [Myr]
    if z1 == 'z6': Ages        =   10.**np.array([-0.5,0,0.5,1,1.5,2,2.5,3])
    if z1 == 'z2': Ages        =   10.**np.array([1.8,2.0,2.2,2.4,2.6,2.8,3,3.2])
    if z1 == 'z0': Ages        =   10.**np.array([0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0])
    
    # Metallicities
    if z1 == 'z6': Zs          =   10.**np.array([-1.4,-1.0,-0.6,-0.2])
    if z1 == 'z2': Zs          =   10.**np.array([-1.5,-1.0,-0.5,0.0])
    if z1 == 'z0': Zs          =   10.**np.linspace(-2,0.2,4)
    Z_sb99      =   ['51','52','53','54','55'] # Z actually available in starburst99
    # f           =   interp1d([0.0001,0.002,0.008,0.014,0.040],[0,0,1,2,3,4])       
    # Z1          =   np.zeros(len(Zs))
    # i           =   0
    # for Z in Zs:
    #     Z1[i]       =   Z_sb99[int(round(f(0.02*Z)))]
    #     i           +=  1

    # Use all metallicities available in starburst99:
    # 51=0.001;  52=0.002; 53=0.008; 54=0.014; 55=0.040
    Z1          =   Z_sb99
    Zs          =   np.array([0.0001,0.002,0.008,0.014,0.040])/0.0134 # 0.0134 from R. Dave

    nmodels     =   len(Ages)*1.*len(Zs)
    # pdb.set_trace()

    # Save grid axes
    FUVgrid     =   {'Ages':Ages,'Zs':Zs}
    pickle.dump(FUVgrid,open(d_t+'FUV/FUVgrid_'+z1+'_noneb','wb'))

    foo         =   raw_input('Run starburst99? [default: n] ... ')
    if foo == '': foo =   'n'
    if foo == 'y':
        i           =   0
        for i1 in range(0,len(Ages)):
            Age         =   Ages[i1]
            for i2 in range(0,len(Zs)):
                name                    =   'sb_'+str(i)
                print(name)
                # Add line in the beginning of script and change age, mass and metallicity
                script_in               =   open(d_sb+'template.input','r')     # template file
                script_out              =   open(d_sb+name+'.input','w')
                nextline                =   -1
                for line in script_in:

                    if line.find('<model>') >= 0:
                        line = line.replace('<model>', name)                # log [cm]

                    if line.find('<metals>') >= 0:
                        line = line.replace('<metals>', str(int(Z1[i2])))

                    if line.find('<last_grid_point>') >= 0:
                        line = line.replace('<last_grid_point>', str(Age*1.1))

                    if line.find('<time_step>') >= 0:
                        line = line.replace('<time_step>', str(Age/2.))

                    script_out.write(line)

                script_in.close()
                script_out.close()
                # Edit run file for starburst99 scripts and run them
                name                    =   'sb_'+str(i)
                go_in                   =   open(d_sb+'go_galaxy_template','r')
                go_out                  =   open(d_sb+'go_galaxy_1','w')
                for line in go_in:

                    if line.find('<name>') >= 0:
                        line = line.replace('<name>', name)

                    go_out.write(line)

                go_in.close()
                go_out.close()
                # And run!!
                pro                     =   sub.Popen(['./go_galaxy_1'],cwd=d_sb,stdout=sub.PIPE)
                text                    =   u'Done with stellar population # '+str(i1)
                stdout,stderr           =   pro.communicate()   # wait until starburst is done
                i                       +=  1

    i           =  0

    foo         =   raw_input('Save FUV grid? [default: n] ... ')
    if foo == '': foo =   'n'
    if foo == 'y':
        # Make a dataframe with results
        FUV         =   pd.DataFrame({'Age':np.zeros(int(nmodels)),'Z':np.zeros(int(nmodels)),'L_FUV':np.zeros(int(nmodels))})
        FUV         =   FUV[['Age','Z','L_FUV']] # ordering dataframe
        i           =   0
        for i1 in range(0,len(Ages)):
            Age         =   Ages[i1]
            for i2 in range(0,len(Zs)):
                name                    =   'sb_'+str(i)
                Z                       =   Zs[i2]
                FUV['Age'][i]           =   Age
                FUV['Z'][i]             =   Z
                # Calculate luminosity of this population:
                columns                 =   ['time','wavelength','ltot','lstellar','lnebular']
                spec                    =   pd.read_table(d_sb+name+'.spectrum1',names=columns,skiprows=6,sep='\s*',engine='python')
                x                       =   spec['wavelength'][spec['time']==spec['time'].max()].values  # AA
                y                       =   spec['lstellar'][spec['time']==spec['time'].max()].values       # ergs/s/AA
                int_range               =   clight*hplanck/np.array([6,13.6])*1e10            # eV -> AA
                L_FUV                   =   scipy.integrate.simps(10**y[(x>int_range[1]) & (x<int_range[0])],x[(x>int_range[1]) & (x<int_range[0])])
                print(i,L_FUV,' ergs/s')
                # Write lum into DataFrame
                FUV['L_FUV'][i]         =   L_FUV
                # Convert to erg/s/cm^2, for a distance of 1 kpc
                distance                =   kpc2cm                    # cm
                flux_FUV                =   3*L_FUV/(4*np.pi*distance**2)
                Habing                  =   2.74e-3                         # LAMDA [erg/s/cm^2]
                print(flux_FUV/Habing,' Habing ')
                i                       +=  1
        # Save results in txt file
        np.savetxt(d_t+r'FUV/FUVtable_'+z1+'_noneb'+'.txt',FUV.values,fmt='%-14.4f\t%-14.4f\t%-14.4e',\
            header='Age [Myr]\tZ \t\tL_FUV [ergs/s]')
        FUV.to_pickle(d_t+'FUV/FUV_'+z1+'_noneb')
    pdb.set_trace()

def amb_FUV(plotting=True):
    print('Calculate background FUV spectrum from simulation')
    # Using background spectra from gadget: http://galaxies.northwestern.edu/uvb/
    columns     =   ['ryd','Jnu']
    spec        =   pd.read_table(d_t+'fg_uvb_dec11/fg_uvb_dec11_z_'+str(int(zred))+'.0.dat',names=columns,skiprows=2,sep=r'\s*',engine='python')
    freq        =   spec['ryd'].values*Ryd/hplanck             # [Hz]
    Efreq       =   spec['ryd'].values*Ryd                     # [eV]
    freq_HI     =   spec['ryd'].values*Ryd/13.6                 # [v_HI]
    Jfreq       =   spec['Jnu'].values                         # [1e-21 erg s^-1 cm^-2 Hz^-1 sr^-1]
    #Habing unit
    Habing      =   1.6e-3                              # [ergs/s/cm^2]

    # Extraplate UV spectrum to E = 6eV and 13.5eV, and extract Efreq=6-13.6eV region. 
    s            =   InterpolatedUnivariateSpline(Efreq, Jfreq, k=1)
    freq_Habing  =   freq[(6 <= Efreq) & (Efreq <= Ryd)]
    freq_Habing  =   np.append(6./hplanck,freq_Habing)
    Jfreq_Habing =   spec['Jnu'][(6 < Efreq) & (Efreq <= Ryd)]
    Jfreq_Habing =   np.append(s(6),Jfreq_Habing)
    # Integrate extracted UV spectrum (Efreq=6-13.6eV) and convert to Habing
    # units
    FUV_amb     =   scipy.integrate.simps(Jfreq_Habing,freq_Habing)*4*np.pi*1e-21/Habing     # [Habing]
    print('Ambient FUV at z = '+str(zred)+' [Habing] : '+str(FUV_amb))
    FUV_amb     =   FUV_amb*Habing/FUV_ISM  # [local ISM unit]
    print('Ambient FUV at z = '+str(zred)+' [local ISM normalized] : '+str(FUV_amb))

    # Integrate over given energy range
    I           =   scipy.integrate.simps(Jfreq,freq)*4*np.pi*1e-21     # [ergs/s/cm^2]
    print('Log intensity over known range is: '+str(np.log10(I))+' ergs/s/cm^2')

    # plot metagalactic UV spectrum from 6 to 13.6eV
    if plotting:
        plt.close('all')        # close all window
        plot.simple_plot(fig=0,  x1=spec['ryd'], y1=Jfreq*1e-21*4*np.pi, ls1='-',  col1='b',\
        xlog='y', ylog='y',xr=[1e-1,1e4],yr=[1e-32,1e-18],\
        xlab='Energy [Ryd]',ylab='Flux from simulation [erg s^-1 cm^-2 sr^-1]',\
        figname='plots/FUV/UV_bkg_z6',figtype='png')
        plot.simple_plot(fig=1,  x1=freq, y1=Jfreq*1e-21*4*np.pi, ls1='-',  col1='b',\
        xlog='y', ylog='y',xr=[1e15,2e19],yr=[1e-32,1e-18],\
        xlab='Frequency [Hz]',ylab='Flux from simulation [erg s^-1 cm^-2 sr^-1]',\
        figname='plots/FUV/UV_freq_bkg_z6',figtype='png')
        plt.show(block=False)
        # plot.simple_plot(fig=1, x1=freq_HI, y1=Jfreq*1e-21*4*np.pi, ls1='-',  col1='b',\
        # xlog='y', ylog='y',\
        # xlab=r'Frequency [$\nu_{\mathrm{HI}}$]',ylab='Flux [erg s^-1 cm^-2 sr^-1]',\
        # figname='plots/FUV/UV_bkg_z6',figtype='png')
        # plt.show(block=False)

    # store strength of metagalactic UV field
    pickle.dump(FUV_amb,open(d_t+'FUV/FUV_amb'+z1+'','wb'))

    pdb.set_trace()


