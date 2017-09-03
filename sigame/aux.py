"""
###     Submodule: aux.py of SIGAME              		###
"""

import numpy as np
import pandas as pd
import pdb as pdb
from scipy import optimize
import scipy.stats as stats
import scipy.integrate as integrate
import os
import matplotlib.pyplot as plt
import linecache as lc
import re as re
import sys as sys
import cPickle

params                      =   np.load('sigame/temp/params.npy').item()
for key,val in params.items():
    exec(key + '=val')

def global_properties(igalnum=0,nGal=1,data_format='text'):
    # Convert data to pandas dataframe if not already and save global properties in a separate file:

	ext               		=   ''
	if ext_DIFFUSE == '_FUV': ext = '_FUV'
	models_path 			=	'sigame/temp/global_results/'+z1+'_'+str(nGal)+'gals'+ext_DENSE+ext_DIFFUSE+ext

	# galname,zred 		=	galnames[igalnum],zreds[igalnum]


	if os.path.exists(models_path):
		models                 	=   pd.read_pickle(models_path)
		galname 				=	models['galnames'][igalnum]
		zred 					=	models['zreds'][igalnum]
		print('Loading previous results and continuing...')
		# if igalnum == 0: models['M_dm'] 			=	np.zeros(nGal)

	else:
		print('Cannot load previous results')
		print('Starting new dataframe')
		# To be saved for each galaxy:
		targets 				=	['SFR','Zsfr','R_gal','M_star','M_dm','SFRsd',\
									'M_gas','M_GMC','M_DNG','M_DIG','M_dense','M_tot',\
									'M_H2_GMC','M_HI_GMC','M_HII_GMC','M_C_GMC','M_CII_GMC','M_CIII_GMC','M_CO_GMC','M_dust_GMC',\
									'M_H2_DNG','M_HI_DNG','M_HII_DNG','M_C_DNG','M_CII_DNG','M_CIII_DNG','M_CO_DNG','M_dust_DNG',\
									'M_H2_DIG','M_HI_DIG','M_HII_DIG','M_C_DIG','M_CII_DIG','M_CIII_DIG','M_CO_DIG','M_dust_DIG']
		for line in lines: targets.append('L_'+line+'_GMC')
		for line in lines: targets.append('L_'+line+'_DNG')
		for line in lines: targets.append('L_'+line+'_DIG')

		# Order galaxies by the original snapshot order
		models                  =   pd.DataFrame({'galnames':galnames})
		models['zreds'] 		=	zreds
		for target in targets:
			models[target] 			=	np.zeros(nGal)
		galname 				=	galnames[igalnum]
		zred 					=	zreds[igalnum]

	if data_format == 'dataframe':
		# Overwrite global properties
		simgas          		=   pd.read_pickle(d_sim+'z'+'{:.2f}'.format(zred)+'_'+galname+'_sim0.gas')
		simstar          		=   pd.read_pickle(d_sim+'z'+'{:.2f}'.format(zred)+'_'+galname+'_sim0.star')
		simdm           		=   pd.read_pickle(d_sim+'z'+'{:.2f}'.format(zred)+'_'+galname+'_sim0.dm')
		r_gas                   =   np.sqrt(simgas['x'].values**2.+simgas['y'].values**2.+simgas['z'].values**2.)
		r_star                  =   np.sqrt(simstar['x'].values**2.+simstar['y'].values**2.+simstar['z'].values**2.)
		age                     =   simstar['age'].values[r_star < np.max(r_gas)]
		m_stars                 =   simstar['m'].values[r_star < np.max(r_gas)]
		models.set_value(igalnum,'galnames',galname)
		models.set_value(igalnum,'zreds',zred)
		models.set_value(igalnum,'R_gal',np.max(r_gas))
		models.set_value(igalnum,'M_star',np.sum(m_stars))
		models.set_value(igalnum,'M_gas',np.sum(simgas['m'].values))
		models.set_value(igalnum,'M_dm',np.sum(simdm['m'].values))
		models.set_value(igalnum,'M_tot',np.sum(m_stars)+np.sum(simgas['m'].values)+np.sum(simdm['m'].values))
		models.set_value(igalnum,'M_dense',np.sum(simgas['m'].values*simgas['f_HI'].values*simgas['f_H2'].values))
		models.set_value(igalnum,'SFR',np.sum(m_stars[age < 100.])/100e6)
		models.set_value(igalnum,'SFRsd',models['SFR'][igalnum]/(np.pi*np.max(r_gas)**2.))
		models.set_value(igalnum,'Zsfr',np.sum(simgas['Z'].values*simgas['SFR'].values)/np.sum(simgas['SFR'].values))
	if data_format == 'text':
		simgas              	=   pd.read_table('sigame/temp/raw_SPH//z'+'{:.2f}'.format(zreds[igalnum])+'_'+galnames[igalnum]+'_gas.txt',sep='\t',skiprows=1,\
		                                names=['x','y','z','vx','vy','vz','m','Tk','Z','h','SFR','f_ion','f_H2'],index_col=False)
		simstar                	=   pd.read_table('sigame/temp/raw_SPH//z'+'{:.2f}'.format(zreds[igalnum])+'_'+galnames[igalnum]+'_stars.txt',sep='\t',skiprows=1,\
		                                names=['x','y','z','vx','vy','vz','age','m','Z'],index_col=False)
		simgas.to_pickle('sigame/temp/SPH/z'+'{:.2f}'.format(zreds[igalnum])+'_'+galnames[igalnum]+'_SPH0.gas')
		simstar.to_pickle('sigame/temp/SPH/z'+'{:.2f}'.format(zreds[igalnum])+'_'+galnames[igalnum]+'_SPH0.star')
		r_gas                   =   np.sqrt(simgas['x'].values**2.+simgas['y'].values**2.+simgas['z'].values**2.)
		r_star                  =   np.sqrt(simstar['x'].values**2.+simstar['y'].values**2.+simstar['z'].values**2.)
		age                     =   simstar['age'].values[r_star < R_gal[igalnum]]
		m_stars                 =   simstar['m'].values[r_star < R_gal[igalnum]]
		models.set_value(i,'zred',zreds[igalnum])
		models.set_value(i,'R_gal',R_gals[igalnum])
		models.set_value(i,'M_star',np.sum(m_stars))
		models.set_value(i,'M_gas',np.sum(simgas['m'].values[r_gas < R_gals[igalnum]]))
		models.set_value(i,'SFR',np.sum(m_stars[age < 100.])/100e6)
		models.set_value(i,'SFRsd',models['SFR'][igalnum]/(4.*np.pi*models['R_gal'][igalnum]**2.))
		models.set_value(i,'Zsfr',np.sum(simgas['Z'].values*simgas['m'].values)/np.sum(simgas['m'].values))

	# Print these properties
	print('+%20s+%20s+%15s+' % ((20*'-'), (20*'-'), (15*'-')))
	print('|%20s|%20s|%15s|' % ('Property'.center(20), 'Value'.center(20), 'Name in code'.center(15)))
	print('+%20s+%20s+%15s+' % ((20*'-'), (20*'-'), (15*'-')))
	print('|%20s|%20s|%15s|' % ('Redshift'.center(20), '{:.3f}'.format(zreds[igalnum]), 'zred'.center(15)))
	print('|%20s|%20s|%15s|' % ('Radius'.center(20), '{:.3f}'.format(np.max(r_gas)), 'R_gal'.center(15)))
	print('|%20s|%20s|%15s|' % ('Stellar mass'.center(20), '{:.3e}'.format(models['M_star'][igalnum]), 'M_star'.center(15)))
	print('|%20s|%20s|%15s|' % ('ISM mass'.center(20), '{:.3e}'.format(models['M_gas'][igalnum]), 'M_gas'.center(15)))
	print('|%20s|%20s|%15s|' % ('DM mass'.center(20), '{:.3e}'.format(models['M_dm'][igalnum]), 'M_dm'.center(15)))
	print('|%20s|%20s|%15s|' % ('Dense gas mass'.center(20), '{:.3e}'.format(models['M_dense'][igalnum]), 'M_dense'.center(15)))
	print('|%20s|%20s|%15s|' % ('SFR'.center(20), '{:.3f}'.format(models['SFR'][igalnum]), 'SFR'.center(15)))
	print('|%20s|%20s|%15s|' % ('SFR surface density'.center(20), '{:.4f}'.format(models['SFRsd'][igalnum]), 'SFRsd'.center(15)))
	print('|%20s|%20s|%15s|' % ('SFR-weighted Z'.center(20), '{:.4f}'.format(models['Zsfr'][igalnum]), 'Zsfr'.center(15)))
	print('+%20s+%20s+%15s+' % ((20*'-'), (20*'-'), (15*'-')))
	return models

def find_model_results(**kwargs):
	'''
	Load model results, which is returned as dataframe
	'''

	try:
		filename 				=	global_save_file
		if kwargs.has_key('filename'): filename = kwargs['filename']
		models                  =   pd.read_pickle(filename)
		for key in models.keys():
			exec(key + '=models[key].values')
	except IOError:
		print('\nNo file with results found, did you run SIGAME? \n'+\
		    '(searched for '+filename+')\n')
		sys.exit()

	return models

def add_to_model_results(new_results):
	'''
	Add to model results and save
	'''

	models 				=   find_model_results(filename=global_save_file)

	for key,val in new_results.items():
		models[key] 		=	new_results[key]

	save_model_results(models)

def save_model_results(results):

	cPickle.dump(results,open(global_save_file,'wb'))

def rad(foo,labels):
    # Calculate distance from [0,0,0] in 3D for DataFrames!
    if len(labels)==3: return np.sqrt(foo[labels[0]]**2+foo[labels[1]]**2+foo[labels[2]]**2)
    if len(labels)==2: return np.sqrt(foo[labels[0]]**2+foo[labels[1]]**2)

def find_nearest(array,value,find='value'):
    idx = (np.abs(array-value)).argmin()
    if find == 'value': return array[idx]
    if find == 'index': return idx

def find_nearest_index(array,value):
	"""
	finds the index of the closest element
	of an array to a specific value

	Arguments
	---------
	array: np.array
	value: float/int
	"""
	return (np.abs(array-value)).argmin()

def save_galaxy_as_txt(galnames=galnames):
	print('Save SPH information for stars and gas in text format')

	galnum       	=  	raw_input('Which galaxy number? [default:0]')
	galname 		=	galnames[galnum]

	simgas          =   pd.read_pickle('sigame/temp/SPH/z'+str(int(zred))+'_'+galname+'_SPH0.gas')
	simstar         =   pd.read_pickle('sigame/temp/SPH/z'+str(int(zred))+'_'+galname+'_SPH0.star')

	simgas 			=	simgas['x','y','z','vx','vy','vz','Z','nH','Tk',\
							'h','f_HI','f_H2','m']
	# np.savetxt(r'SPH/galaxies/G'+str(int(galnum))+'_gas.txt',\
	# 	simgas.values,fmt=,header=)


	simstar 		=	simstar['x','y','z','vx','vy','vz','Z','nH','Tk',\
					'h','f_HI','f_H2','m']

def luminosity_distance(zred):
	"""luminosity distance

	args
	----
	zred: redshift

	returns
	-------
	luminosity distance - scalar - Mpc
	"""

	def I(a): # function to integrate
		return 1/(a * np.sqrt( (omega_r/a**2) + (omega_m/a) + (omega_lambda * a**2) )) # num
	LL = 1/(1+zred) # num
	UL = 1 # num
	integral = integrate.quad(lambda a: I(a), LL, UL)
	d_p = (clight/1000./(100.*hubble)) * integral[0]
	return d_p * (1+zred)

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
	""" Converts solar luminosity/(km/s) to jansky/(km/s)

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

def Jy2solLum(Jykms, nu_rest, zred, d_L):
	"""returns integrated Jy to solar luminosity

	args
	----
	Jykms: scalar
	total in Jy*km/s (W/Hz/m^2*km/s)

	nu_rest: scalar
	rest frequency (GHz)

	zred: scalar
	redshift (num)

	d_L: scalar
	luminosity distance (Mpc)

	returns
	-------
	scalar
	solar luminosity
	"""
	print("'Jy2solLum' still needs verification")
	return 1.04e-3 * Jykms * (nu_rest/(1+zred)) * d_L**2

def disp2FWHM(sig):
    return 2*np.sqrt(2*np.log(2)) * sig

#===============================================================================
""" For line profiles mainly """
#-------------------------------------------------------------------------------

def gauss_peak(L,sig):
	"""peak of gausian profile

	args
	----
	L: scalar - luminosity of cloud
	sig: scalar - velocity dispersion of cloud

	returns
	-------
	scalar
	"""
	return L/np.sqrt(2*np.pi*sig**2)

def single_cloud_profile(VEL,L,sig,v0):
    """ make a single gauss profile for a single cloud

    Parameters
    ----------
    VEL:    velocity array
    L:      total luminosity of cloud
    sig:    velocity dispersion of cloud
    v0:     average velocity of cloud

    Returns
    LP:     line profile array with same dimension as VEL array
    """

    peak    =   gauss_peak(L,sig)
    LP      =   peak * np.exp( -(VEL-v0)**2 / (2*sig**2) )

    return LP

def single_cloud_weight(v_beg,v_end,N,L,sig,v0):
    """ A weight function to find what fraction of
    the total line profile resides in a velocity bin.

    1) make velocity array for line profile
    2) calculate line profile within velocity bin
    3) find weight:
        a) find amount of profile in velocity bin
        b) find weight of velocity bin
        c) store weight of velocity bin in weight array

    Parameters
    ----------
    v_beg:      v0 of velocity bin
    v_end:      vf of velocity bin
    N:          length of line profile array
    L:          luminosity of cloud
    sig:        velocity dispersion of cloud
    v0:         average velocity of cloud

    Returns
    -------
    weight
    """

    # 1) make velocity array for line profile
    VEL_LP      =   np.linspace(v_beg,v_end,N)
    dv          =   (v_end-v_beg)/N

    # 2) calculate line profile within velocity bin
    LP          =   single_cloud_profile(VEL_LP,L,sig,v0)

    # 3) find weight:
    weight      =   dv * np.sum(LP) / L

    return weight

def sum_cloud_line_profs(VEL,L_ds,sig_ds,v0_ds,L_df,sig_df,v0_df,zred,d_L,nu_rest,galname,path,element='CII',plotting=False,figsize=(12,15),fs=20,fig_path='plots/line_emission/line_profiles/detailed_profiles/'):
	"""line profiles for fine emission lines for model galaxy

	args
	----
	VEL: array - velocity array along - axis=0
	L_ds: array - luminosites of all dense clouds - axis=1
	sig_ds: array - velocity dispersion of all dense clouds - axis=1
	v0_ds: array - average velocity of all dense clouds - axis=1
	L_df: array - luminosites of all diffuse clouds - axis=1
	sig_df: array - velocity dispersion of all diffuse clouds - axis=1
	v0_df: array - average velocity of all diffuse clouds - axis=1
	zred: redshift
	nu_rest: rest wavelength
	element: name of element
	galname: string - name of galaxy
	path: string - path to numpy folder
	** element: string - name of fine emission line, default='CII'
	** plotting: boolean - whether or not to plot, default=False
	** figsize: size of plotted figure, default=(12,15)
	** fs: fontsize, default=20
	** fig_path: string - path to save figure, default='../plots/line_emission/line_profiles/detailed_profiles/'

	returns
	-------
	single summed line profile for element from GMC and diffuse clouds
	"""

	assert len(L_ds) == len(sig_ds) == len(v0_ds), "L_ds, sig_ds, and v0_ds must be same length!"
	assert len(L_df) == len(sig_df) == len(v0_df), "L_df, sig_df, and v0_df must be same length!"

	dV = VEL[1]-VEL[0] # velocity resolution (bin size)

	print("beginning array creation")
	PEAKS_dense = gauss_peak(L_ds,sig_ds)
	PEAKS_diffuse = gauss_peak(L_df,sig_df)
	DENSE = np.zeros([len(L_ds),len(VEL)])
	DIFFUSE = np.zeros([len(L_df),len(VEL)])

	print("calculating profiles for dense clouds into massive array")
	for i in range(0,len(DENSE)):
		DENSE[i,:] = PEAKS_dense[i] * np.exp( -(VEL-v0_ds[i])**2 / (2*sig_ds[i]**2) )

	print("calculating profiles for diffuse clouds into massive array")
	for i in range(0,len(DIFFUSE)):
		DIFFUSE[i,:] = PEAKS_diffuse[i] * np.exp( -(VEL-v0_df[i])**2 / (2*sig_df[i]**2) )

	print("summing massive arrays")
	DENSE_sum = DENSE.sum(axis=0)
	DIFFUSE_sum = DIFFUSE.sum(axis=0)
	TOTAL = np.vstack((DENSE_sum,DIFFUSE_sum)).sum(axis=0)

	print("converting summed arrays to mJy")
	TOTAL = solLum2Jy(TOTAL, zred, d_L, nu_rest) * 1000
	DENSE_sum = solLum2Jy(DENSE_sum, zred, d_L, nu_rest) * 1000

	DIFFUSE_sum = solLum2Jy(DIFFUSE_sum, zred, d_L, nu_rest) * 1000

	print('checkpoint: converting back to Lsun/km/s:')
	L_DENSE = Jy2solLum(DENSE_sum/1000.*dV, nu_rest, zred, d_L)
	print("%s Lsun" % np.sum(L_DENSE))

	print("saving summed array(s)")
	np.save(path+galname+'_%s_total.npy' % element, TOTAL)
	np.save(path+galname+'_%s_dense.npy' % element, DENSE_sum)
	np.save(path+galname+'_%s_diffuse.npy' % element, DIFFUSE_sum)

	if plotting == True:

		print("plotting=True: starting plotting sequence")

		print("converting 'DENSE' and 'DENSE_sum' arrays to mJy")
		print("%s Lsun" % np.sum(DENSE*dV))
		DENSE = solLum2Jy(DENSE, zred, d_L, nu_rest) * 1000

		print('checkpoint: converting back to Lsun/km/s:')
		L_DENSE = Jy2solLum(DENSE/1000.*dV, nu_rest, zred, d_L)
		print("%s Lsun" % np.sum(L_DENSE))

		print("converting 'DIFFUSE' and 'DIFFUSE_sum' arrays to mJy")
		DIFFUSE *= solLum2Jy(DIFFUSE, zred, d_L, nu_rest) * 1000

		print("now plotting arrays")
		plt.close('all')
		fig = plt.figure(figsize=figsize)

		base_title = "$z$ = %s: " % zred

		print("working on [%s] single dense plot..." % element)
		ax1 = plt.subplot(321) # dense single
		ax1.set_title(base_title+"Single [%s] GMCs" % element, fontsize=fs+2)
		for row in DENSE:
			ax1.plot(VEL,row,'-',color='b')

		print("working on [%s] combined dense plot..." % element)
		ax2 = plt.subplot(322) # dense combined
		ax2.set_title(base_title+"Total [%s] from GMCs" % element, fontsize=fs+2)
		ax2.plot(VEL,DENSE_sum,'-',color='b')

		print("working on [%s] single diffuse plot..." % element)
		ax3 = plt.subplot(323) # diffuse single
		ax3.set_title(base_title+"Single [%s] Diffuse Clouds" % element, fontsize=fs+2)
		for row in DIFFUSE:
			ax3.plot(VEL,row,'-',color='g')

		print("working on [%s] combined diffuse plot..." % element)
		ax4 = plt.subplot(324) # diffuse combined
		ax4.set_title(base_title+"Total [%s] from Diffuse Clouds" % element, fontsize=fs+2)
		ax4.plot(VEL,DIFFUSE_sum,'-',color='g')

		print("working on [%s] total diffuse and combined plot..." % element)
		ax5 = plt.subplot(313) # total
		ax5.set_title(base_title+"Total [%s] of Galaxy" % element, fontsize=fs+2)
		ax5.plot(VEL,TOTAL,'-',color='r')

		ax = [ax1,ax2,ax3,ax4,ax5]
		for a in ax:
			a.set_xlabel('km/s', fontsize=fs)
			a.set_ylabel("mJy", fontsize=fs)
			a.set_xlim([min(VEL),max(VEL)])

		plt.tight_layout()

		# if saveA == True:
		fig_name = galname
		title = fig_path+fig_name+'.png'
		fig.savefig(title)
		plt.close()

	return

def read_cloudy_mass_fractions(model_output):
	output        			=   open(model_output,'r')
	lc.clearcache()
	for i,line in enumerate(output):
	    if line.find('Gas Phase Chemical Composition') >= 0:
			# Save the next three lines:
			raw 				=	re.sub('\n','',lc.getline(model_output,i+2))
			raw 				+=	re.sub('\n','',lc.getline(model_output,i+3))
			raw 				+=	re.sub('\n','',lc.getline(model_output,i+4))
	raw 					=	raw.replace(':',' ')
	raw 					=	re.split(r'\s+',raw)
	mf 						=	np.array([])
	elements 				=	['He','C','N','O','Ne','Mg','Si','S','Ca','Fe']
	i 						=	0
	for raw1 in raw:
		raw1 				=	raw1.replace(':','')
		if raw1 in elements:
			raw2 				=	raw[i+1]
			raw2 				=	raw2.replace(':','')
			try:
				mf 					=	np.append(mf,float(raw2))
			except:
				mf 					=	np.append(mf,float(raw[i+2]))
		i 						+= 	1

	return(10.**mf)

def lower_res(lp_old,v_old,v_new):
	lp_new 	=	np.zeros_like(v_new)
	n 		= 	np.zeros_like(v_new)

	for i,v in enumerate(v_old):
		i_new 				= 	find_nearest_index(v_new,v)
		lp_new[i_new]		=	lp_new[i_new] + lp_old[i]
		n[i_new] 			+=	1
	lp_new /= n[i_new]
	return lp_new

def degrade_arrays(V_old,ds_old,df_old,sum_old,dv_new=17.5):
	"""
	takes 'high' resolution line profiles and
	degrades them to 'low' resolution line profiles.
	Each (x_old,y_old) value is sent to the nearest
	matching lower resolution bin. The values in
	the new bins are summed up and divided by the
	number of old values that were sent to the new bin.

	Aguments
	--------
	dv_new: new velocity bin size - float or int
	v_old: old velocity array
	ds_old: old dense profile array
	df_old: old diffuse profile array
	sum_old: old total profile array

	Returns
	-------
	new degraded arrays - not saved
	"""

	# create new VEL array
	v_min,v_max 	=	min(V_old),max(V_old)
	V_new 			=	np.arange(v_min,v_max+dv_new,dv_new)

	# create lower resolution profile arrays
	ds_new 	=	lower_res(ds_old,V_old,V_new)
	df_new	= 	lower_res(df_old,V_old,V_new)
	sum_new	=	lower_res(sum_old,V_old,V_new)

	# pdb.set_trace()

	return V_new, ds_new, df_new, sum_new

def degrade_single(V_old,V_new,profile_old):
	""" degrades single higher resolution profile to lower resolution

	Arguments
	---------
	V_old: 			old velocity array
	V_new:			new velocity array
	profile_old:	higher resolution profile

	Returns
	-------
	array: 			lower resolution profile
	"""

	profile_new 	= 	lower_res(profile_old,V_old,V_new)

	return profile_new

def gauss_fit(X,Y,X_fit,num_profiles=1):
    """ find the gaussian fit for total galaxy profile

	Arguments
	---------
	X: 		data along x-axis - array
	Y: 		data along y_axis -	array
	X_fit: 	new velocity array
    num_profiles:   ** number of allowed profiles to fit. takes either '1' or '2'. Default = '1'

	Returns
	-------
	Y_fit:	fitted curve to X,Y data, plotted with X_fit - array
	FWHM:	Full Width at Half Max
	"""

    if num_profiles == 1:
        # fitting function
        def fitfunc(x,p):
            return p[0] * np.exp( -(x-p[1])**2 / (2*p[2]**2) )
        def errfunc(p,x,y):
            return y - fitfunc(x,p)

        # make rough fitting params
    	ymax 			= 	np.max(Y)
    	x0 				= 	0
    	sig 			= 	200
    	p_rough 		= 	[ymax , x0 , sig]

        # optimize parameters
        qout,success 	= 	optimize.leastsq(errfunc,p_rough,args=(X,Y),maxfev=5000)
        if success == False:
            print("galaxy fail!!")
            return X_fit,np.zeros_like(X_fit),0
        else:
            # find Y_fit and FWHM
            Y_fit 	=	fitfunc(X_fit,qout)
            FWHM    =   disp2FWHM(qout[2])
            return Y_fit,FWHM

    elif num_profiles == 2:
        # fitting function
        def fitfunc(x,p):
            return p[0] * np.exp(-(x-p[1])**2 / (2*p[2]**2) )\
            + p[3] * np.exp(-(x-p[4])**2 / (2*p[5]**2) )
        def errfunc(p,x,y):
            return y - fitfunc(x,p)

        # make rough fitting params
    	ymax 			= 	np.max(Y)
    	x0 				= 	200
    	sig 			= 	200
    	p_rough 		= 	[ymax , -x0 , sig , ymax , x0 , sig]

        # optimize parameters
        qout,success 	= 	optimize.leastsq(errfunc,p_rough,args=(X,Y),maxfev=5000)
        if success == False:
            print("galaxy fail!!")
            return X_fit,np.zeros_like(X_fit),0
        else:
            # find Y_fit and FWHM
            Y_fit   =	fitfunc(X_fit,qout)
            FWHM1   =   disp2FWHM(qout[2])
            FWHM2   =   disp2FWHM(qout[5])
            return Y_fit,FWHM1,FWHM2

    else:
        def fitfunc(x,p):
            return p[0] * np.exp(-(x-p[1])**2 / (2*p[2]**2) )\
            + p[3] * np.exp(-(x-p[4])**2 / (2*p[5]**2) )\
            + p[6] * np.exp(-(x-p[7])**2 / (2*p[8]**2) )
        def errfunc(p,x,y):
            return y - fitfunc(x,p)

        ymax        =   np.max(Y)
        x0          =   200
        sig         =   100
        p_rough     =   [ ymax , -x0 , sig , ymax , 0 , sig , ymax , x0 , sig ]

        qout,success 	= 	optimize.leastsq(errfunc,p_rough,args=(X,Y),maxfev=5000)
        if success == False:
            print("galaxy fail!!")
            return X_fit,np.zeros_like(X_fit),0
        else:
            # find Y_fit and FWHM
            Y_fit   =	fitfunc(X_fit,qout)
            FWHM1   =   disp2FWHM(qout[2])
            FWHM2   =   disp2FWHM(qout[5])
            FWHM3   =   disp2FWHM(qout[8])
            return Y_fit,FWHM1,FWHM2,FWHM3

def gauss_fit_daddi(X,Y,X_fit):
    """ find gauss fit according to http://iopscience.iop.org/article/10.1088/0004-637X/713/1/686/pdf
    1) fit galaxy profile to 2 gaussians
    2) fit both gaussians to single FWHM

    Parameters
    ----------
    X:      velocity data array
    Y:      luminosity flux density array
    X_fit:  velecity array to fit profile to
    """

    def fitfunc(p,x):
        return p[0] * np.exp(-(x-p[1])**2 / (2*p[2]**2) )\
        + p[3] * np.exp(-(x-p[4])**2 / (2*p[2]**2) )
    def errfunc(p,x,y):
        return y - fitfunc(p,x)

    # initial guess at fitting parameters
    ymax        =   np.max(Y)*.75
    x0          =   200
    sig         =   100
    p_rough     =   [ ymax , -x0 , sig , ymax , x0 ]

    qout,success 	= 	optimize.leastsq(errfunc,p_rough,args=(X,Y),maxfev=5000)
    Y_fit           =   fitfunc(qout,X_fit)
    FWHM            =   disp2FWHM(qout[2])
    return Y_fit,FWHM

def within_dex(x,x0,value,range):
    # Return dataframe x, sliced around x0 with a range of 'range':
    return x[(x[value].values > x0-range) & (x[value].values < x0+range)]


#===============================================================================
""" For radial profiles mainly """
#-------------------------------------------------------------------------------

def read_cloudy_mass_fractions(model_output):
	output        			=   open(model_output,'r')
	lc.clearcache()
	for i,line in enumerate(output):
	    if line.find('Gas Phase Chemical Composition') >= 0:
			# Save the next three lines:
			raw 				=	re.sub('\n','',lc.getline(model_output,i+2))
			raw 				+=	re.sub('\n','',lc.getline(model_output,i+3))
			raw 				+=	re.sub('\n','',lc.getline(model_output,i+4))
	raw 					=	raw.replace(':',' ')
	raw 					=	re.split(r'\s+',raw)
	mf 						=	np.array([])
	elements 				=	['He','C','N','O','Ne','Mg','Si','S','Ca','Fe']
	i 						=	0
	for raw1 in raw:
		raw1 				=	raw1.replace(':','')
		if raw1 in elements:
			raw2 				=	raw[i+1]
			raw2 				=	raw2.replace(':','')
			try:
				mf 					=	np.append(mf,float(raw2))
			except:
				mf 					=	np.append(mf,float(raw[i+2]))
		i 						+= 	1

	return(10.**mf)

#===============================================================================
""" For radial profiles mainly """
#-------------------------------------------------------------------------------

def running_mean(x,y,x_final,scale): # x_final = galaxy radius in kpc

    dx_final                =	x_final[1]-x_final[0]

    new_y 					=	np.zeros(len(x_final))
    for i in range(0,len(x_final)):
    	y1 						=	y[np.where((x > x_final[i]) & (x < x_final[i]+dx_final))]
    	if len(y1)>0: new_y[i] 	= 	np.mean(y1)
    new_x         			=   x_final+dx_final/2.
    # new_y      				=   interp_func(new_x)

    # Scale such that the sum stays the same!
    if scale: new_y   		=   new_y*np.sum(y)/np.sum(new_y)
    return(new_x, new_y, dx_final)

#===============================================================================
""" variables """
#-------------------------------------------------------------------------------

def float_prec(x):
    str_x = str(x)
    find_e = str.find(str_x,'e-')
    if find_e != -1:
        p = float(str_x[find_e+2:])
    else:
        str_x = str_x[2:]
        p = 1
        for i in range(len(str_x)):
            if str_x[i] == '0':
                p += 1
            else:
                break
    return p

def int_prec(x,p,err=False):
    if p == 1:
        err=True

    if err == False:
        mult = 10**(p)
        x_new = int(round(x/mult)) * mult
    else:
        mult = 10**(p-1)
        x_new = int(round(x/mult)) * mult

    return x_new

class var:
    def __init__(self,val,err):
        self.val = float(val)
        self.err = float(err)

        try: # try for scalar values in val and err

            if all(( self.err < 1, self.err != 0.0 )):

                p = int(float_prec(self.err))
                self.p = p
                self.pval = round(self.val,p)
                self.perr = round(self.err,p)

            elif self.err == 0.0:

                if self.val < 1:
                    p = int(float_prec(self.val))
                    self.p = p
                    self.pval = round(self.val,p)
                    self.perr = .5 * 10**-p
                else:
                    p = len(str(int(self.val)))
                    self.p = p
                    self.pval = int_prec(self.val,p)
                    self.perr = .5 * 10**-p

            elif self.err > 1:
                p = len(str(int(self.err)))
                self.p = p
                self.pval = int_prec(self.val,p)
                self.perr = int_prec(self.err,p,err=True)

        except:
            print("made into an array")

            try: # try for same length array/list in val and err

                self.val = np.array(self.val)
                self.err = np.array(self.err)
                self.p = np.zeros_like(self.err)
                self.pval = np.zeros_like(self.err)
                self.perr = np.zeros_like(self.err)

                for i in range(len(self.err)):

                    if self.err[i] < 1:

                        p = float_prec(self.err[i])
                        self.p[i] = p
                        self.pval[i] = round(self.val[i],p)
                        self.perr[i] = round(self.err[i],p)

                    else:

                        p = len(str(int(self.err[i])))
                        self.p[i] = p
                        self.pval[i] = int_prec(self.val[i],p)
                        self.perr[i] = int_prec(self.err[i],p,err=True)

                self.p = np.array(self.p).astype(int)

            except:
                print("try something else")

#===============================================================================
""" Linearized Data Analysis """
#-------------------------------------------------------------------------------

def SS_xx(X): #
	x_bar = np.average(X)
	sum = 0
	for x in X:
		sum += (x-x_bar)**2
	return sum

def SS_xy(X,Y):
	if len(X) != len(Y):
		print("X and Y must be same length")
		print(len(X),len(Y))
		raise ValueError

	x_bar = np.average(X)
	y_bar = np.average(Y)

	sum = 0
	N = len(X)
	for n in range(N):
		sum += (X[n]-x_bar)*(Y[n]-y_bar)

	return sum

def m_exp(X,Y):
    """slope of linear fit to data
    Parameters
    ----------
    X: domain data
    Y: range data
    """
    if len(X) != len(Y):
        print("X and Y must be same length")
        print(len(X),len(Y))
        raise ValueError
    return SS_xy(X,Y)/SS_xx(X)

def b_exp(X,Y):
    """y-intercept of linear fit to data
    Parameters
    ----------
    X: domain data
    Y: range data
    """
    if len(X) != len(Y):
        print("X and Y must be same length")
        print(len(X),len(Y))
        raise ValueError

    y_bar = np.average(Y)
    x_bar = np.average(X)
    m = m_exp(X,Y)

    return y_bar - m*x_bar

def SS_yy(Y):
    """total sum of squares
    Parameters
    ----------
    Y: range data
    """
    y_bar = np.average(Y)
    sum = 0
    for y in Y:
        sum += (y-y_bar)**2
    return sum

def SS_E(X,Y):
    """ error in sum of squares
    Parameters
    ----------
    X: domain data
    Y: range data
    """
    if len(X) != len(Y):
        print("X and Y must be same length")
        raise ValueError

    N = len(X)
    m = m_exp(X,Y)
    b = b_exp(X,Y)

    sum = 0
    for n in range(N):
        y_exp = m*X[n] + b
        sum += (Y[n]-y_exp)**2

    return sum

def SS_R(X,Y):
    """ regression sum of squares
    Parameters
    ----------
    X: domain data
    Y: range data
    """
    if len(X) != len(Y):
        print("X and Y must be same length")
        raise ValueError

    return SS_yy(Y) - SS_E(X,Y)

def sig_y(X,Y):
    """standard deviation of y(x)"""
    if len(X) != len(Y):
        print("X and Y must be same length")
        raise ValueError

    deg_free = len(X)-2 # degrees of freedom, length of array minus the 2 regression params
    assert deg_free != 0, "arrays with length 2 have 0 degrees of freedom. dg.fits.sig_y(X,Y)"
    return np.sqrt(SS_E(X,Y)/deg_free)

def sig_m(X,Y):
    """ standard deviation of fitted slope
    Parameters
    ----------
    X: domain data
    Y: range data
    """
    if len(X) != len(Y):
        print("X and Y must be same length")
        raise ValueError

    return np.sqrt(sig_y(X,Y)/SS_xx(X))

def sig_b(X,Y):
    """ standard deviation of fitted y-intercept
    Parameters
    ----------
    X: domain data
    Y: range data
    """
    if len(X) != len(Y):
        print("X and Y must be same length")
        print(len(X),len(Y))
        raise ValueError

    n = len(X)
    x_bar2 = np.average(X)**2

    return np.sqrt(sig_y(X,Y) * ( (1/n) + (x_bar2/SS_xx(X)) ) )

def RMSE(X,Y):
    """ Root Mean Square Error
    Parameters
    ----------
    X: domain data
    Y: range data
    """
    if len(X) != len(Y):
        print("X and Y must be same length")
        raise ValueError

    n = len(X)
    return np.sqrt(SS_E(X,Y)/n)

def R2(X,Y):
    """ Coefficient of Determination - the variability of y_i accounted for by linear model
    Parameters
    ----------
    X: domain data
    Y: range data
    """
    return SS_R(X,Y)/SS_yy(Y)

def lin_fit(X_data,Y_data,X_lin):
    """ complete linear fit of data

    Parameters
    ----------
    X_data: domain of data
    Y_data: range of data
    X_lin:  higher resolution domain to plot smooth linear fit

    Returns
    -------
    Y_lin:  high resolution range of fitted line
    m:      dg.var class object - slope
    b:      dg.var class object - y-intercept
    """

    X_data=np.array(X_data)
    Y_data=np.array(Y_data)

    if len(X_data) > 2:
        mval    =   m_exp(X_data,Y_data)
        bval    =   b_exp(X_data,Y_data)
        Y_lin   =   mval*X_lin + bval

        merr    =   sig_m(X_data,Y_data)
        berr    =   sig_b(X_data,Y_data)

        R2val   =   R2(X_data,Y_data)

    m           =   var(mval,merr)
    b           =   var(mval,merr)
    R_2         =   var(R2val,0)

    return {'Y_fit':Y_lin, 'm':m, 'b':b, 'R2':R_2}


#===============================================================================
""" Other functions """
#-------------------------------------------------------------------------------

def n_comments(fn, comment='#'):
	'''
	Purpose
	---------
	Counts the number of lines to ignore in a cloudy file to get to the last iteration

	Arguments
	---------
	fn: name of cloudy output file - str

	comment: the string used to comment lines out in cloudy - str
	default = '#'


	'''

	with open(fn, 'r') as f:
	    n_lines         =   0
	    n_comments      =   0
	    n_tot           =   0
	    pattern = re.compile("^\s*{0}".format(comment))
	    for l in f:
	        if pattern.search(l) is None:
	            n_lines += 1         # how many lines of data?
	        else:
	            n_comments += 1      # how many lines of comments?
	            n_lines = 0          # reset line count
	        n_tot       +=      1
	n_skip      =   n_tot - n_lines     # (total number of lines in file minus last iteration)
	# n_skip      =   (n_comments-1)*n_lines+n_comments
	#print('Skip this many lines: ',n_skip)
	return n_skip

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

