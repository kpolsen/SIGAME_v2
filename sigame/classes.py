# coding=utf-8
###     Module: classes.py of SIGAME                ###
###     Makes classes to do quick analysis of:      ###
###     - results for GMCs                          ###
###     - results for DNG      						###
###     - results for DIG      						###


import pandas as pd
import numpy as np
import pdb as pdb
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

params                      =   np.load('sigame/temp/params.npy').item()
for key,val in params.items():
    exec(key + '=val')

class sim(object):
	# Class for handling gas data

	def __init__(self,obj_path):
		## Here "obj_path" is a list of paths to the dataframes containing the results from running SÍGAME on a galaxy sample ##
    	
		self.galaxies 		= 	[pd.read_pickle(path) for path in obj_path]


	def mass(self):
		# Total mass of gas elements:
		return([sum(galaxy['m']) for galaxy in self.galaxies])

	def surf_gas(self):
		# Surface density of gas:
		return([sum(galaxy['Tk']*galaxy['m'])/sum(galaxy['m']) for galaxy in self.galaxies])

	def Zsfr(self):
		# Mass weighted FUV:
		return([sum(galaxy['Z']*galaxy['SFR'])/sum(galaxy['SFR']) for galaxy in self.galaxies])

	def Zmean(self):
		# Mass weighted Z:
		return([sum(galaxy['Z'])/len(galaxy['Z']) for galaxy in self.galaxies])

	# Mass-weighted stuff

	def Zmw(self):
		# Mass weighted FUV:
		return([sum(galaxy['Z']*galaxy['m'])/sum(galaxy['m']) for galaxy in self.galaxies])

	def FUVmw(self):
		# Mass weighted FUV:
		return([sum(galaxy['FUV']*galaxy['m'])/sum(galaxy['m']) for galaxy in self.galaxies])

	def Tkmw(self):
		# Mass weighted Tk:
		return([sum(galaxy['Tk']*galaxy['m'])/sum(galaxy['m']) for galaxy in self.galaxies])

	def nHmw(self):
		# Mass weighted nH:
		return([sum(galaxy['nH']*galaxy['m'])/sum(galaxy['m']) for galaxy in self.galaxies])

	def Pextmw(self):
		# Mass weighted Tk:
		return([sum(galaxy['P_ext']*galaxy['m'])/sum(galaxy['m']) for galaxy in self.galaxies])

class star(object):
	# Class for handling star data

	def __init__(self,obj_path):
		## Here "obj_path" is a list of paths to the dataframes containing the results from running SÍGAME on a galaxy sample ##
    	
		self.galaxies 		= 	[pd.read_pickle(path) for path in obj_path]


	def mass(self):
		# Total mass of star particles:
		return(np.array([sum(galaxy['m']) for galaxy in self.galaxies]))


	def SFR(self,R):
		# Total SFR averaged over passed 100 Myr within R_gal:
		i 					=	0
		SFR 				=	np.array([])
		for galaxy in self.galaxies:
			galaxy['r'] 		=	np.sqrt(galaxy['x']**2.+galaxy['y']**2.+galaxy['z']**2.)
			SFR 				=	np.append(SFR,sum(galaxy['m'].values[(galaxy['age'].values < 100) & (galaxy['r'].values < R[i])])/100e6)
			i 					+=	1
		return(SFR)

	def SFRsd(self,R):
		# Total surface density of SFR averaged over passed 100 Myr within R_gal:
		for galaxy in self.galaxies:
			galaxy['r'] 		=	np.sqrt(galaxy['x']**2.+galaxy['y']**2.+galaxy['z']**2.)
		return([sum(galaxy['m'].values[(galaxy['age'].values < 100) & (galaxy['r'].values < R)])/100e6/(np.pi*R**2.) for galaxy in self.galaxies])


	def Rhl(self):
		# half-light radius:
		FWHM 		=	np.array([])
		bins 		=	300.	
		j=0	
		for galaxy in self.galaxies:
			r 				=	np.sqrt(galaxy['x'].values**2.+galaxy['y'].values**2.+galaxy['z'].values**2.)
			r_bin 			=	np.arange(0,max(r),max(r)/bins)
			r_bin 			=	np.arange(0,max(r),max(r)/bins)
			# Encircled luminosity:
			FUV_binned 		=	np.array([sum(galaxy['L_FUV'][(r >= r_bin[i]) & (r < r_bin[i+1])]) for i in range(0,len(r_bin)-1)])
			FUV_accum 		=	np.array([sum(galaxy['L_FUV'][r < r_bin[i+1]]) for i in range(0,len(r_bin)-1)])
			# Smooth out FUV profile a bit:
			FUV_accum1 		=	lowess(FUV_accum,r_bin[0:len(r_bin)-1],frac=0.1,is_sorted=True,it=0)
			FWHMi 			=	min(r_bin[0:len(r_bin)-1][FUV_accum1[:,1] > sum(galaxy['L_FUV'])/2.])
			FWHM 			=	np.append(FWHM,FWHMi)
			# fig 			=	plt.figure(j)
			# ax1 			=	fig.add_subplot(111)
			# ax2				=	ax1.twinx()
			# ax1.plot(r_bin[0:len(r_bin)-1],FUV_accum,'g')
			# ax2.plot(r_bin[0:len(r_bin)-1],FUV_binned,'r')
			# ax2.spines['right'].set_color('r')
			# ax2.yaxis.label.set_color('r')
			# ax2.tick_params(axis='y', colors='r')
			# ax1.plot(FUV_accum1[:,0],FUV_accum1[:,1],'b--')
			# ax1.plot([FWHMi,FWHMi],[0,sum(galaxy['L_FUV'])],'k--')
			# plt.show(block=False)
			# pdb.set_trace()
			j+=1
			# pdb.set_trace()
		return(FWHM)

class GMC(object):


	def __init__(self,obj_path):
		## Here "obj_path" is a list of paths to the dataframes containing the results from running SÍGAME on a galaxy sample ##
    	
		self.galaxies 		= 	[pd.read_pickle(path) for path in obj_path]

	def mass(self):
		# Total mass of GMCs:
		return(np.array([sum(galaxy['m']) for galaxy in self.galaxies]))

	def M_CII(self):
		# Total mass of CII:
		return(np.array([sum(galaxy['m_CII']) for galaxy in self.galaxies]))

	def M_CIII(self):
		# Total mass of CIII:
		return(np.array([sum(galaxy['m_CIII']) for galaxy in self.galaxies]))

	def M_CI(self):
		# Total mass of CI:
		return(np.array([sum(galaxy['m_CI']) for galaxy in self.galaxies]))

	def M_CO(self):
		# Total mass of CO:
		return(np.array([sum(galaxy['m_CO']) for galaxy in self.galaxies]))

	def M_C(self):
		# Total mass of C:
		return(np.array([sum(galaxy['m_C']) for galaxy in self.galaxies]))

	def M_dust(self):
		# Total mass of dust:
		return(np.array([sum(galaxy['m_dust']) for galaxy in self.galaxies]))

	def FUVmean(self):
		# Mass weighted FUV:
		return(np.array([sum(galaxy['FUV'])/len(galaxy['FUV']) for galaxy in self.galaxies]))

	def Zmean(self):
		# Mass weighted FUV:
		return([sum(galaxy['Z'])/len(galaxy['Z']) for galaxy in self.galaxies])

		# List of all values!

	def Z(self):
		return([galaxy['Z'].values for galaxy in self.galaxies])

	def Pext(self):
		return([galaxy['P_ext'].values for galaxy in self.galaxies])

	def FUV(self):
		return([galaxy['FUV'].values for galaxy in self.galaxies])

	def m(self):
		return([galaxy['m'].values for galaxy in self.galaxies])

		# Mass-weighted stuff

	def Zmw(self):
		# Mass weighted Z:
		return(np.array([sum(galaxy['Z']*galaxy['m'])/sum(galaxy['m']) for galaxy in self.galaxies]))

	def FUVmw(self):
		# Mass weighted FUV:
		return(np.array([sum(galaxy['FUV']*galaxy['m'])/sum(galaxy['m']) for galaxy in self.galaxies]))

	def Tkmw(self):
		# Mass weighted Tk:
		return([sum(galaxy['Tkmw']*galaxy['m'])/sum(galaxy['m']) for galaxy in self.galaxies])

	def nHmw(self):
		# Mass weighted nH:
		return([sum(galaxy['nHmw']*galaxy['m'])/sum(galaxy['m']) for galaxy in self.galaxies])

	def Pextmw(self):
		# Mass weighted P_ext:
		return(np.array([sum(galaxy['P_ext']*galaxy['m'])/sum(galaxy['m']) for galaxy in self.galaxies]))

	# Line emission

	def L_CII(self):
		# Total [CII] luminosity:
		return(np.array([sum(galaxy['L_CII']) for galaxy in self.galaxies]))

	def L_OI(self):
		# Total [OI] luminosity:
		return(np.array([sum(galaxy['L_OI']) for galaxy in self.galaxies]))

	def L_OIII(self):
		# Total [OIII] luminosity:
		return(np.array([sum(galaxy['L_OIII']) for galaxy in self.galaxies]))

	def L_NII(self):
		# Total [NII] luminosity:
		return(np.array([sum(galaxy['L_NII']) for galaxy in self.galaxies]))

class DNG(object):


	def __init__(self,obj_path):
		## Here "obj_path" is a list of paths to the dataframes containing the results from running SÍGAME on a galaxy sample ##
    	
		self.galaxies 		= 	[pd.read_pickle(path) for path in obj_path]

	def mass(self):
		# Total mass of DNG clouds:
		return(np.array([sum(galaxy['m_DNG']) for galaxy in self.galaxies]))

	def M_CII(self):
		# Total mass of CII:
		return(np.array([sum(galaxy['m_CII_DNG']) for galaxy in self.galaxies]))

	def M_CIII(self):
		# Total mass of CIII:
		return(np.array([sum(galaxy['m_CIII_DNG']) for galaxy in self.galaxies]))

	def M_CI(self):
		# Total mass of CI:
		return(np.array([sum(galaxy['m_CI_DNG']) for galaxy in self.galaxies]))

	def M_CO(self):
		# Total mass of CO:
		return(np.array([sum(galaxy['m_CO_DNG']) for galaxy in self.galaxies]))

	def M_C(self):
		# Total mass of C:
		return(np.array([sum(galaxy['m_C_DNG']) for galaxy in self.galaxies]))

	def M_dust(self):
		# Total mass of dust:
		return(np.array([sum(galaxy['m_dust_DNG']) for galaxy in self.galaxies]))

	def Zmean(self):
		# Mass weighted Z:
		return([sum(galaxy['Z'])/len(galaxy['Z']) for galaxy in self.galaxies])

		# List of all values!

	def Z(self):
		return([galaxy['Z'].values for galaxy in self.galaxies])

	def FUV(self):
		return([galaxy['FUV'].values for galaxy in self.galaxies])

	def m(self):
		return([galaxy['m_DNG'].values for galaxy in self.galaxies])

	# Mass-weighted stuff

	def Tkmw(self):
		# Mass weighted Tk:
		return([sum(galaxy['Tk_DNG']*galaxy['m_DNG'])/sum(galaxy['m_DNG']) for galaxy in self.galaxies])

	def nHmw(self):
		# Mass weighted nH:
		return([sum(galaxy['nH']*galaxy['m_DNG'])/sum(galaxy['m_DNG']) for galaxy in self.galaxies])

	def Zmw(self):
		# Mass weighted Z:
		return([sum(galaxy['Z']*galaxy['m_DNG'])/sum(galaxy['m_DNG']) for galaxy in self.galaxies])

	# Line emission

	def L_CII(self):
		# Total [CII] luminosity:
		return(np.array([sum(galaxy['L_CII_DNG']) for galaxy in self.galaxies]))

	def L_OI(self):
		# Total [OI] luminosity:
		return(np.array([sum(galaxy['L_OI_DNG']) for galaxy in self.galaxies]))

	def L_OIII(self):
		# Total [OIII] luminosity:
		return(np.array([sum(galaxy['L_OIII_DNG']) for galaxy in self.galaxies]))

	def L_NII(self):
		# Total [NII] luminosity:
		return(np.array([sum(galaxy['L_NII_DNG']) for galaxy in self.galaxies]))

class DIG(object):


	def __init__(self,obj_path):
		## Here "obj_path" is a list of paths to the dataframes containing the results from running SÍGAME on a galaxy sample ##
    	
		self.galaxies 		= 	[pd.read_pickle(path) for path in obj_path]

	def mass(self):
		# Total mass of GMCs:
		return(np.array([sum(galaxy['m_DIG']) for galaxy in self.galaxies]))

	def M_CII(self):
		# Total mass of CII:
		return(np.array([sum(galaxy['m_CII_DIG']) for galaxy in self.galaxies]))

	def M_CIII(self):
		# Total mass of CIII:
		return(np.array([sum(galaxy['m_CIII_DIG']) for galaxy in self.galaxies]))

	def M_CO(self):
		# Total mass of CO:
		return(np.array([sum(galaxy['m_CO_DIG']) for galaxy in self.galaxies]))

	def M_CI(self):
		# Total mass of CI:
		return(np.array([sum(galaxy['m_CI_DIG']) for galaxy in self.galaxies]))

	def M_C(self):
		# Total mass of C:
		return(np.array([sum(galaxy['m_C_DIG']) for galaxy in self.galaxies]))

	def M_dust(self):
		# Total mass of dust:
		return(np.array([sum(galaxy['m_dust_DIG']) for galaxy in self.galaxies]))

		# List of all values!

	def Z(self):
		return([galaxy['Z'].values for galaxy in self.galaxies])

	def FUV(self):
		return([galaxy['FUV'].values for galaxy in self.galaxies])

	def m(self):
		return([galaxy['m_DIG'].values for galaxy in self.galaxies])

	def Zmean(self):
		# Mass weighted FUV:
		return([sum(galaxy['Z'])/len(galaxy['Z']) for galaxy in self.galaxies])

	# Mass-weighted stuff

	def Tkmw(self):
		# Mass weighted Tk:
		return([sum(galaxy['Tk_DIG']*galaxy['m_DIG'])/sum(galaxy['m_DIG']) for galaxy in self.galaxies])

	def nHmw(self):
		# Mass weighted nH:
		return(np.array([sum(galaxy['nH']*galaxy['m_DIG'])/sum(galaxy['m_DIG']) for galaxy in self.galaxies]))

	def nHmean(self):
		# Mean nH:
		return(np.array([sum(galaxy['nH'])/len(galaxy['nH']) for galaxy in self.galaxies]))

	def Zmw(self):
		# Mass weighted Z:
		return([sum(galaxy['Z']*galaxy['m_DIG'])/sum(galaxy['m_DIG']) for galaxy in self.galaxies])

	# Line emission

	def L_CII(self):
		# Total [CII] luminosity:
		return(np.array([sum(galaxy['L_CII_DIG']) for galaxy in self.galaxies]))

	def L_OI(self):
		# Total [OI] luminosity:
		return(np.array([sum(galaxy['L_OI_DIG']) for galaxy in self.galaxies]))

	def L_OIII(self):
		# Total [OIII] luminosity:
		return(np.array([sum(galaxy['L_OIII_DIG']) for galaxy in self.galaxies]))

	def L_NII(self):
		# Total [NII] luminosity:
		return(np.array([sum(galaxy['L_NII_DIG']) for galaxy in self.galaxies]))


	# self.FUVmw = 

	# self.path = obj_path

	# name = obj_path.split('z'+str(int(zred))+'_')[1]
	# name = name.split('_GMC')[0]
	# self.name = name


