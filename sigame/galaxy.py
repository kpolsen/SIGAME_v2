"""
Module with classes to set up the main galaxy object, create and load related data products from the
simulation itself (particle_data class) and processed outputs (datacube class).
"""

# Import other SIGAME modules
import sigame.aux as aux
import sigame.global_results as glo
# import sigame.plot as plot

# Import other modules
import numpy as np
import pandas as pd
import pdb as pdb
from scipy import interpolate
from scipy.interpolate import InterpolatedUnivariateSpline,interp1d,interp2d
import matplotlib.cm as cm
import multiprocessing as mp
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
import astropy as astropy
import astropy.convolution as convol
import os as os

#===============================================================================
"""  Load parameters """
#-------------------------------------------------------------------------------

params                      =   aux.load_parameters()
for key,val in params.items(): exec(key + '=val')

#===========================================================================
""" Main galaxy data classes """
#---------------------------------------------------------------------------

class galaxy:
    """An object referring to one particular galaxy.

    Parameters
    ----------
    gal_index: int
        Galaxy index, default: 0

    Examples
    --------
    >>> import galaxy as gal
    >>> gal_ob = gal.galaxy(gal_index=0)

    """

    def __init__(self, gal_index):

        # get global results
        GR                  =   glo.global_results()

        if verbose: print("constructing galaxy...")

        # grab some info from global results for this galaxy
        self.gal_index      =   gal_index
        self.radius         =   GR.R_gal[gal_index]
        self.name           =   GR.galnames[gal_index]
        self.zred           =   GR.zreds[gal_index]
        self.SFR            =   GR.SFR[gal_index]
        self.Zsfr           =   GR.Zsfr[gal_index]
        self.SFRsd          =   GR.SFRsd[gal_index]
        self.UV_str         =   aux.get_UV_str(z1,self.SFRsd)
        self.lum_dist       =   GR.lum_dist[gal_index]
        self.ang_dist_kpc   =   self.lum_dist*1000./(1+self.zred)**2

        # TEST: skipping this part while trying to switch to python 3

        # add attributes from args
        # for key in args: setattr(self,key,args[key])

        # add objects
        self.add_attr('datacube')
        self.add_attr('particle_data')

        if verbose: print("galaxy %s constructed.\n" % self.name)

    def get_radial_axis(self):
        """ returns 1D radius array for galaxy """
        radial_axis =   np.linspace(0,self.radius,self.N_radial_bins+1)
        dr          =   radial_axis[1]-radial_axis[0]
        return radial_axis + dr/2.

    def check_classification(self):
        """ checks if galaxy classification is correct (all galaxies are
        initialized with a 'spherical' classification.) """
        self.particle_data.classify_galaxy(self)

    def add_attr(self,attr_name,verbose=False):
        """ creates desired attribute and adds it to galaxy. """
        if hasattr(self, attr_name):
            if verbose: print("%s already has attribute %s" % (self.name,attr_name))
        else:
            if verbose: print("Adding %s attribute to %s ..." % (attr_name,self.name) )
            if attr_name == 'datacube': ob = datacube(self)
            if attr_name == 'particle_data': ob = particle_data(self)
            setattr(self,attr_name,ob)

    def check_for_attr(self,attr_name,**kwargs):
        """ checks if galaxy has a specific attribute, if not then adds
        it. """
        if hasattr( self , attr_name ):
            print("%s has %s." % (self.name,attr_name) )
        else:
            print("%s doesn't have %s. Adding to galaxy ..." % (self.name,attr_name) )
            self.add_attr(attr_name,**kwargs)

class particle_data:
    """An object referring to the particle data (sim or ISM)

    .. note:: Must be added as an attribute to a galaxy object.

    Parameters
    ----------
    gal_ob : object
        Instance of galaxy class.
    silent : bool
        Parameter telling the code to do (or not) print statements.

    Examples
    --------
    >>> import galaxy as gal
    >>> gal_ob = gal.galaxy(gal_index=0)
    >>> simgas = gal_ob.particle_data.get_data(data='sim')['gas']

    """

    def __init__(self,gal_ob,**kwargs):

        for key,val in kwargs.items():
            exec('globals()["' + key + '"]' + '=val')

        if verbose: print("constructing particle_data object...")

        self.xyz_units = xyz_units

        # add labels for spatial dimentions
        dim     =   ['x','y','z']
        for x in dim: setattr(self,x + 'label','%s [%s]' % (x,xyz_units))

        # add galaxy
        self.gal_ob =   gal_ob

        # add dictionaries of names for all sim types and ISM phases
        self.add_names()

        if verbose: print("particle_data object constructed for %s.\n" % gal_ob.name)


    #---------------------------------------------------------------------------
    # Getting and rotating data
    #---------------------------------------------------------------------------

    def __get_sim_name(self,sim_type):
        return d_data + 'particle_data/sim_data/z%.2f_%s_sim.%s' % (self.gal_ob.zred, self.gal_ob.name, sim_type)

    def __get_phase_name(self,ISM_phase):
        return d_data + 'particle_data/ISM_data/z%.2f_%s_%s.h5' % (self.gal_ob.zred, self.gal_ob.name, ISM_phase)

    # KPO: deprecating...
    def add_names(self):
        """Add file location for particle data file (sim_type or ISM_phase)
        """

        # make empty containers to collect names
        sim_names   =   {}
        ISM_names   =   {}

        # add names to containers
        for sim_type in sim_types: sim_names[sim_type] = aux.get_file_location(gal_ob=self.gal_ob, sim_type=sim_type)
        for ISM_phase in ISM_phases: ISM_names[ISM_phase] = aux.get_file_location(gal_ob=self.gal_ob, ISM_phase=ISM_phase)

        for sim_type in sim_types: sim_names[sim_type] = self.__get_sim_name(sim_type)
        for ISM_phase in ISM_phases: ISM_names[ISM_phase] = self.__get_phase_name(ISM_phase)

        self.sim_names  =   sim_names
        self.ISM_names  =   ISM_names


    def get_data(self,**kwargs):
        """Returns **rotated** particle data as dataframe.

        Parameters
        ----------
        particle_name : str
            Name of particles, can be ['gas','star','dm','GMC','dif'], default: 'GMC'

        """

        # handle default values and kwargs
        args                =   dict(particle_name='GMC')
        args                =   aux.update_dictionary(args,kwargs)

        if args['particle_name'] in ['gas','star','dm']: args['data'] = 'sim'
        if args['particle_name'] in ['GMC','dif']: args['data'] = 'ISM'

        # get raw data
        data  =   self.get_raw_data(**args)[args['particle_name']]

        # get empty arrays for rotated positions and velocities
        size    =   data.shape[0]
        X       =   np.zeros( (size,3) )
        V       =   np.zeros_like(X)

        # populate X and V with unrotated values
        for i,x in enumerate(['x','y','z']):
            X[:,i]  =   data[x].values
            V[:,i]  =   data['v'+x].values

        # rotate X and Y
        X   =   self.__get_rotated_matrix(X)
        V   =   self.__get_rotated_matrix(V)

        # update data with rotated values
        for i,x in enumerate(['x','y','z']):
            data[x]     =   X[:,i]
            data['v'+x] =   V[:,i]

        # add radius and speed columns
        data['radius']  =   self.__get_magnitudes(X)
        data['speed']   =   self.__get_magnitudes(V)

        return data

    def get_raw_data(self,**kwargs):
        """Returns raw **not rotated** particle data in dictionary.
        """

        for key,val in kwargs.items():
            exec('globals()["' + key + '"]' + '=val')

        # choose which particle_data to load
        sim_types
        if data == 'sim':
            names   =   self.sim_names
        if data == 'ISM':
            names   =   self.ISM_names
        # make empty container for pandas data
        collection  =   {}

        # collect all data into container
        for key,name in names.items():
            if data == 'sim': data1            =   aux.load_temp_file(gal_ob=self.gal_ob,sim_type=key)
            if data == 'ISM': data1            =   aux.load_temp_file(gal_ob=self.gal_ob,ISM_phase=key)

            collection[key]     =   data1

        return collection

    def __get_rotated_matrix(self,vectors):
        """Rotates positions/velocities and returns as numpy array.
        """

        # rotation angle (radians)
        phi =   np.deg2rad( float(inc_dc) )

        # rotation matrix
        rotmatrix       =   aux.rotmatrix( phi, axis='y' )
        return vectors.dot(rotmatrix)

    def __get_magnitudes(self,vectors):
        """Calculates lengths of vectors in numpy array and returns as numpy array.
        """
        return np.sqrt( (vectors**2).sum(axis=1) )

    def plot_map(self,**kwargs):
        ''' Creates map of surface densities (or sum) of some parameters in sim or ISM data.


        '''

        # handle default values and kwargs
        args        =   dict(quan='m', ISM_phase='', sim_type='', grid_length=100, x_res=0.5, get_sum=False)
        args        =   aux.update_dictionary(args,kwargs)
        for key in args: exec(key + '=args[key]')

        map_array   =   self.get_map(**args)
        x, y        =   np.meshgrid(map_array['X'], map_array['Y'])

        if quan == 'SFR': lab = getlabel('SFR')

        plot.simple_plot(figsize=(8, 8),plot_margin=0.15,xr=[-R_max,R_max],yr=[-R_max,R_max],\
            x1=x,y1=y,col1=map_array['Z'],\
            colorbar1=True,lab_colorbar1=lab,\
            aspect='equal',\
            contour_type1='mesh',nlev1=100,xlab='x [kpc]',ylab='y [kpc]',title='G%s' % (self.gal_ob.gal_index+1),\
            textfs=9,textcol='white')

    def get_map(self,**kwargs):
        ''' Creates map of surface densities (or sum) of some parameters in sim or ISM data.

        Parameters
        ----------

        quan : str
            What gets mapped, default: 'm'

        ISM_phase : str
            If set, ISM data will be used, default: ''

        sim_type : str
            If set, sim data will be used, default: ''

        get_sum : bool
            If True, return map of summed values not divided by area, default: False
        '''

        # handle default values and kwargs
        args        =   dict(quan='m', ISM_phase='', sim_type='', grid_length=100, x_res=0.5, get_sum=False)
        args        =   aux.update_dictionary(args,kwargs)
        for key in args: exec(key + '=args[key]')

        # pick data to load
        if ISM_phase != '':
            data    =   self.get_raw_data(data='ISM')[ISM_phase]
        if sim_type != '':
            data    =   self.get_raw_data(data='sim')[sim_type]

        # get positional grid
        X           =   np.arange(-x_max_pc/1000., x_max_pc/1000., x_res) # kpc
        dx          =   X[-1] - X[-2]
        grid        =   np.matmul( X[:,None] , X[None,:] )

        # create empty map
        map_array   =   np.zeros_like(grid)

        for j,y in enumerate(X[:-1]):
            for k,x in enumerate(X[:-1]):

                # create mask that selects particles in a pixel at position [x,y]
                gas1            =   data[ (data.y >= X[j]) & (data.y < X[j+1]) & (data.x >= X[k]) & (data.x < X[k+1]) ]

                # get total mass of pixel
                if get_sum: map_array[j,k]  =   np.sum( gas1[quan].values )
                if not get_sum: map_array[j,k]  =   np.sum( gas1[quan].values )/dx**2.

        # collect results
        results     =   dict(X=X, Y=X, Z=map_array, title='', plot_style='map')
        results     =   aux.update_dictionary(results,kwargs)

        # return plot_it instance
        return results

    def add_Z_map(self,**kwargs):
        """Adds metallicity map as numpy array to particle data object, if not stored already in sigame/temp/maps/metallicity/.
        """

        # handle default args and kwargs
        args    =   dict(ow=False)
        args    =   aux.update_dictionary(args,kwargs)
        for key in args: exec(key + '=args[key]')

        for key in args: exec(key + '=args[key]')
        file_location    =    aux.get_file_location(gal_ob=self.gal_ob,map_type='Z')
        if os.path.exists(file_location):
            print('Particle data object already has Z map - loading')
            if kwargs['ow'] == True:
                Z_map    =   self.get_Z_map(**kwargs)
            print('aa')
            setattr(self, 'Z_map',aux.load_temp_file(gal_ob=self.gal_ob,map_type='Z')['data'][0])
        else:
            print('Z map not stored for this particle data object, creating it. ')
            Z_map    =   self.get_Z_map(**kwargs)
            setattr(self, 'Z_map',Z_map)

class datacube:
    """An object referring to the datacube constructed for a galaxy (in one of the ISM phases)

    .. note:: Must be added as an attribute to a galaxy object.

    """

    def __init__(self,gal_ob,**kwargs):
        """Initializing the datacube object of a galaxy.

        Examples
        --------
        >>> import galaxy as gal
        >>> gal_ob = gal.galaxy(gal_index=0)
        >>> dc_CII_GMC = gal_ob.get_dc(ISM_dc_phase='GMC',target='L_CII')['data']
        """

        for key,val in kwargs.items():
            exec('globals()["' + key + '"]' + '=val')

        self.xlabel = xlabel
        self.ylabel = ylabel
        self.vlabel = vlabel
        self.flabel = flabel
        self.color = color

        if verbose: print("constructing datacube...")

        # add attributes
        self.gal_ob        =   gal_ob
        self.add_shape()

        if verbose: print("datacube constructed for %s.\n" % gal_ob.name)

    #---------------------------------------------------------------------------
    # Basic methods
    #---------------------------------------------------------------------------

    def get_dc(self,**kwargs):
        """Returns one datacube as numpy array.
        """

        # handle default values and kwargs
        args        =   dict(ISM_dc_phase='',target='')
        args        =   aux.update_dictionary(args,kwargs)
        for key in args: exec(key + '=args[key]')

        if ISM_dc_phase in ISM_dc_phases:
            return self.__get_dc_phase(**args)
        else:
            return self.__get_dc_summed(**args)


    def get_total_sum(self,**kwargs):
        """Returns total value of datacube for all datacube ISM phases in dictionary.
        """

        # handle default values and kwargs
        args        =   dict(target='L_CII', all_phases=False, test=False)
        args        =   aux.update_dictionary(args,kwargs)
        for key in args: exec(key + '=args[key]')

        tot_line_lum    =   {}
        for i,ISM_dc_phase in enumerate(ISM_dc_phases):
            dc_i                =   self.get_dc(ISM_dc_phase=ISM_dc_phase,**kwargs)
            if 'L_' in target:
                mom0                =   dc_i*v_res # Jy * km/s
                freq                =   params[kwargs['target'].replace('L_','f_')]
                mom0                =   aux.Jykm_s_to_L_sun(mom0,freq,self.gal_ob.zred,self.gal_ob.lum_dist) # Lsun
                # L = aux.Jykm_s_to_L_sun(dc_Jy*v_res, freq, self.gal_ob.zred,self.gal_ob.lum_dist)
                tot                 =   np.sum(mom0)
            else:
                tot = np.sum(dc_i)
            tot_line_lum[ISM_dc_phase]    =  tot

        if all_phases:
            tot_line_lum    =   sum(tot_line_lum.values())

        return tot_line_lum

    def add_shape(self):
        """Adds tuple of datacube dimensions (v length, x length, y length) as an attribute to datacube object.
        """
        x   =   int(2*x_max_pc/x_res_pc)
        v   =   int(2*v_max/v_res)
        self.shape  =   (v,x,x)

    def get_kpc_per_arcsec(self):
        """Returns physical scale for this galaxy in kpc per arcsec.
        """
        return np.arctan(1./60/60./360.*2.*np.pi)*self.gal_ob.ang_dist_kpc

    def get_x_axis_arcsec(self):
        """Returns positional axis (x or y) in arcsec.
        """
        return aux.get_x_axis_kpc()/self.get_kpc_per_arcsec()

    #---------------------------------------------------------------------------
    # "Controllers" that call get_ methods
    #---------------------------------------------------------------------------

    def add_moment0_map(self,**kwargs):
        """Adds moment 0 map as numpy array to datacube object, if not there already.
        """

        for key,val in kwargs.items():
            exec('globals()["' + key + '"]' + '=val')

        for key in kwargs: exec(key + '=kwargs[key]')
        if hasattr(self, 'mom0_%s_%s' % (line,ISM_dc_phase)):
            print('Datacube object already has %s %s moment0 attribute' % (ISM_dc_phase,line))
        else:
            print('%s %s moment0 attribute not in datacube object already, creating it. ' % (ISM_dc_phase,line))
            mom0               =   self.get_moment0_map(**kwargs)
            setattr(self, 'mom0_%s_%s' % (line,ISM_dc_phase),mom0)

    def add_m_map(self,**kwargs):
        """Adds mass map as numpy array to datacube object, if not there already.
        """

        kwargs['target'] = 'm'
        for key in kwargs: exec(key + '=kwargs[key]')

        if hasattr(self, 'm_map_%s' % (ISM_dc_phase)):
            print('Datacube object already has %s mass maps attribute' % (ISM_dc_phase))
        else:
            print('%s mass map attribute not in datacube object already, creating it. ' % (ISM_dc_phase))
            m_map    =   self.get_m_map(**kwargs)
            setattr(self, 'm_map_%s' % (ISM_dc_phase),m_map)

    def add_Z_map(self,**kwargs):
        """Adds metallicity map as numpy array to datacube object, if not there already.
        """

        for key in kwargs: exec(key + '=kwargs[key]')
        if hasattr(self, 'Z_map_%s' % (ISM_dc_phase)):
            print('Datacube object already has %s mass maps attribute' % (ISM_dc_phase))
        else:
            print('%s Z map attribute not in datacube object already, creating it. ' % (ISM_dc_phase))
            Z_map    =   self.get_Z_map(target='Z',**kwargs)
            setattr(self, 'Z_map_%s' % (ISM_dc_phase),Z_map)

    def add_vw_map(self,**kwargs):
        NotImplemented

    def add_vw_disp_map(self,**kwargs):
        NotImplemented

    #---------------------------------------------------------------------------
    # Auxilliary methods
    #---------------------------------------------------------------------------

    def __get_dc_phase(self,**kwargs):
        """Returns datacube as a numpy array.

        Parameters
        ----------
        ISM_dc_phase : str
            default: 'GMC'
        target : str
            default: 'L_CII'

        Returns
        -------
        A numpy array or 0 if no datacube was found

        """


        for key,val in kwargs.items():
            exec('globals()["' + key + '"]' + '=val')

        dc          =   0
        dc          =   aux.load_temp_file(gal_ob=self.gal_ob,target=target,ISM_dc_phase=ISM_dc_phase)
        try:
            dc          =   aux.load_temp_file(gal_ob=self.gal_ob,target=target,ISM_dc_phase=ISM_dc_phase)
        except:
            pass

        return dc

    def __get_dc_summed(self,**kwargs):
        """Returns sum of datacube for all datacube ISM phases as numpy array.
        """

        for key,val in kwargs.items():
            exec('globals()["' + key + '"]' + '=val')

        # handle default values and kwargs
        # args        =   dict(ISM_dc_phase='GMC',target='L_CII')
        # args        =   aux.update_dictionary(args,kwargs)
        # for key in args: exec(key + '=args[key]')

        # make empty summed array
        summed  =   np.zeros( self.shape )
        # iterate over list of datacubes
        try:
            for i,ISM_dc_phase in enumerate(ISM_dc_phases):
                dc_i            =   self.__get_dc_phase(ISM_dc_phase=ISM_dc_phase,target=target)
                summed          +=  dc_i
        except:
            summed = dict(datacube=0)

        return summed

    def get_moment0_map(self,**kwargs):
        """Returns moment 0 map in Jy*km/s per pixel as numpy array.

        Parameters
        ----------
        line : str
            Line to use for map, default: 'CII'
        convolve : bool
            if True: convolve with beam of FWHM also supplied, default: False
        FWHM : float
            FWHM of beam to convolve with, default: None
        ISM_dc_phase : str
            ISM datacube phase to use for moment 0 map, default: 'tot' (all ISM phases)
        units : str
            units for moment0 map, default: 'Jykms' for Jy*km/s (other options: 'Wm2' for W/m^2)
        """

        for key,val in kwargs.items():
            exec('globals()["' + key + '"]' + '=val')

        if ISM_dc_phase == 'tot': dc_summed       =   self.__get_dc_summed(target=line)
        if ISM_dc_phase != 'tot': dc_summed       =   self.__get_dc_phase(target=line, ISM_dc_phase=ISM_dc_phase)

        mom0            =   dc_summed.sum(axis=0)*v_res # Jy*km/s per pixel

        if convolve:
            self.FWHM_arcsec    =   aux.get_Herschel_FWHM(line)
            # self.FWHM_arcsec                =   1
            self.FWHM_kpc       =   np.arctan(self.FWHM_arcsec/60./60./360.*2.*np.pi)*self.gal_ob.ang_dist_kpc
            print('Convolving moment 0 map with beam of FWHM of size %.4s arcsec (%.4s kpc)' % (self.FWHM_arcsec,self.FWHM_kpc))

            kernel              =   convol.Gaussian2DKernel(aux.FWHM_to_stddev(self.FWHM_kpc))
            mom0                =   convol.convolve(mom0, kernel)

        mom0                    =   mom0.T # in order to compare with particle data when plotting

        return mom0

    def get_line_prof(self,**kwargs):
        """Returns line profile in pixel-integrated Jy as numpy array.

        Parameters
        ----------
        line : str
            Line to use for map, default: 'CII'

        """

        for key,val in kwargs.items():
            exec('globals()["' + key + '"]' + '=val')

        if ISM_dc_phase == 'tot': dc_summed       =   self.__get_dc_summed(target=line)
        if ISM_dc_phase != 'tot': dc_summed       =   self.__get_dc_phase(target=line, ISM_dc_phase=ISM_dc_phase)

        line_prof           =   dc_summed.sum(axis=2).sum(axis=1) # Jy per velocity bin

        vel                 =   aux.get_v_axis() # km/s

        return vel,line_prof

    #---------------------------------------------------------------------------
    # Backend
    #---------------------------------------------------------------------------

    def create_dc(self,ISM_dc_phase):
        """Calculates the datacube of a specific ISM_phase

        Parameters
        ----------
        ISM_dc_phase: str
            The datacube ISM phase

        Examples
        --------
        >>> gal_ob.datacube.create_dc('GMC')

        """

        if ISM_dc_phase == 'GMC': dataframe            =   self.gal_ob.particle_data.get_raw_data(data='ISM')['GMC']
        if ISM_dc_phase in ['DNG','DIG']: dataframe    =   self.gal_ob.particle_data.get_raw_data(data='ISM')['dif']

        dc,dc_sum           =   aux.mk_datacube(self.gal_ob,dataframe,ISM_dc_phase=ISM_dc_phase)

        # Fix units if looking at line emission
        if 'L_' in target:
            dc_Lsun             =   np.nan_to_num(dc)
            print('Max in Lsun: %.2e ' % np.max(dc_Lsun))
            freq                =   params[target.replace('L_','f_')] #Lsunkms, zred, d_L, nu_rest
            dc                  =   aux.solLum2Jy(dc_Lsun/v_res,self.gal_ob.zred,self.gal_ob.lum_dist,freq)
            print('Max in Jy: %.2e ' % np.max(dc))

        # Save what
        # aux.save_temp_file(dataframe,gal_ob=self.gal_ob,ISM_dc_phase=ISM_dc_phase)


        # Save datacube
        dc                      =   pd.DataFrame({'data':[dc]})
        # dc.metadata             =   {'dc_sum':dc_sum}

        # test convert back to Lsun
        # L = aux.Jykm_s_to_L_sun(dc_Jy*v_res, freq, self.gal_ob.zred,self.gal_ob.lum_dist)
        aux.save_temp_file(dc,gal_ob=self.gal_ob,target=target,ISM_dc_phase=ISM_dc_phase)

#===========================================================================
""" Classes for backend.py """
#---------------------------------------------------------------------------

class subgrid_galaxy(galaxy):
    """
    An object that will contain the subgrid information for one galaxy.
    Child class that inherits from parent class 'galaxy'.
    """

    pass
    def setup_tasks(self):
        '''Controls tasks to be executed, based on existing files and the overwrite [ow] parameter
        '''

        self.gal_ob = dict(zred=self.zred,galname=self.name,gal_index=self.gal_index)

        # If overwriting, do all subgridding
        if ow:
            self.do_FUV         =   True
            self.do_P_ext       =   True
            self.do_GMCs        =   True
            self.do_dif         =   True
            print('Overwrite is ON')
        # If not overwriting, check if subgridding has been done
        if not ow:
            self.do_FUV         =   False
            self.do_P_ext       =   False
            self.do_GMCs        =   False
            self.do_dif         =   False

            # Check for FUV and P_ext
            simgas              =   aux.load_temp_file(gal_ob=self.gal_ob,sim_type='gas')

            if 'FUV' not in simgas.keys(): self.do_FUV = True
            if 'P_ext' not in simgas.keys(): self.do_P_ext = True
            # Check for GMCs
            GMCgas              =   aux.load_temp_file(gal_ob=self.gal_ob,ISM_phase='gmc') #self.particle_data.get_raw_data(data='ISM')['GMC']
            if type(GMCgas) == int: self.do_GMCs = True
            # Check for dif
            difgas              =   aux.load_temp_file(gal_ob=self.gal_ob,ISM_phase='dif') #self.particle_data.get_raw_data(data='ISM')['dif']
            if type(difgas) == int: self.do_dif = True
            print('Overwrite is OFF, will do:')
            if self.do_FUV: print('- Add FUV')
            if self.do_P_ext: print('- Add P_ext')
            if self.do_GMCs: print('- Create GMCs')
            if self.do_dif: print('- Create diffuse gas clouds')
            if self.do_FUV + self.do_P_ext + self.do_GMCs + self.do_dif == 0: print('Nothing!')

    def add_FUV(self):
        '''Adds FUV radiation field to galaxy and stores gas/star sim particle data files again with the new information.
        '''

        print('\nADDING FUV RADIATION FIELD TO GALAXY')

        global simgas, simstar, L_FUV

        simgas              =   aux.load_temp_file(gal_ob=self.gal_ob,sim_type='gas')
        simstar             =   aux.load_temp_file(gal_ob=self.gal_ob,sim_type='star')

        # TEST
        # simgas = simgas[0:1000]
        # simstar = simstar[0:1000]

        # Get FUV grid results from starburst99
        Zs,ages,L_FUV_grid  =   aux.get_FUV_grid_results(z1)

        # Get stellar metallicity and FUV from simulation
        Zsim                =   simstar.copy()['Z']
        agesim              =   simstar.copy()['age']
        agesim[agesim > 1e4] = 1e4 # since sb99 cannot handle anything older than 10Gyr

        # Calculate FUV luminosity of each stellar particle [ergs/s]
        part                    =   0.1
        L_FUV                   =   np.zeros(len(simstar))
        for i in range(0,len(simstar)):
            f                       =   interp2d(np.log10(Zs),np.log10(ages),np.log10(L_FUV_grid))
            L_FUV[i]                =   simstar.loc[i]['m']/1e5*10.**f(np.log10(Zsim[i]),np.log10(agesim[i]))
            if np.isnan(L_FUV[i]) > 0: pdb.set_trace()
            if 1.*i/len(simstar) > part:
                print(int(part*100),' % done!')
                part                    =   part+0.1
        simstar['L_FUV']        =   L_FUV

        print('Minimum FUV luminosity: %s Lsun' % (np.min(L_FUV)))
        print('Maximum FUV luminosity: %s Lsun' % (np.max(L_FUV)))
        if np.min(L_FUV) < 0: 
            print('SB99 grid not sufficient: Some FUV stellar luminosity is negative')
            pdb.set_trac()

        # Find FUV flux at gas particle positions
        F_FUV                   =   np.zeros(len(simgas))
        print('(Multiprocessing starting up! %s cores in use)' % N_cores)
        pool                    =   mp.Pool(processes=N_cores)                    # 8 processors on my Mac Pro, 16 on Betty
        results                 =   [pool.apply_async(aux.FUVfunc, args=(i,simstar,simgas,L_FUV,)) for i in range(0,len(simgas))]#len(simgas)
        pool.close()
        pool.join()

        res                     =   [p.get() for p in results]
        res.sort(key=lambda x: x[0])
        F_FUV = [res[_][1] for _ in range(len(res))]
        print('(Multiprocessing done!)')

        # Store FUV field in local FUV field units
        F_FUV_norm              =   np.array(F_FUV)/(kpc2cm**2*FUV_ISM)
        simgas['FUV']           =   F_FUV_norm

        # Store CR intensity in local CR intensity units
        simgas['CR']            =   F_FUV_norm*CR_ISM

        print('Minimum FUV flux: %s x ISM value' % (np.min(F_FUV_norm)))
        print('Maximum FUV flux: %s x ISM value' % (np.max(F_FUV_norm)))

        # Store new simgas and simstar files
        aux.save_temp_file(simgas, gal_ob=self.gal_ob, sim_type='gas', subgrid=True)
        aux.save_temp_file(simstar,gal_ob=self.gal_ob, sim_type='star', subgrid=True)

        del simgas, simstar, L_FUV

        setattr(self,'FUV_added',True)

    def add_P_ext(self):
        '''Adds external pressure to galaxy and stores gas/star sim particle data files again with the new information.
        '''

        print('\nADDING EXTERNAL PRESSURE FIELD TO GALAXY')

        # Make global variables for Pfunc function
        global simgas, simgas1, simstar, m_gas, m_star

        simgas   =   aux.load_temp_file(gal_ob=self.gal_ob,sim_type='gas')
        simstar  =   aux.load_temp_file(gal_ob=self.gal_ob,sim_type='star')

        # TEST
        # simgas = simgas[0:1000]
        # simstar = simstar[0:1000]

        # Extract star forming gas only:
        simgas1                 =   simgas.copy()
        simgas1                 =   simgas1[simgas1['SFR'] > 0].reset_index()

        # Extract gas and star masses
        m_gas,m_star            =   simgas1['m'].values,simstar['m'].values

        print('(Multiprocessing starting up! %s cores in use)' % N_cores)
        pool                    =   mp.Pool(processes=N_cores)                   # 8 processors on my Mac Pro, 16 on Betty
        results                 =   [pool.apply_async(aux.Pfunc, args=(i, simgas1, simgas, simstar, m_gas, m_star,)) for i in range(0,len(simgas))]#len(simgas)
        pool.close()
        pool.join()

        # sort results since apply_async doesn't return results in order
        res                     =   [p.get() for p in results]
        res.sort(key=lambda x: x[0])
        print('(Multiprocessing done!)')

        # Store pressure,velocity dispersion and surface densities
        for i,key in enumerate(['P_ext','surf_gas','surf_star','sigma_gas','sigma_star','vel_disp_gas']):

            simgas[key]     =   [res[_][i+1] for _ in range(len(res))]
            if key == 'P_ext': simgas[key] = simgas[key]*Msun**2/kpc2m**4/kB/1e6 # K/cm^3

        # Store new simgas and simstar files
        aux.save_temp_file(simgas,gal_ob=self.gal_ob, sim_type='gas', subgrid=True)
        aux.save_temp_file(simstar,gal_ob=self.gal_ob, sim_type='star', subgrid=True)

        del simgas, simgas1, simstar, m_gas, m_star

        setattr(self,'P_ext_added',True)


    def add_GMCs(self):
        '''Generates GMCs and creates new GMC particle data file for a galaxy.
        '''

        print('\nSPLITTING DENSE GAS INTO GMCs FOR THIS GALAXY')

        simgas              =   aux.load_temp_file(gal_ob=self.gal_ob, sim_type='gas', verbose=True)

        # Checking that the sim gas data has what we need
        if 'P_ext' not in simgas.keys():
            print('\nExternal pressure not calculated, doing so now')
            self.add_P_ext()
        if 'FUV' not in simgas.keys():
            print('\nFUV radiation field not calculated, doing so now')
            self.add_FUV()

        simgas              =   simgas[['Z','a_C','a_Ca','a_Fe','a_He','a_Mg','a_N','a_Ne','a_O','a_S','a_Si','f_H21','f_HI1','f_neu','m','h',\
                                'vx','vy','vz','x','y','z','FUV','CR','P_ext','surf_gas','surf_star','sigma_gas','sigma_star','vel_disp_gas']]

        # TEST
        # simgas = simgas[0:100]

        # Neutral (dense) gas mass
        Mneu                =   simgas['m'].values*simgas['f_H21'].values

        print('Number of particles with enough dense mass: %s' % (len(simgas.loc[Mneu > 1e4]['m'])))
        print('Out of: %s' % (len(simgas)))
        print('Dense gas mass fraction out of total ISM mass: %s %%' % (np.sum(Mneu)/np.sum(simgas['m'])*100.))

        if len(Mneu[Mneu > 1e4]) == 0:
            print(" No dense gas w/ mass > 10^4 Msun. Skipping subgrid for this galaxy.")
            pdb.set_trace()

        print('Min and max particle dense gas mass: %s Msun' % (np.min(Mneu[Mneu>0]))+' '+str(np.max(Mneu)))
        print('Throwing away this much gas mass: %s Msun' % (np.sum(Mneu[Mneu < 1e4])))
        print('In percent: %s %%' % (np.sum(Mneu[Mneu < 1e4])/np.sum(Mneu)))

        # Create mass spectrum (probability function = dN/dM normalized)
        b                   =   1.8                                # MW powerlaw slope [Blitz+07]
        if ext_DENSE == '_b3p0': b = 3.0                       # outer MW and M33 powerlaw slope [Rosolowsky+05, Blitz+07]
        if ext_DENSE == '_b1p5': b = 1.5                       # inner MW powerlaw slope [Rosolowsky+05, Blitz+07]
        print('Powerlaw slope for mass spectrum (beta) used is %s' % b)
        Mmin                =   1.e4                               # min mass of GMC
        Mmax                =   np.max(Mneu)                       # max mass of GMC
        tol                 =   Mmin                               # precision in reaching total mass
        nn                  =   100                                # max draw of masses in each run
        n_elements          =   simgas.size
        simgas              =   simgas[Mneu > 1e4]                  # Cut out those with masses < 1e4 !!
        Mneu                =   Mneu[Mneu > 1e4]
        h                   =   simgas['h'].values
        simgas.index        =   range(0,len(simgas))

        if N_cores == 1:
            print('(Not using multiprocessing - 1 core in use)')
            f1,Mgmc,newx,newy,newz      =   aux.GMC_generator(np.arange(0,len(simgas)),Mneu,h,Mmin,Mmax,b,tol,nn,N_cores)
            GMCs                        =   [f1,Mgmc,newx,newy,newz]
            print('(Done!)')
            print('Append results to new dataframe')

            GMCgas              =   pd.DataFrame()
            Mgmc                =   np.array([])
            newx                =   np.array([])
            newy                =   np.array([])
            newz                =   np.array([])
            part                =   0.1

            for i in range(0,len(simgas)):
                Mgmc                =   np.append(Mgmc,GMCs[1][i]) # appending mass
                newx                =   np.append(newx,simgas.loc[i]['x']+GMCs[2][i]) # appending x position
                newy                =   np.append(newy,simgas.loc[i]['y']+GMCs[3][i]) # appending y position
                newz                =   np.append(newz,simgas.loc[i]['z']+GMCs[4][i]) # appending z position
                for ii in range(0,int(GMCs[0][i])):
                    # For each GMC created, duplicate remaining sim gas particle properties
                    GMCgas = pd.DataFrame.append(GMCgas,simgas.loc[i],ignore_index=True)
                if 1.*i/len(simgas) > part:
                    percent             =   np.floor(1.*i/len(simgas)*10.)*10.
                    print('%s %% done!' % percent)
                    part                =   percent/100.+0.1

            try:
                GMCgas['m']         =   Mgmc
            except:
                print(Mgmc)
                pdb.set_trace()

        if N_cores > 1:

            print('(Multiprocessing starting up! %s cores in use)' % N_cores)
            pool                =   mp.Pool(processes=N_cores)
            np.random.seed(len(simgas))                         # so we don't end up with the same random numbers for every galaxy

            results             =   [pool.apply_async(aux.GMC_generator, ([i],Mneu,h,Mmin,Mmax,b,tol,nn,N_cores,)) for i in range(0,len(simgas))]
            pool.close()
            pool.join()
            GMCs                     =   [p.get() for p in results]
            GMCs.sort(key=lambda x: x[0])

            print('(Multiprocessing done!)')

            print('Append results to new dataframe')

            GMCgas              =   pd.DataFrame()
            Mgmc                =   np.array([])
            newx                =   np.array([])
            newy                =   np.array([])
            newz                =   np.array([])
            part                =   0.1

            for i in range(0,len(simgas)):
                # For each sim gas particle with enough dense gas, add GMCs created
                # pdb.set_trace()
                Mgmc                =   np.append(Mgmc,GMCs[i][2]) # appending mass
                newx                =   np.append(newx,simgas.loc[i]['x']+GMCs[i][3]) # appending x position
                newy                =   np.append(newy,simgas.loc[i]['y']+GMCs[i][4]) # appending y position
                newz                =   np.append(newz,simgas.loc[i]['z']+GMCs[i][5]) # appending z position

                for i1 in range(0,int(GMCs[i][1])):
                    # For each GMC created, duplicate remaining sim gas particle properties
                    GMCgas = pd.DataFrame.append(GMCgas,simgas.loc[i],ignore_index=True)
                # Keep track of GMCs added

                if 1.*i/len(simgas) > part:
                    percent             =   np.floor(1.*i/len(simgas)*10.)*10.
                    print('%s %% done!' % percent)
                    part                =   percent/100.+0.1
            try:
                GMCgas['m']         =   Mgmc
            except:
                print(Mgmc)
                pdb.set_trace()

        GMCgas['x']         =   newx
        GMCgas['y']         =   newy
        GMCgas['z']         =   newz
        GMCgas['Rgmc']      =   (GMCgas['m']/290.0)**(1.0/2.0)

        print('Mass of all GMCs created: %.2e - should not exceed:' % np.sum(GMCgas['m']))
        print('Total neutral gas: %.2e ' % np.sum(Mneu))
        print(np.min(GMCgas['Rgmc']),np.max(GMCgas['Rgmc']))

        print(str(len(GMCgas))+' GMCs created!')

        # Store new GMCgas file
        # GMCgas.metadata     =   {'beta':b}
        aux.save_temp_file(GMCgas,gal_ob=self.gal_ob,ISM_phase='GMC')


    def add_dif(self):
        '''Generates diffuse gas clouds and creates new dif particle data file for a galaxy.
        '''

        print('\nCREATING DIFFUSE GAS CLOUDS FOR THIS GALAXY')

        simgas              =   aux.load_temp_file(gal_ob=self.gal_ob,sim_type='gas')

        # TEST
        # simgas = simgas[0:1000]

        # Start new dataframe with only the diffuse gas
        difgas              =   simgas.copy()
        difgas['m']         =   simgas['m'].values*(1.-simgas['f_H21'].values)
        print('Total gas mass in galaxy: %.2e Msun' % (sum(simgas['m'])))
        print('Diffuse gas mass in galaxy: %.2e Msun' % (sum(difgas['m'])))
        print('in percent: %.2f %%' % (sum(difgas['m'])/sum(simgas['m'])*100.))

        # Set radius of diffuse gas clouds
        difgas['R']                     =   difgas['h']

        # Calculate density of diffuse gas clouds [Hydrogen atoms per cm^3]
        difgas['nH']                    =   0.75*np.array(difgas['m'],dtype=np.float64)/(4/3.*np.pi*np.array(difgas['R'],dtype=np.float64)**3.)*Msun/mH/kpc2cm**3

        # Set nH and R to zero when there is no mass
        mask = difgas.m == 0
        difgas.loc[mask, 'nH'] = 0
        difgas.loc[mask, 'R'] = 0

        # Store new difgas file
        aux.save_temp_file(difgas,gal_ob=self.gal_ob,ISM_phase='dif')

class interpolate_clouds(galaxy):
    """
    An object that will interpolate in cloud models of info such as line luminosity for one galaxy.

    Child class that inherits from parent class 'galaxy'.
    """

    pass

    def setup_tasks(self):
        """Controls tasks to be executed, based on existing files and the overwrite [ow] parameter
        """

        self.gal_ob = dict(zred=self.zred,galname=self.name,gal_index=self.gal_index)

        # If overwriting, do all subgridding
        if ow:
            self.do_interpolate_GMCs    =   True
            self.do_interpolate_dif     =   True
            print('Overwrite is ON')
        # If not overwriting, check if subgridding has been done
        if not ow:
            self.do_interpolate_GMCs    =   True
            self.do_interpolate_dif     =   True
            # Check for GMCs
            GMCgas              =   aux.load_temp_file(gal_ob=self.gal_ob,ISM_phase='GMC')
            if type(GMCgas) != int:
                if 'L_CII' in GMCgas.keys(): self.do_interpolate_GMCs = False
            # Check for diffuse gas
            difgas              =   aux.load_temp_file(gal_ob=self.gal_ob,ISM_phase='dif')
            if type(difgas) != int:
                if 'L_CII_DNG' in difgas.keys(): self.do_interpolate_dif = False
            print('Overwrite is OFF, will do:')
            if self.do_interpolate_GMCs: print('- Interpolate in cloud models for GMCs')
            if self.do_interpolate_dif: print('- Interpolate in cloud models for diffuse gas clouds (DNG and DIG)')
            if self.do_interpolate_GMCs + self.do_interpolate_dif == 0: print('Nothing!')

    def interpolate_GMCs(self):
        """Adds info from cloud model runs to GMCs
        """
        print('\nADDING EMISSION INFO TO GMCs IN THIS GALAXY')

        GMCgas              =   aux.load_temp_file(gal_ob=self.gal_ob,ISM_phase='GMC', verbose=True)

        # Load cloudy model grid:
        cloudy_grid_param   =   pd.read_pickle(d_cloudy_models +'GMCgrid'+ext_DENSE+'_'+z1+'.param')

        cloudy_grid_param['ms'] = cloudy_grid_param['Mgmcs']
        cloudy_grid         =   pd.read_pickle(d_cloudy_models + 'GMCgrid'+ext_DENSE+'_'+z1+'_em.models')


        # print(cloudy_grid_param['Mgmcs'])
        # print(cloudy_grid.shape)


        GMCgas_new          =   aux.interpolate_in_GMC_models(GMCgas,cloudy_grid_param,cloudy_grid)

        aux.save_temp_file(GMCgas_new,gal_ob=self.gal_ob,ISM_phase='GMC')

    def interpolate_dif(self):
        """Adds info from cloud model runs to diffuse gas clouds
        """
        print('\nADDING EMISSION INFO TO DIFFUSE GAS IN THIS GALAXY')

        difgas              =   aux.load_temp_file(gal_ob=self.gal_ob,ISM_phase='dif')

        # Get diffuse background FUV to use for this galaxy (OLD WAY)
        self.UV_str         =   aux.get_UV_str(z1,self.SFRsd)

        # Load cloudy model grid:
        cloudy_grid_param   =   pd.read_pickle(d_cloudy_models + 'difgrid_'+self.UV_str+'UV'+ext_DIFFUSE+'_'+z1+'.param')
        cloudy_grid         =   pd.read_pickle(d_cloudy_models + 'difgrid_'+self.UV_str+'UV'+ext_DIFFUSE+'_'+z1+'_em.models')
        difgas_new          =   aux.interpolate_in_dif_models(difgas,cloudy_grid_param,cloudy_grid)

        aux.save_temp_file(difgas_new,gal_ob=self.gal_ob,ISM_phase='dif')

#===============================================================================
""" misc classes """
#-------------------------------------------------------------------------------

# variables used in plots and for presentation
class var:
    """
    takes a value and its err and returns strings of printable values with
    appropriate significant figures for presenting data.
    """

    def __init__(self,val,err):
        self.val    =   float(val)
        self.err    =   float(err)
        ylen        =   len(str(int(self.err)))
        n           =   0

        if self.err > 1:
            self.err_scale  =   10**(ylen-1)
            y               =   int(round(self.err / self.err_scale))
            if self.err > 1000:
                self.print_err   =   '%s $\\times$ 10$^{%s}$' % (y,(ylen-1))
            else:
                self.print_err   =   str(y*self.err_scale)

        elif self.err < 1:
            y               =   self.err
            while y < 1:
                n           +=  1
                y           *=  10
            self.err_scale  =   10**(-n)
            y               =   int(round(self.err / self.err_scale))
            if self.err < .001:
                self.print_err   =   '%s $\\times$ 10$^{-%s}$' % (y,n)
            else:
                self.print_err   =   str(y * self.err_scale)

        if self.val > 1:
            x               =   int(round(self.val / self.err_scale))
            if self.err > 1000:
                self.print_val   =   '%s $\\times$ 10$^{%s}$' % (x,(ylen-1))
            else:
                self.print_val   =   str(x*self.err_scale)

        elif self.val < 1:
            x               =   int(round(self.val / self.err_scale))
            if self.err < .001:
                self.print_val   =   '%s $\\times$ 10$^{-%s}$' % (x,n)
            else:
                self.print_val   =   str(x * self.err_scale)

# generic object
class dict_to_attr:
    """Essentially turns dictionary into a attributes of an object. """
    def __init__(self,dictionary):
        for key in dictionary: setattr(self,key,dictionary[key])
