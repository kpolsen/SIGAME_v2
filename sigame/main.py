###     Module: main.py of SIGAME                   ###
###     - calls all other sub-modules               ###
###     - collects global results in 'models'       ###

import numpy as np
import pandas as pd
import pdb as pdb
import cPickle
# From SIGAME submodules:
import subgrid_module as subgrid_module
import GMC_module as GMC_module
import dif_module as dif_module
import classes as cl
import aux as aux
import plot as plot
import analysis as analysis

params                      =   np.load('sigame/temp/params.npy').item()
for key,val in params.items():
    exec(key + '=val')

def run():

    print('** Run SIGAME for a selection of galaxies **')

    print('\n Number of galaxies in selection: '+str(len(galnames)))

    params                      =   np.load('sigame/temp/params.npy').item()
    for key,val in params.items():
        exec(key + '=val')
    nGal        	=   len(galnames)

    if do_load_galaxy:
    	load_module.get_galaxy()

    for igalnum in range(0,1):
        print('\n--------------------------------------------------------------\n')

        # Load or add to existing models dataframe:
        print('\n' + (' Global properties of galaxy G'+str(int(igalnum+1))+':').center(20+20+10+10))
        models                  =   aux.global_properties(igalnum=igalnum,nGal=nGal,data_format='dataframe')
        
        for key in models.keys():
            exec(key + '=models[key].values')

        galname                 =   galnames[igalnum]
        print('Full name: %s' % galname)

        if do_subgrid:
            subgrid_module.subgrid(galname=galnames[igalnum],zred=zreds[igalnum])

        if do_create_GMCs:
            GMC_module.create_GMCs(galname=galnames[igalnum],zred=zreds[igalnum])

        if do_line_calc_GMC:
            GMC_results 				=	GMC_module.calc_line_emission(galname=galnames[igalnum],zred=zreds[igalnum])
            for key in GMC_results.keys():
                models[key][models['galnames'] == galnames[igalnum]] = GMC_results[key]

        if do_create_dif_gas:
            dif_module.create_dif(galname=galnames[igalnum],zred=zreds[igalnum])

        if do_line_calc_dif:
            dif_results 				=   dif_module.calc_line_emission(galname=galnames[igalnum],zred=zreds[igalnum],SFRsd=models['SFRsd'][igalnum])
            for key in dif_results.keys():
                models[key][models['galnames'] == galname] = dif_results[key]

        # Save results to buid on them later...
        aux.save_model_results(models)
        # cPickle.dump(models,open('sigame/temp/global_results/'+z1+'_'+str(nGal)+'gals'+ext_DENSE+ext_DIFFUSE+ext,'wb'))
        

    # Sort results (and galaxy names) according to M_star:
    models 					=	models.sort_values(by = 'M_star')
    models                  =   models.reset_index(drop=True)

    aux.save_model_results(models)

    print('\n--------------------------------------------------------------\n')
    
    print('Done!')

def print_results():
    '''
    Print results for selected galaxy model
    '''

    models                      =   aux.find_model_results()
    for key in models.keys():
        exec(key + '=models[key].values')

    dir_DENSE               =   ''
    if ext_DENSE != '_abun': dir1 = '/tests'
    dir_DIFFUSE             =   ''
    if ext_DIFFUSE != '_abun': dir1 = '/tests'
    
    print('\n ONLY SIMULATION STUFF')
    print('+%70s+' % ((5+20+15+15+10+10+10+8)*'-'))
    print('|%5s|%20s|%15s|%15s|%10s|%10s|%10s' % ('Name'.center(5), 'sim name'.center(20), 'Stellar mass'.center(15), 'Gas mass'.center(15), 'SFR'.center(10), 'R_gal'.center(10), 'Z_SFR'.center(10)))
    print('|%5s|%20s|%15s|%15s|%10s|%10s|%10s' % (''.center(5), ''.center(20), '[10^9 Msun]'.center(15), '[10^9 Msun]'.center(15), '[Msun/yr]'.center(10), '[kpc]'.center(10), '[solar]'.center(10)))
    print('+%70s+' % ((5+20+15+15+10+10+10+8)*'-'))
    for igalnum in range(0,len(galnames)):
        print('|%5s|%20s|%15s|%15s|%10s|%10s|%10s' % ('G'+str(igalnum+1),galnames[igalnum].center(20),\
            '{:.4f}'.format(M_star[igalnum]/1e9),\
            '{:.4f}'.format(M_gas[igalnum]/1e9),\
            '{:.4f}'.format(SFR[igalnum]),\
            '{:.4f}'.format(R_gal[igalnum]),\
            '{:.4f}'.format(Zsfr[igalnum])))
    print('+%70s+' % ((5+20+15+15+10+10+10+8)*'-'))
    

    print('\n ISM PHASES')
    # pdb.set_trace()
    print('+%64s+' % ((10+20+20+10+10+10+4)*'-'))
    print('|%10s|%20s|%20s|%10s|%10s|%10s' % ('Name'.center(10), 'sim name'.center(20), 'Total gas mass'.center(20), 'GMC mass'.center(10), 'DNG mass'.center(10), 'DIG mass'.center(10)))
    print('|%10s|%20s|%20s|%10s|%10s|%10s' % (''.center(10), ''.center(20), '[10^9 Msun]'.center(20), '[%]'.center(10), '[%]'.center(10), '[%]'.center(10)))
    print('+%64s+' % ((10+20+10+10+10+4)*'-'))
    M_ISM,M_star,f_GMC,f_DNG,f_DIG,f_ion,M_GMC_out,M_GMC,surfgas,Zsfr	=	[np.zeros(len(galnames)) for i in range(0,10)]
    for igalnum in range(0,len(galnames)):
        GMCgas              =   pd.read_pickle('sigame/temp/GMC/'+'z'+'{:.2f}'.format(zreds[igalnum])+'_'+galnames[igalnum]+'_GMC'+'.gas')
        # GMCgas              =   pd.read_pickle(GMC_path+'z'+'{:.2f}'.format(zreds[igalnum])+'_'+galnames[igalnum]+'_GMC'+ext_DENSE+'_em.gas')
        SPHgas              =   pd.read_pickle('sigame/temp/sim_FUV/'+'z'+'{:.2f}'.format(zreds[igalnum])+'_'+galnames[igalnum]+'_sim1.gas')
        SPHstar             =   pd.read_pickle('sigame/temp/sim_FUV/'+'z'+'{:.2f}'.format(zreds[igalnum])+'_'+galnames[igalnum]+'_sim1.star')
        Zsfr[igalnum]       =   np.sum(SPHgas['SFR']*20.*SPHgas['Z'])/np.sum(SPHgas['SFR'])
        r_sim 			    =	np.sqrt(SPHgas['x']**2+SPHgas['y']**2)
        surfgas[igalnum]    =	np.sum(SPHgas['m'][r_sim < 1])/(np.pi*1000.**2) # Msun/pc^2
        M_star[igalnum]	    =	M_star[igalnum]
        M_GMC[igalnum]      =   np.sum(GMCgas['m'])
        M_ISM[igalnum]      =   np.sum(SPHgas['m'])
        M_GMC_out[igalnum]  =   np.sum(SPHgas['m'].values*SPHgas['f_H2'].values)-M_GMC[igalnum]
        f_GMC[igalnum] 	    =	M_GMC[igalnum]/M_gas[igalnum]*100.
        difgas              =   pd.read_pickle(dif_path+'z'+'{:.2f}'.format(zreds[igalnum])+'_'+galnames[igalnum]+'_dif'+ext_DIFFUSE+'_em.gas')
        DNGgas              =   difgas['m_DNG']
        DIGgas              =   difgas['m_DIG']
        f_DNG[igalnum] 	    =	sum(difgas['m_DNG'])/sum(SPHgas['m'])*100.
        f_DIG[igalnum] 	    =	sum(difgas['m_DIG'])/sum(SPHgas['m'])*100.
        print('|%10s|%20s|%20s|%10s|%10s|%10s' % ('G'+str(igalnum+1),galnames[igalnum].center(20),\
        	'{:.4f}'.format(M_gas[igalnum]/1e9),\
        	'{:.2f}'.format(f_GMC[igalnum]),\
        	'{:.2f}'.format(f_DNG[igalnum]),\
        	'{:.2f}'.format(f_DIG[igalnum])))
    print(np.mean(Zsfr))
    print(np.min(Zsfr),np.max(Zsfr))
    print('Check: ')
    print(f_GMC + f_DNG + f_DIG)
    print('Min and max of gas element masses: %.2f to %.2f Msun' % (min(M_ISM), max(M_ISM)))
    print('Mean: %.2f Msun' % np.mean(M_ISM))
    print('Min and max of GMC fractions: %.2f to %.2f %%' % (min(f_GMC), max(f_GMC)))
    print('Mean: %.2f %%' % np.mean(f_GMC))
    print('Min and max of DNG fractions: %.2f to %.2f %%' % (min(f_DNG), max(f_DNG)))
    print('Mean: %.2f %%' % np.mean(f_DNG))
    print('Min and max of DIG fractions: %.2f to %.2f %%' % (min(f_DIG), max(f_DIG)))
    print('Mean: %.2f %%' % np.mean(f_DIG))
    
    print('\nGas mass fractions')
    f_gas 			=	M_ISM/(M_ISM+M_star)*100.
    print('Min and max of gas mass fractions: %.2f to %.2f %%' % (min(f_gas), max(f_gas)))
    print('Mean: %.2f %%' % np.mean(f_gas))
    
    print('\nMolecular gas mass fractions (M_mol/(M_mol+M_star))')
    f_mol 			=	M_GMC/(M_GMC+M_star)*100.
    print('Min and max of molecular gas mass fractions: %.2f to %.2f %%' % (min(f_mol), max(f_mol)))
    print('Mean: %.2f %%' % np.mean(f_mol))

    print('\nThrowing away this much gas mass before forming GMCs: ')
    print('Min and max: '+str(np.min(M_GMC_out/M_ISM)*100.)+' '+str(np.max(M_GMC_out/M_ISM)*100.)+' %')

    print('\nThrowing away this much GMC gas mass before forming GMCs: ')
    print('Min and max: '+str(np.min(M_GMC_out/M_GMC)*100.)+' '+str(np.max(M_GMC_out/M_GMC)*100.)+' %')
    # pdb.set_trace()

    print('\nGas surface densities in central 1 kpc: ')
    print('Min and max: '+str(min(surfgas))+' '+str(max(surfgas)))
    print('Mean: '+str(np.mean(surfgas)))

    cont                =   raw_input('continue (y/n)? [default: y]')
    if cont == '': cont = 'y'
    if cont == 'y':

        print('\n [CII] from different phases')
        print('+%75s+' % ((10+10+10+10+20+10+10+10+5)*'-'))
        print('|%10s|%10s|%10s|%10s|%20s|%10s|%10s|%10s' % ('Name'.center(10), 'sim name'.center(10), 'SFR'.center(10), 'Sigma_SFR'.center(10), 'Total [CII] lum'.center(20), 'GMC'.center(10), 'DNG'.center(10), 'WIM/DIG'.center(10)))
        print('|%10s|%10s|%10s|%10s|%20s|%10s|%10s|%10s' % (''.center(10), ''.center(10), '[Msun/yr]'.center(10), '[-/kpc^2]'.center(10), '[1e8 Lsun]'.center(20), '[%]'.center(10), '[%]'.center(10), '[%]'.center(10)))
        print('+%75s+' % ((10+10+10+10+20+10+10+10+5)*'-'))
        L_GMC,L_DNG,L_DIG 		=	[np.zeros(len(galnames)) for i in range(0,3)]
        for igalnum in range(0,len(galnames)):
            SPHgas          =   pd.read_pickle('sigame/temp/sim_FUV/z'+'{:.2f}'.format(zreds[igalnum])+'_'+galnames[igalnum]+'_sim1.gas')
            GMCgas			=	pd.read_pickle(GMC_path+'z'+'{:.2f}'.format(zreds[igalnum])+'_'+galnames[igalnum]+'_GMC'+ext_DENSE+'_em.gas')
            difgas 			=	pd.read_pickle(dif_path+'z'+'{:.2f}'.format(zreds[igalnum])+'_'+galnames[igalnum]+'_dif'+ext_DIFFUSE+'_em.gas')
            L_CII 			=	np.sum(np.sum(GMCgas['L_CII'].values)+np.sum(difgas['L_CII_DNG'].values)+np.sum(difgas['L_CII_DIG'].values))
            M_ISM[igalnum] 	=	np.log10(sum(SPHgas['m']))
            L_GMC[igalnum] 	=	sum(GMCgas['L_CII'].values)/L_CII*100.
            L_DNG[igalnum] 	=	sum(difgas['L_CII_DNG'].values)/L_CII*100.
            L_DIG[igalnum] 	=	sum(difgas['L_CII_DIG'].values)/L_CII*100.
            print('|%10s|%10s|%10s|%10s|%20s|%10s|%10s|%10s' % ('G'+str(igalnum+1),\
                galnames[igalnum],\
            	'{:.3f}'.format(SFR[igalnum]),\
            	'{:.3f}'.format(SFRsd[igalnum]),\
            	'{:.4f}'.format(L_CII/1e8),\
            	'{:.2f}'.format(L_GMC[igalnum]),\
            	'{:.2f}'.format(L_DNG[igalnum]),\
            	'{:.2f}'.format(L_DIG[igalnum])))
        print('Min and max of GMC fractions: '+str(min(L_GMC))+' '+str(max(L_GMC)))
        print('Mean: '+str(np.mean(L_GMC)))
        print('Min and max of DNG fractions: '+str(min(L_DNG))+' '+str(max(L_DNG)))
        print('Mean: '+str(np.mean(L_DNG)))
        print('Min and max of DIG fractions: '+str(min(L_DIG))+' '+str(max(L_DIG)))
        print('Mean: '+str(np.mean(L_DIG)))
        
        print('\n ISM MASS-WEIGHTED TEMPERATURE')
        print('+%64s+' % ((10+20+10+10+10+4)*'-'))
        print('|%10s|%20s|%10s|%10s|%10s' % ('Name'.center(10), 'sim [K]'.center(20), 'GMC/PDR'.center(10), 'DNG'.center(10), 'WIM/DIG'.center(10)))
        print('+%64s+' % ((10+20+10+10+10+4)*'-'))
        Tk_DNG 			=	np.zeros([len(galnames),3])
        Tk_DIG 			=	np.zeros([len(galnames),3])
        sims 			=	cl.sim(sim_paths)
        GMCs 			=	cl.GMC(GMC_paths)
        DNGs 			=	cl.DNG(dif_paths)
        DIGs 			=	cl.DIG(dif_paths)
        
        Tkmw_sim,nHmw_sim		    =	sims.Tkmw(),sims.nHmw()
        Tkmw_GMC,nHmw_GMC 			=	GMCs.Tkmw(),GMCs.nHmw()
        Tkmw_DNG,nHmw_DNG 			=	DNGs.Tkmw(),DNGs.nHmw()
        Tkmw_DIG,nHmw_DIG 			=	DIGs.Tkmw(),DIGs.nHmw()
        
        for igalnum in range(0,len(galnames)):
            print('|%10s|%20s|%10s|%10s|%10s' % ('G'+str(igalnum+1), \
            	'{:.2f}'.format(Tkmw_sim[igalnum]),\
            	'{:.2f}'.format(Tkmw_GMC[igalnum]),\
            	'{:.2f}'.format(Tkmw_DNG[igalnum]),\
            	'{:.2f}'.format(Tkmw_DIG[igalnum])))
        print('Min and max of GMC mass-weigthed Tk: '+str(min(Tkmw_GMC))+' '+str(max(Tkmw_GMC)))
        print('Mean: ',np.mean(Tkmw_GMC))
        print('Min and max of DNG mass-weigthed Tk: '+str(min(Tkmw_DNG))+' '+str(max(Tkmw_DNG)))
        print('Mean: ',np.mean(Tkmw_DNG))
        print('Min and max of DIG mass-weigthed Tk: '+str(min(Tkmw_DIG))+' '+str(max(Tkmw_DIG)))
        print('Mean: ',np.mean(Tkmw_DIG))
    

        cont                =   raw_input('continue (y/n)? [default: y]')
        if cont == '': cont = 'y'
        if cont == 'y':
            print('\n ISM MASS-WEIGHTED HYDROGEN DENSITY')
            # print('+%42s+' % ((10+20+10+2)*'-'))
            print('+%64s+' % ((10+20+10+10+10+4)*'-'))
            print('|%10s|%20s|%10s|%10s|%10s' % ('Name'.center(10), 'sim [cm^-3]'.center(20), 'GMC/PDR'.center(10), 'DNG'.center(10), 'WIM/DIG'.center(10)))
            # print('|%10s|%20s|%10s' % ('Name'.center(10),  'DNG'.center(10), 'WIM/DIG'.center(10)))
            print('+%64s+' % ((10+20+10+10+10+4)*'-'))
            # print('+%42s+' % ((10+20+10+2)*'-'))
            for igalnum in range(0,len(galnames)):
                print('|%10s|%20s|%10s|%10s|%10s' % ('G'+str(igalnum+1), \
                	'{:.2f}'.format(nHmw_sim[igalnum]),\
                	'{:.2f}'.format(nHmw_GMC[igalnum]),\
                	'{:.2f}'.format(nHmw_DNG[igalnum]),\
                	'{:.2f}'.format(nHmw_DIG[igalnum])))
            print('Min and max of GMC mass-weigthed nH: '+str(min(nHmw_GMC))+' '+str(max(nHmw_GMC)))
            print('Mean: ',np.mean(nHmw_GMC))
            print('Min and max of DNG mass-weigthed nH: '+str(min(nHmw_DNG))+' '+str(max(nHmw_DNG)))
            print('Mean: ',np.mean(nHmw_DNG))
            print('Min and max of DIG mass-weigthed nH: '+str(min(nHmw_DIG))+' '+str(max(nHmw_DIG)))
            print('Mean: ',np.mean(nHmw_DIG))
            
            print('\nSave text for table in latex')
            table           =    open('sigame/temp/table.txt','w')

            # table.write('Galaxy name | z | R_gal | M_star | M_gas | SFR | Sigma_SFR | Z | halo ID \\\\\n')
            table.write('Galaxy name | z | R_gal | M_star | M_gas | SFR | Sigma_SFR | Z | M_GMC/M_gas | M_DNG/M_gas | M_DIG/M_gas $  \\\\\n')
            Zsfr 			=	sims.Zsfr()
            zs              =   np.array([])
            halos           =   np.array([])
            snaps           =   np.array([])
            names 		    =	np.array([])
            for igalnum in range(0,len(galnames)):
                zs 					=	np.append(zs,zreds[igalnum])
                halo 				=	galnames[igalnum].split('hh')[1].split('_')[0]
                halos               =   np.append(halos,int(halo))
                snap                =   galnames[igalnum].split('s')[1].split('_')[0]
                snaps               =   np.append(snaps,int(snap))
                names               =   np.append(names,'G'+str(igalnum+1))
                line = 'G'+str(igalnum+1)+\
                ' & \t'+'{:.2f}'.format(zreds[igalnum])+\
                ' & \t'+'{:.2f}'.format(R_gal[igalnum])+\
                ' \t& \t'+'{:.3f}'.format(M_star[igalnum]/1e9)+\
                ' & \t'+'{:.3f}'.format(M_gas[igalnum]/1e9)+\
                ' & \t'+'{:.3f}'.format(SFR[igalnum])+\
                ' & \t'+'{:.3f}'.format(Zsfr[igalnum])+ \
                ' & \t'+'{:.3f}'.format(M_GMC[igalnum]/M_gas[igalnum])+ \
                ' & \t'+'{:.3f}'.format(M_DNG[igalnum]/M_gas[igalnum])+ \
                ' & \t'+'{:.3f}'.format(M_DIG[igalnum]/M_gas[igalnum])+ ' \\\\\n'
                # ' & \t'+halo + ' \\\\\n'
                # ' & \t'+'{:.3f}'.format(models['SFRsd'][igalnum])+\
                table.write(line)
            table.close()
            print('Mean redshift: '+str(np.mean(zs)))
            
            galaxy_sample           =   pd.DataFrame({'names':names,'halo number':halos.astype(int),'snapshot':snaps.astype(int)})
            galaxy_sample.to_pickle('Tables/galaxy_sample')

            pdb.set_trace()














