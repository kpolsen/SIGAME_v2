# coding=utf-8
"""
###     Submodule: plot.py of SIGAME                    ###
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
from matplotlib import gridspec
import pandas as pd
import numpy as np
import pdb as pdb
from scipy.interpolate import RegularGridInterpolator
from matplotlib.mlab import griddata
import scipy as scipy
from matplotlib.colors import ListedColormap
import scipy.stats as stats
from scipy.stats import chisquare
import scipy.optimize as op
from scipy.optimize import leastsq
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import AxesGrid
import pickle
import re                               # to replace parts of string
import itertools
import math
import os.path
import emcee
import corner
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, ScalarFormatter, MaxNLocator
import classes as cl
import analysis as analysis   
import aux as aux

#===============================================================================
"""  Load parameters """
#-------------------------------------------------------------------------------

params                      =   np.load('sigame/temp/params.npy').item()
for key,val in params.items():
    exec(key + '=val')

#===============================================================================
"""  Basic plotting """
#-------------------------------------------------------------------------------

def simple_plot(**kwargs):
    '''
    Purpose
    ---------
    A function to standardize all plots

    What this function does
    ---------
    Plots that can be created this way:
    1. errorbar plot (with x and/or y errorbars)
    2. line
    3. histogram
    4. markers
    5. bar
    6. hexbin
    7. scatter plot
    8. filled region


    Arguments
    ---------
    The '1' below can be replaced by '2', '3', '4' etc for several plotting options in the same axis

    add: if True add to existing axis object, otherwise new figure+axes will be created - bool
    default = False

    fig: figure number - int
    default = 0

    figsize: figure size - (x,y) tuple
    default = (8,6)

    x1: x values - list or numpy array

    y1: y values - list or numpy array

    xr,yr: x and/or y range - list
    default = set automatically by matplotlib

    xlog,ylog: determines if x and/or y axes should be logarithmic - bool
    default = False (range set automatically by matplotlib)

    fill1: if defined, markers will be used - str
    options:
        - 'y': use filled markers
        - 'n': use open markers
    default = 'y'

    ls1: linestyle - str
    default = 'None' (does markers by default)

    ma1: marker type - str
    default = 'x'

    ms1: marker size - float/int
    default = 5

    mew1: marker edge width - float/int
    default = 2

    col1: color of markers/lines - str
    default = 'k'

    ecol1: edgecolor - str
    default = 'k'

    lab1: label for x1,y7 - 'str'
    default = ''

    alpha1: transparency factor - float
    default = '1.1'

    dashes1: custom-made dashes/dots - str
    default = ''

    legend: whether to plot legend or not - bool
    default = False

    leg_fs: legend fontsize - float
    default = not defined (same as fontsize for labels/tick marks)

    legloc: legend location - str or list of coordinates
    default =

    cmap1: colormap for contour plots - str
    default = 'viridis'

    xlab: x axis label - str
    default = no label

    ylab: y axis label - str
    default = no label

    title: plot title - str
    default = no title

    xticks,yticks: whether to put x ticks and/or y ticks or not - bool
    default = True,True

    fontsize: fontsize of tick labels - float
    default = 15

    lab_to_tick: if axis labels should be larger than tick marks, say how much here - float/int
    default = 1.0

    lex1,uex1: lower and upper errorbars on x1 - list or numpy array
    options:
        - if an element in uex1 is 0 that element will be plotted as upper limit in x
    default = None

    ley1,uey1: lower and upper errorbars on y1 - list or numpy array
    options:
        - if an element in uey1 is 0 that element will be plotted as upper limit in y
    default = None

    histo1: whether to make a histogram of x1 values - bool
    default = False

    histo_real1: whether to use real values for histogram or percentages on y axis - bool
    default = False

    bins1: number of bins in histogram - int
    default = 100

    weights1: weights to histogram - list or numpy array
    default = np.ones(len(x1))

    hexbin1: if True, will make hexbin contour plot - bool
    default = False

    contour_type1: it defined, will make contour plot - 'str'
    options:
        -
    default = not defined (will not make contour plot)

    barwidth1: if defined, will make bar plot with this barwidth - float
    default = not defined (will not do bar plot)

    scatter_color1: if defined, will make scatter plot with this color - list or numpy array
    default = not defined (will not do scatter plot)

    colormin1: minimum value for colorbar in scatter plot - float
    default = not defined (will use all values in scatter_color1)

    lab_colorbar: label for colorbar un scatter plot - str
    default = not defined (will not make colorbar)

    hatchstyle1: if defined, make hatched filled region - str
    options:
        - if set to '', fill with one color
        - otherwise, use '/' '//' '///'' etc.
    default = not defined (will not make hatched region)

    text: if defined, add text to figure with this string - str
    default: not defined (no text added)

    textloc: must be specified in normalized axis units - list
    default = [0.1,0.9]

    textbox: if True, put a box around text - bool
    default = False

    fontsize_text: fontsize of text on plot - float/int
    default = 0/7 * fontsize

    grid: turn on grid lines - bool
    default = False

    figname: if defined, a figure will be saved with this path+name - str
    default = not defined (no figure saved)

    figtype: format of figure - str
    default = 'png'

    figres: dpi of saved figure - float
    default = 1000

    SC_return: return scatter plot object or not - bool
    default = False

    '''

    # Set fontsize
    if mpl.rcParams['ytick.labelsize'] == 'medium':
        fontsize            =   15
        if kwargs.has_key('fontsize'): fontsize = kwargs['fontsize']
        mpl.rcParams['ytick.labelsize'] = fontsize
        mpl.rcParams['xtick.labelsize'] = fontsize
    else: fontsize            =   mpl.rcParams['ytick.labelsize']
    lab_to_tick         =   1.
    if kwargs.has_key('lab_to_tick'): lab_to_tick = kwargs['lab_to_tick']
    textcol             =   'black'
    fontsize_text       =   fontsize*0.7


    if kwargs.has_key('add'):
        ax1                 =   plt.gca()
    else:
        fig                 =   0                                       # default figure number
        if kwargs.has_key('fig'): fig = kwargs['fig']
        figsize             =   (8,6)                                   # slightly smaller figure size than default
        if kwargs.has_key('figsize'): figsize = kwargs['figsize']
        fig                 =   plt.figure(fig,figsize=figsize)
        ax1                 =   fig.add_subplot(1,1,1)

    # Default line and marker settings
    ls0                 =   'None'              # do markers by default
    lw0                 =   2                   # linewidth
    ma0                 =   'x'                 # marker type
    ms0                 =   5                   # marker size
    mew0                =   2                   # marker edge width
    col0                =   'k'                 # color
    ecol0               =   'k'                 # color
    lab0                =   ''                  # label
    fill0               =   'y'
    alpha0              =   1.0
    dashes0             =   ''
    only_one_colorbar   =   1
    legend              =   False
    cmap0               =   'viridis'
    bins0               =   100

    # Set axis settings
    xlab,ylab           =   '',''
    if kwargs.has_key('xlab'):
        ax1.set_xlabel(kwargs['xlab'],fontsize=fontsize*lab_to_tick)
    if kwargs.has_key('ylab'):
        ax1.set_ylabel(kwargs['ylab'],fontsize=fontsize*lab_to_tick)
    if kwargs.has_key('title'):
        ax1.set_title(kwargs['title'],fontsize=fontsize*lab_to_tick)
    if kwargs.has_key('histo'):
        ax1.set_ylabel('Number fraction [%]',fontsize=fontsize*lab_to_tick)


    # Add lines/points to plot
    for i in range(1,10):
        done            =   'n'
        if kwargs.has_key('x'+str(i)):
            if kwargs.has_key('x'+str(i)): x = kwargs['x'+str(i)]
            if kwargs.has_key('y'+str(i)): y = kwargs['y'+str(i)]
            # If no x values, make them up
            if not kwargs.has_key('x'+str(i)): x = np.arange(len(y))+1
            ls              =   ls0
            lw              =   lw0
            mew             =   mew0
            ma              =   ma0
            col             =   col0
            ecol            =   ecol0
            ms              =   ms0
            lab             =   lab0
            ls              =   ls0
            fill            =   fill0
            alpha           =   alpha0
            cmap            =   cmap0
            bins            =   bins0
            if kwargs.has_key('ls'+str(i)):
                if kwargs['ls'+str(i)] != 'None': ls = kwargs['ls'+str(i)]
                if kwargs['ls'+str(i)] == 'None': ls = 'None'
            if kwargs.has_key('lw'+str(i)): lw = kwargs['lw'+str(i)]
            if kwargs.has_key('lw'): lw = kwargs['lw'] # or there is a general keyword for ALL lines...
            if kwargs.has_key('mew'+str(i)): mew = kwargs['mew'+str(i)]
            if kwargs.has_key('ma'+str(i)): ma = kwargs['ma'+str(i)]
            if kwargs.has_key('ms'+str(i)): ms = kwargs['ms'+str(i)]
            if kwargs.has_key('col'+str(i)): col = kwargs['col'+str(i)]
            if kwargs.has_key('ecol'+str(i)): ecol = kwargs['ecol'+str(i)]
            if kwargs.has_key('lab'+str(i)): lab = kwargs['lab'+str(i)]
            if kwargs.has_key('lab'+str(i)): legend = 'on' # do make legend
            if kwargs.has_key('legend'):
                if not kwargs['legend']: legend = 'off' # no legend, if legend = False
            if kwargs.has_key('ls'+str(i)): ls = kwargs['ls'+str(i)]
            if kwargs.has_key('fill'+str(i)): fill = kwargs['fill'+str(i)]
            if kwargs.has_key('alpha'+str(i)): alpha = kwargs['alpha'+str(i)]
            if kwargs.has_key('cmap'+str(i)): cmap = kwargs['cmap'+str(i)]


            # ----------------------------------------------
            # 1. Errorbar plot
            # Errorbars/arrows in x AND y direction
            if kwargs.has_key('lex'+str(i)):
                if kwargs.has_key('ley'+str(i)):
                    for x1,y1,lex,uex,ley,uey in zip(x,y,kwargs['lex'+str(i)],kwargs['uex'+str(i)],kwargs['ley'+str(i)],kwargs['uey'+str(i)]):
                        ax1.errorbar(x1,y1,color=col,linestyle="None",xerr=[[lex],[uex]],yerr=[[ley],[uey]],elinewidth=lw,capsize=0,\
                            capthick=0,marker=kwargs['ma'+str(i)],label=kwargs['lab'+str(i)])
            # Errorbars/arrows in x direction
            if kwargs.has_key('lex'+str(i)):
                # print('>> Adding x errorbars!')
                for x1,y1,lex,uex in zip(x,y,kwargs['lex'+str(i)],kwargs['uex'+str(i)]):
                    if uex > 0: # not upper limit, plot errobars
                        ax1.errorbar(x1,y1,color=col,linestyle="None",xerr=[[lex],[uex]],elinewidth=lw,capsize=0,\
                            capthick=0,marker=kwargs['ma'+str(i)])
                    if uex == 0: # upper limit, plot arrows
                        ax1.errorbar(x1,y1,color=col,xerr=lex,\
                           xuplims=True,linestyle="None",linewidth=lw,mew=0,capthick=lw*2)
            # Errorbars/arrows in y direction
            if kwargs.has_key('ley'+str(i)):
                # print('>> Adding y errorbars!')
                for x1,y1,ley,uey in zip(x,y,kwargs['ley'+str(i)],kwargs['uey'+str(i)]):
                    if uey > 0: # not upper limit, plot errobars
                        ax1.errorbar(x1,y1,color=col,linestyle="None",yerr=[[ley],[uey]],elinewidth=lw,capsize=0,\
                            capthick=0,marker=kwargs['ma'+str(i)])
                    if uey == 0: # upper limit, plot arrows
                        ax1.errorbar(x1,y1,color=col,yerr=ley,\
                           uplims=True,linestyle="None",linewidth=lw,mew=0,capthick=lw*2)

            # ----------------------------------------------
            # 2. Line connecting the dots
            if kwargs.has_key('y'+str(i)):

                if type(kwargs['y'+str(i)]) == str: y = ax1.get_ylim()
                if kwargs.has_key('dashes'+str(i)):
                    # print('>> Line plot!')
                    ax1.plot(x,y,linestyle=ls,color=col,lw=lw,label=lab,dashes=kwargs['dashes'+str(i)])
                    continue
                else:
                    if kwargs.has_key('ls'+str(i)):
                        # print('>> Line plot!')
                        ax1.plot(x,y,linestyle=ls,color=col,lw=lw,label=lab)
                        continue

            # ----------------------------------------------
            # 3. Histogram
            if kwargs.has_key('histo'+str(i)):
                # print('>> Histogram!')
                if ls == 'None': ls = '-'
                weights             =   np.ones(len(x))
                if kwargs.has_key('bins'+str(i)): bins = kwargs['bins'+str(i)]
                if kwargs.has_key('weights'+str(i)): weights = kwargs.has_key['weights'+str(i)]
                if kwargs.has_key('histo_real'+str(i)):
                    make_histo(x,bins,col,lab,percent=False,weights=weights,lw=lw,ls=ls)
                else:
                    make_histo(x,bins,col,lab,percent=True,weights=weights,lw=lw,ls=ls)
                continue

            # ----------------------------------------------
            # 4. Marker plot
            if kwargs.has_key('fill'+str(i)):
                # print('>> Marker plot!')
                if kwargs['fill'+str(i)] == 'y':
                    ax1.plot(x,y,linestyle='None',color=col,marker=ma,mew=mew,ms=ms,label=lab,alpha=alpha)
                    continue
                if kwargs['fill'+str(i)] == 'n':
                    ax1.plot(x,y,linestyle='None',color=col,marker=ma,mew=mew,ms=ms,label=lab,fillstyle='none')
                    continue

            # ----------------------------------------------
            # 5. Bar plot
            if kwargs.has_key('barwidth'+str(i)):
                # print('>> Bar plot!')
                plt.bar(x,y,width=kwargs['barwidth'+str(i)],color=col,alpha=alpha)
                continue

            # ----------------------------------------------
            # 6. Hexbin contour bin
            if kwargs.has_key('hexbin'+str(i)):
                # print('>> Hexbin contour plot!')
                bins                =   300
                if kwargs.has_key('bins'+str(i)): bins = kwargs['bins'+str(i)]
                if kwargs.has_key('alpha'+str(i)): alpha = kwargs['alpha'+str(i)]
                if kwargs.has_key('col'+str(i)):
                    colors          =   kwargs['col'+str(i)]
                    CS              =   ax1.hexbin(x, y, C=colors, gridsize=bins, cmap=cmap, alpha=alpha)
                else:
                    CS              =   ax1.hexbin(x, y, gridsize=bins, cmap=cmap, alpha=alpha)
                continue

            # ----------------------------------------------
            # 7. Contour map

            if kwargs.has_key('contour_type'+str(i)):
                CS                  =   make_contour(i,fontsize,kwargs=kwargs)

            # ----------------------------------------------
            # 8. Scatter plot (colored according to a third parameter)
            if kwargs.has_key('scatter_color'+str(i)):
                # print('>> Scatter plot!')
                SC              =   ax1.scatter(x,y,marker=ma,lw=mew,s=ms,c=kwargs['scatter_color'+str(i)],cmap='viridis',alpha=alpha,label=lab,edgecolor=ecol)
                if kwargs.has_key('colormin'+str(i)): SC.set_clim(kwargs['colormin'+str(i)],max(kwargs['scatter_color'+str(i)]))
                if kwargs.has_key('lab_colorbar'):
                    if only_one_colorbar > 0:
                        cbar                    =   plt.colorbar(SC,pad=0)
                        cbar.set_label(label=kwargs['lab_colorbar'],size=fontsize-2)   # colorbar in it's own axis
                        cbar.ax.tick_params(labelsize=fontsize-2)
                        only_one_colorbar       =   -1
                continue

            # ----------------------------------------------
            # 8. Filled region
            if kwargs.has_key('hatchstyle'+str(i)):
                # print('>> Fill a region!')
                from matplotlib.patches import Ellipse, Polygon
                ax1.add_patch(Polygon([[x[0],y[0]],[x[0],y[1]],[x[1],y[1]],[x[1],y[0]]],closed=True,fill=False,hatch=kwargs['hatchstyle'+str(i)],color=col))
                if kwargs['hatchstyle'+str(i)] == '': ax1.fill_between(x,y[0],y[1],facecolor=col,color=col,alpha=alpha,lw=0)
                continue

    # Log or not log?
    if kwargs.has_key('xlog'):
        if kwargs['xlog']: ax1.set_xscale('log')
    if kwargs.has_key('ylog'):
        if kwargs['ylog']: ax1.set_yscale('log')

    # Legend
    if legend:
        legloc          =   [0.4,0.02]
        if kwargs.has_key('legloc'): legloc = kwargs['legloc']
        frameon         =   not kwargs.has_key('frameon') or kwargs['frameon']          # if "not" true that frameon is set, take frameon to kwargs['frameon'], otherwise always frameon=True
        handles1, labels1     =   ax1.get_legend_handles_labels()
        leg_fs          =   int(fontsize*0.7)
        if kwargs.has_key('leg_fs'): leg_fs = kwargs['leg_fs']
        ax1.legend(loc=legloc,fontsize=leg_fs,numpoints=1,scatterpoints = 1,frameon=frameon)

    # Add text to plot
    if kwargs.has_key('text'):
        textloc             =   [0.1,0.9]
        if kwargs.has_key('textloc'): textloc = kwargs['textloc']
        fontsize_text       =   fontsize
        if kwargs.has_key('textfs'): fontsize_text=kwargs['textfs']
        if kwargs.has_key('textbox'):
            ax1.text(textloc[0],textloc[1],kwargs['text'][0],\
                transform=ax1.transAxes,verticalalignment='top', horizontalalignment='right',fontsize=fontsize_text,\
                bbox=dict(facecolor='white', edgecolor='k', boxstyle='round,pad=1'))
        else:
            if len(kwargs['text']) == 1:
                ax1.text(textloc[0],textloc[1],kwargs['text'][0],color=textcol,\
                    transform=ax1.transAxes,verticalalignment='top', horizontalalignment='left',fontsize=fontsize_text)
            if len(kwargs['text']) > 1:
                for l,t in zip(textloc,kwargs['text']):
                    ax1.text(l[0],l[1],t,color='black',\
                        verticalalignment='top', horizontalalignment='left',fontsize=fontsize_text)

    if kwargs.has_key('grid'): ax1.grid()

    if kwargs.has_key('xticks'):
        if kwargs['xticks']:
            ax1.tick_params(labelbottom='off')
        else:
            ax1.set_xticks(kwargs['xticks'])
            ax1.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    if kwargs.has_key('xticklabels'): ax1.set_xticklabels(kwargs['xticklabels'])

    if kwargs.has_key('yticks'):
        if kwargs['yticks']:
            ax1.set_yticks([])
            ax1.tick_params(labelleft='off')
        else:
            ax1.set_yticks(kwargs['yticks'])
            ax1.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    if kwargs.has_key('yticklabels'): ax1.set_yticklabels(kwargs['yticklabels'])

    if kwargs.has_key('xr'): ax1.set_xlim(kwargs['xr'])
    if kwargs.has_key('yr'): ax1.set_ylim(kwargs['yr'])

    # plt.tight_layout()

    # Save plot if figure name is supplied
    if kwargs.has_key('figres'):
        dpi = kwargs['figres']
    else:
        dpi = 1000
    if kwargs.has_key('figname'):
        figname = kwargs['figname']
        figtype = 'png'
        if kwargs.has_key('figtype'): figtype = kwargs['figtype']
        plt.savefig(figname+'.'+figtype, format=figtype, dpi=dpi) # .eps for paper!

    # restoring defaults
    # mpl.rcParams['xtick.labelsize'] = u'medium'
    # mpl.rcParams['ytick.labelsize'] = u'medium'

    if kwargs.has_key('SC_return'):
        return SC

def make_histo(x,bins,col,lab,percent=True,weights=1,lw=1,ls='-'):
    '''
    Purpose
    ---------
    Makes a histogram (called by simple_plot)
    '''

    ax1             =   plt.gca()
    hist            =   np.histogram(x,bins,weights=weights)
    hist1           =   np.asarray(hist[0])
    hist2           =   np.asarray(hist[1])
    if percent: hist1           =   hist1*1./sum(hist1)*100.
    wid             =   (hist2[1]-hist2[0])
    # add some zeros to bring histogram down
    hist2           =   np.append([hist2],[hist2.max()+wid])
    hist2           =   np.append([hist2.min()-wid],[hist2])
    hist1           =   np.append([hist1],[0])
    hist1           =   np.append([0],[hist1])
    # plot it!
    ax1.plot(hist2[0:len(hist1)]+wid/2,hist1,ls='steps'+ls,color=col,label=lab,lw=lw)

def make_contour(i,fontsize,kwargs):
    '''
    Purpose
    ---------
    Makes contour plot (called by simple_plot)

    Arguments
    ---------
    contour_type: method used to create contour map - str
    options1:
        - plain: use contourf on colors alone, optionally with contour levels only (no filling)
        - hexbin: use hexbin on colors alone
        - median: use contourf on median of colors
        - mean: use contourf on mean of colors
        - sum: use contourf on sum of colors

    '''

    ax1                 =   plt.gca()

    linecol0            =   'k'
    cmap0               =   'viridis'
    alpha0              =   1.1
    nlev0               =   10
    only_one_colorbar   =   1

    # Put on regular grid!
    if kwargs.has_key('y'+str(i)):

        y               =   kwargs['y'+str(i)]
        x               =   kwargs['x'+str(i)]
        colors          =   kwargs['col'+str(i)]
        linecol         =   linecol0
        if kwargs.has_key('linecol'+str(i)): linecol = kwargs['linecol'+str(i)]
        cmap            =   cmap0
        if kwargs.has_key('cmap'+str(i)): cmap = kwargs['cmap'+str(i)]
        alpha           =   alpha0
        if kwargs.has_key('alpha'+str(i)): alpha = kwargs['alpha'+str(i)]
        nlev            =   nlev0
        if kwargs.has_key('nlev'+str(i)): nlev = kwargs['nlev'+str(i)]

        if kwargs['contour_type'+str(i)] == 'plain':

            if cmap == 'none':
                print('no cmap')
                CS = ax1.contour(x,y,colors, nlev, colors=linecol)
                plt.clabel(CS, fontsize=9, inline=1)

            if kwargs.has_key('colormin'+str(i)):
                print('Colormap with a minimum value')
                CS = ax1.contourf(x,y,colors, nlev, cmap=cmap, alpha=alpha, vmin=kwargs['colormin'+str(i)])

            else:
                if kwargs.has_key('alpha'+str(i)):
                    print('with alpha')
                    CS = ax1.contourf(x,y,colors, nlev, cmap=cmap, alpha=kwargs['alpha'+str(i)])
                if not kwargs.has_key('alpha'+str(i)):
                    print('without alpha')
                    CS = ax1.contourf(x,y,colors, nlev, cmap=cmap)#, lw=0, antialiased=True)

        if kwargs['contour_type'+str(i)] == 'hexbin':
            CS              =   ax1.hexbin(x, y, C=colors, cmap=cmap)

        if kwargs['contour_type'+str(i)] in ['median','mean','sum']:
            gridx           =   np.arange(min(x),max(x),kwargs['dx'+str(i)])
            gridy           =   np.arange(min(y),max(y),kwargs['dy'+str(i)])
            lx,ly           =   len(gridx),len(gridy)
            gridx1          =   np.append(gridx,max(gridx)+kwargs['dx'+str(i)])
            gridy1          =   np.append(gridy,max(gridy)+kwargs['dy'+str(i)])
            z               =   np.zeros([lx,ly])+min(colors)
            for i1 in range(0,lx):
                for i2 in range(0,ly):
                    # pdb.set_trace()
                    colors1         =   colors[(x > gridx1[i1]) & (x < gridx1[i1+1]) & (y > gridy1[i2]) & (y < gridy1[i2+1])]
                    if len(colors1) > 1:
                        if kwargs['contour_type'+str(i)] == 'median': z[i1,i2]       =   np.median(colors1)
                        if kwargs['contour_type'+str(i)] == 'mean': z[i1,i2]         =   np.mean(colors1)
                        if kwargs['contour_type'+str(i)] == 'sum': z[i1,i2]          =   np.sum(colors1)
            if kwargs.has_key('nlev'+str(i)): nlev0 = kwargs['nlev'+str(i)]
            CS               =   ax1.contourf(gridx, gridy, z.T, nlev0, cmap=cmap)
            mpl.rcParams['contour.negative_linestyle'] = 'solid'
            # CS               =   ax1.contour(gridx, gridy, z.T, 5, colors='k')
            # plt.clabel(CS, inline=1, fontsize=10)
            if kwargs.has_key('colormin'+str(i)): CS.set_clim(kwargs['colormin'+str(i)],max(z.reshape(lx*ly,1)))
            if kwargs.has_key('colormin'+str(i)): print(kwargs['colormin'+str(i)])
            CS.cmap.set_under('k')

    if kwargs.has_key('colorbar'+str(i)):
        if kwargs['colorbar'+str(i)]:
            if only_one_colorbar == 1: pad = 0
            if only_one_colorbar < 0: pad = 0.03
            cbar                    =   plt.colorbar(CS,pad=pad)
            cbar.set_label(label=kwargs['lab_colorbar'+str(i)],size=fontsize-2)   # colorbar in it's own axis
            cbar.ax.tick_params(labelsize=fontsize-2)
            only_one_colorbar       =   -1


    # plt.subplots_adjust(left=0.13, right=0.94, bottom=0.14, top=0.95)

    return CS

def histos(bins=100,galnames=galnames,col=col,add=False,one_color=True):
    '''
    Purpose
    ---------
    Makes figure with several histograms on top of each other.

    Arguments
    ---------
    bins: number of bins - float/int
    default = 100

    '''

    fs_labels       =   15
    mpl.rcParams['xtick.labelsize'] = fs_labels
    mpl.rcParams['ytick.labelsize'] = fs_labels

    # models          =   pd.read_pickle('sigame/temp/global_results/'+z1+'_'+str(len(galnames))+'gals'+ext_DENSE+ext_DIFFUSE)
    # galnames_sorted =   models['galname'].values
    nGal            =   len(galnames)

    # Ask a lot of questions!!
    foo1            =   raw_input('For which type of gas? [default: sim] '+\
                        '\n gmc for Giant Molecular Clouds'+\
                        '\n sim for raw simulation data'+\
                        '\n dng for Diffuse Neutral Gas'+\
                        '\n dig for Diffuse Ionized Gas'+\
                        '...?\n')
    if foo1 == '': foo1 =   'sim'
    if foo1 == 'sim':
        foo11        =   raw_input('gas or star? [default: gas] ... ')
        if foo11 == '': foo11 =   'gas'
    foo4            =   raw_input('over what quantity? [default: m]... ')
    if foo4 == '': foo4 =   'm'
    foo2            =   raw_input('mass or number-weighted (m vs n)? [default: n] ... ')
    if foo2 == '': foo2 =   'n'
    foo31           =   raw_input('logarithmix x-axis? [default: y] ... ')
    if foo31 == '': foo31 =   'y'
    foo32           =   raw_input('logarithmix y-axis? [default: y] ... ')
    if foo32 == '': foo32 =   'y'

    # Save which phase we're looking at, for plots:
    if foo1 == 'sim': phase = 'gas elements'
    if foo1 == 'DNG': phase = 'DNG'
    if foo1 == 'DIG': phase = 'DIG'
    if foo1 == 'gmc': phase = 'GMCs'
    if foo1 == 'halo': phase = 'halo'

    # Start plotting (fignum = 1: first plot)
    if not add:
        plt.close('all')
        plt.ion()
    redo        =   'y'
    fignum      =   1
    while redo == 'y':
        if add:
            print('adding to already existing figure')
            fig         =   plt.gcf()
            ax1         =   fig.add_subplot(add[0],add[1],add[2])
        else:
            print('creating new figure')
            fig         =   plt.figure(fignum,figsize=(8,6))
            ax1         =   fig.add_subplot(1,1,1)
        if fignum >1:
            foo4        =   raw_input('over what quantity? [default: m]... ')
            if foo4 == '': foo4 =   'm'
        # Get data
        igal        =   0
        histos1     =   np.zeros([len(galnames),bins+2])
        histos2     =   np.zeros([len(galnames),bins+3])
        for zred,galname in zip(zreds,galnames):
            if foo1 == 'halo':
                dat0 = pd.read_pickle('sigame/temp/halo_x_e_z6_'+galnames_sorted[0])
            if foo1 == 'sim':
                # if foo11 == 'gas': dat0 = pd.read_pickle('sigame/temp/sim/z'+'{:.2f}'.format(zred)+'_'+galname+'_sim0.gas')
                if foo11 == 'gas': dat0 = pd.read_pickle('sigame/temp/sim_FUV/z'+'{:.2f}'.format(zred)+'_'+galname+'_sim1.gas')
                if foo11 == 'star': dat0 = pd.read_pickle('sigame/temp/sim/z'+'{:.2f}'.format(zred)+'_'+galname+'_sim0.star')
                # if foo11 == 'star': dat0 = pd.read_pickle('sigame/temp/sim_FUV/z'+str(int(zred))+'_'+galname+'_sim1.star')
            if foo1 == 'gmc': dat0 = pd.read_pickle('sigame/temp/GMC/'+'z'+'{:.2f}'.format(zred)+'_'+galname+'_GMC.gas')
            if foo1 == 'DNG' or foo1 == 'DIG':
                dat0 = pd.read_pickle(dif_path+'z'+'{:.2f}'.format(zred)+'_'+galname+'_dif'+ext_DIFFUSE+'_em.gas')
                if foo1 == 'DNG': dat0 = dat0[dat0['m_DNG'] > dat0['m_DIG']]
                if foo1 == 'DIG': dat0 = dat0[dat0['m_DIG'] > dat0['m_DNG']]
            if foo4 == 'm_mol': dat0['m_mol'] = dat0['f_H2'].values*dat0['m'].values
            dat         =   dat0[foo4].values.squeeze()
            print('Percent of particles with value = 0: %s %%' % (100.*len(dat[dat == 0])/len(dat)))
            # print('median: ',np.median(dat))
            if foo2 == 'm': w           =   dat0['m']                   # default weights
            if foo2 == 'n': w           =   1./len(dat0)                   # default weights
            if foo1 == 'sim':
                if foo4 == 'nH': dat = dat/(mH*1000.)/1e6       # Hydrogen only per cm^3
            if foo31 == 'y':            # if x-axis is logarithmic, take log and remove NaNs from data and their weights
                if foo4 == 'Z': dat[dat == 0] = 1e-30 # to avoid crash if metallicity is zero
                # if foo4 == 'Z':
                    # print('TEST: 20 x Z')
                    # dat     =   dat*20. # to avoid crash if metallicity is zero
                dat = np.log10(dat)
                # remove data values = 0 !!
                i_nan   =   np.isnan(dat)
                if foo2 == 'm':  w       =   w[i_nan == False]
                dat     =   dat[i_nan == False]
            print('min and max: %s and %s ' % (np.min(dat[dat > -100]),dat.max()))
            # remove data values = -inf !!
            if foo31 == 'n':
                if foo2 == 'm':  w       =   w[dat > -10.**(20)]
                dat     =   dat[dat > -10.**(20)]
            if foo31 == 'y':
                if foo2 == 'm':  w       =   w[dat > -20]
                dat     =   dat[dat > -20]
            if foo2 == 'n':    hist        =   np.histogram(dat,bins=bins)
            if foo2 == 'm':      hist        =   np.histogram(dat,bins=bins,weights=w)
            if foo4 == 'f_HI':
                print('Particles are above 0.9: %s %%' % (1.*len(dat[dat > 0.9])/len(dat)*100.))
                print('Particles are below 0.1: %s %%' % (1.*len(dat[dat < 0.1])/len(dat)*100.))
            if foo4 == 'f_H2':
                print('Particles are above 0.9: %s %%' % (1.*len(dat[dat > 0.9])/len(dat)*100.))
                print('Particles are below 0.1: %s %%' % (1.*len(dat[dat < 0.1])/len(dat)*100.))
            hist1            =  np.asarray(hist[0])
            hist2            =  np.asarray(hist[1])
            hist1            =  hist1*1./sum(hist1)*100.
            wid              =  (hist2[1]-hist2[0])
            # add some zeros to bring histogram down
            hist2            =  np.append([hist2],[hist2.max()+wid])
            hist2            =  np.append([hist2.min()-wid],[hist2])
            hist1            =  np.append([hist1],[0])
            hist1            =  np.append([0],[hist1])
            histos1[igal,:]  =   hist1
            histos2[igal,:]  =   hist2

            if not one_color:
                ax1.plot(hist2[0:len(hist1)]+wid/2,hist1,ls='steps',color=col[igal],label='G'+str(int(igal+1)))

            igal             +=  1

        if one_color:
            # Plot as background the 2 sigma distribution around the mean in each bin
            minhistos1,maxhistos1,meanhistos1       =   np.zeros(bins+2), np.zeros(bins+2), np.zeros(bins+2)
            for i in range(0,bins+2):
                meanhistos1[i]     =    np.mean(histos1[:,i])
                minhistos1[i]      =   meanhistos1[i]-2.*np.std(histos1[:,i])
                maxhistos1[i]      =   meanhistos1[i]+2.*np.std(histos1[:,i])
            ax1.fill_between(hist2[0:len(hist1)]+wid/2, minhistos1, maxhistos1, facecolor='lightgreen', alpha=0.5, lw=0)

            # Now plot actual histograms
            for galname,i in zip(galnames,range(0,len(galnames))):
                hist2           =   histos2[i,:]
                hist1           =   histos1[i,:]
                ax1.plot(hist2[0:len(hist1)]+wid/2,hist1,ls='steps',color='teal',label='G'+str(int(i+1)),alpha=0.7,lw=1)

            # Now plot mean of histograms
            if len(galnames) > 1: ax1.plot(hist2[0:len(hist1)]+wid/2,meanhistos1,ls='steps',color='blue',lw=1)
        if foo32 == 'y':     ax1.set_yscale('log')

        # labels and ranges
        xl          =   getlabel(foo4)
        if foo31    == 'y': xl = getlabel('l'+foo4)
        ax1.set_xlabel(xl,fontsize=fs_labels)
        if foo2     == 'n': ax1.set_ylabel('Number fraction [%]',fontsize=fs_labels)
        if foo2     == 'm': ax1.set_ylabel('Mass fraction [%]',fontsize=fs_labels)
        # leg    =   ax1.legend(fontsize=10,loc=[0.03,0.7],ncol=5,handlelength=2)
        ax1.set_ylim([max(hist1)/1e4,max(hist1)*10.])
        if not add:
            if foo1 == 'halo': ax1.set_title('halo mass: '+str.format("{0:.2f}",sum(dat0['m'].values)/1e10)+' x10^10 Msun')
            fig.canvas.draw()

        # legends
        foo6        =   raw_input('Change x limits? [default: n] ... ')
        if foo6 == '': foo6 =   'n'
        if foo6 == 'y':
            x1          =   raw_input('Lower x lim (in log if used): ')
            x2          =   raw_input('Upper x lim (in log if used): ')
            ax1.set_xlim([float(x1),float(x2)])
        foo7        =   raw_input('Change y limits? [default: n] ... ')
        if foo7 == '': foo7 =   'n'
        if foo7 == 'y':
            y1          =   raw_input('Lower y lim (in log if used): ')
            y2          =   raw_input('Upper y lim (in log if used): ')
            ax1.set_ylim([10.**float(y1),10.**float(y2)])
        fig.canvas.draw()
        # mv          =   raw_input('move legend to the right? [default: n] ... ')
        # if mv == '': mv='n'
        # if mv == 'y':
        #     leg.remove()
        #     ax1.legend(fontsize=10,loc=[0.8,0.45],ncol=5,handlelength=2)
        # fig.canvas.draw()

        savefig         =   raw_input('Save figure? [default: n] ... ')
        if savefig == '': savefig = 'n'
        if savefig == 'y':
            name            =   raw_input('Figure name? ... ')
            plt.savefig(name+'.png', format='png', dpi=250) # .eps for paper!

        pdb.set_trace()

        # New figure?
        if add:
            redo = 'n'
        else:
            redo        =   raw_input('plot another quantity? [default: n] ... ')
            if redo == '': redo='n'
            if redo == 'n':
                # restoring defaults
                mpl.rcParams['xtick.labelsize'] = u'medium'
                mpl.rcParams['ytick.labelsize'] = u'medium'
                break
            fignum      +=1
            foo6, foo7  =   'n','n'

#===============================================================================
""" Global gas properties in simulation """
#-------------------------------------------------------------------------------

def SFR_Mstar():
    '''
    Purpose
    ---------
    Plot of SFR vs stellar mass
    '''

    plt.close('all')

    models                      =   aux.find_model_results()
    for key in models.keys():
        exec(key + '=models[key].values')

    # Mass-weighted ages
    ages            =   np.zeros(len(models))
    i               =   0
    for galname,zred in zip(galnames,zreds):
        simstar         =   pd.read_pickle(d_sim+'z'+'{:.2f}'.format(zred)+'_'+galname+'_sim0.star')
        ages[i]         =   np.sum(simstar['age'].values*simstar['m'])/np.sum(simstar['m'])
        i               +=  1

    xr              =   np.array([min(M_star)/2.,max(M_star)*1.5])#/1e9
    xr              =   10.**np.array([7.8,10.8])
    yr              =   [min(SFR)/2.,max(SFR)*1.5]
    yr              =   10.**np.array([0.3,2.8])

    if z1 == 'z6':
	age = 0.984                           # age of universe, Omega_m = 0.3, Omega_lambda = 0.7, h = 0.65
    	MS              =   10.**((0.84-0.026*age)*np.log10(xr)-(6.51-0.11*age))
    	# z ~ 6 LBGs and LAEs
    	columns         =   ['ID','age(M)','mass(G)','SFR_Lya','SFR_UV','E(B-V)']
    	J16             =   pd.read_table(d_t+'Observations/SFR_Mstar/Jiang16.txt',names=columns,skiprows=1,sep=r'\s*',engine='python')
    	SFR_J16         =   J16['SFR_UV'].values
    	No_J16          =   J16['ID'].values
    	ebv             =   J16['E(B-V)'].values
    	age_J16         =   J16['age(M)'].values
    	M_star_J13      =   np.array([7.2,3.4,21.1,32.8,57.9,3.8,66.5,17.4,2.7,26.8,2.4,250,81.1,46.4,30.8,172.9,391.1,35.6,9.0,51.5,61.6,66.3,115,9.4,41.3,1.4,32])*1e8
    	No_J13          =   np.array([3,4,15,20,23,24,25,27,28,30,31,34,35,36,43,44,47,49,50,54,58,61,62,63,64,66,67])
    	SFR_J16         =   [float(SFR_J16[No_J16 == No][0]) for No in No_J13]
    	# From Linhua Jiang
    	klambda         =   8.5  #at 2200A
    	SFR_J16         =   SFR_J16 * 10.**(0.4*klambda*ebv)

    simple_plot(fignum=0,xlog='y',ylog='y',xr=xr,yr=yr,fontsize=16,\
        xlab='M$_{\mathrm{*}}$ [M$_{\odot}$]',ylab='SFR [M$_{\odot}$ yr$^{-1}$]',legloc=[0.01,0.75],frameon=False,\
        x1=1,y1=1,ma1='o',col1='lightseagreen',fill1='y',ms1=8,mew1=2,lab1='Model galaxies (this work)$_{}$',\
        x2=xr,y2=MS,col2='k',lw2=1,lab2='$z\sim6$ main sequence [Speagle+14]$_{}$',ls2='-',\
        x3=xr,y3=10.**(np.log10(MS)-0.2),col3='k',lw3=1,ls3='--',\
        x4=xr,y4=10.**(np.log10(MS)+0.2),col4='k',lw4=1,ls4='--',\
        x5=xr,y5=10.**(np.log10(MS)-3.*0.2),col5='k',lw5=1,ls5=':',\
        x6=xr,y6=10.**(np.log10(MS)+3.*0.2),col6='k',lw6=1,ls6=':',legend=False)

    ax1             =   plt.gca()
    if z1 == 'z6':
	simple_plot(add='y',x1=M_star_J13[age_J16 > 30],y1=SFR_J16[age_J16 > 30],fill1='y',ma1='x',col1='r',ms1=10,mew1=2,lab1='Old $z\sim6$ LBGs/LAEs [Jiang+16]',\
			x2=M_star_J13[age_J16 < 30],y2=SFR_J16[age_J16 < 30],fill2='y',ma2='+',col2='b',ms2=11,mew2=2,lab2='Young $z\sim6$ LBGs/LAEs [Jiang+16]',legloc='upper left')

    # Plot models
    SC = ax1.scatter(M_star,SFR,marker='o',lw=2,s=64,c=ages,edgecolor='black',cmap='viridis',alpha=1,label='',zorder=10)
    cbar            =   plt.colorbar(SC,pad=0)
    cbar.set_label(label='Mass-weighted stellar age [Myr]')   # colorbar in it's own axis

    plt.tight_layout()
    plt.show(block=False)
    plt.savefig('plots/galaxy_sims/SFR_Mstar/M_SFR_sample_'+z1+'.eps', format='eps', dpi=1000) # .eps for paper!

#===============================================================================
""" Line emission plotting """
#-------------------------------------------------------------------------------

def CII_SFR_z6(plot_models=True,plot_fits=True,plot_obs=True,twopanel=True,obs_grey=True,mark_reasons=False,mark_uplim=False,mark_det=False,legend=True):
    '''
    Purpose
    ---------
    Plots L_[CII] and SFR together with observations at z~6
    '''

    plt.close('all')        # close all windows

    models                      =   aux.find_model_results()
    for key in models.keys():
        exec(key + '=models[key].values')
    L_CII_tot           =   L_CII_GMC+L_CII_DNG+L_CII_DIG

    z1                  =   'z6'

    # Plotting parameters
    mpl.rcParams['xtick.minor.size'] = 4
    mpl.rcParams['xtick.major.size'] = 6
    mpl.rcParams['xtick.minor.width'] = 1.5
    mpl.rcParams['xtick.major.width'] = 1.5
    fs_labels = 20
    if twopanel: fs_labels = 17
    # Change y or x range here
    xr                  =   10.**np.array([0,3.05])               # SFR range
    yr                  =   10.**np.array([6.5,9.8])              # [CII] range
    if twopanel:
        fig = plt.figure(figsize = (18,8))
        gs = gridspec.GridSpec(1, 2, width_ratios=[6, 5])
        ax1 = plt.subplot(gs[1])
    if not twopanel:
        fig = plt.figure(figsize = (12,9))
        ax1 = fig.add_subplot(1,1,1)
    ax1.set_xlabel('SFR [M$_{\odot}$ yr$^{-1}$]',fontsize=fs_labels)
    ax1.set_ylabel('L$_{[\mathrm{CII}]}$ [L$_{\odot}$]',fontsize=fs_labels)
    ax1.set_xlim(xr)
    ax1.set_ylim(yr)
    xlab                =   ax1.get_xticks()
    ylab                =   ax1.get_yticks()
    ax1.set_xticklabels(xlab,fontsize=fs_labels)
    ax1.set_yticklabels(ylab,fontsize=fs_labels)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.xaxis.set_major_formatter(ScalarFormatter())
    lw                  =   2           # line width

    # ----------------------------------------------------------------------
    # Models!
    # ======================================================================

    # Powerlaw fit to z=0 metal-poor dwarf galaxies [de Looze et al. 2014]
    logL_CII            =  np.array([4,11])
    logSFR              =  -5.73+0.80*logL_CII
    ax1.fill_between(10.**logSFR, 10.**(logL_CII-0.37), 10.**(logL_CII+0.37), facecolor='lightgrey', linewidth=0, alpha=0.5)
    fit0                =   ax1.plot(10.**logSFR,10.**logL_CII,c='grey',ls='--',lw=lw)
    # Powerlaw fit to z=0 starburst galaxies [de Looze et al. 2014]
    logL_CII            =  np.array([4,11])
    logSFR              =  -7.06+1.00*logL_CII
    ax1.fill_between(10.**logSFR, 10.**(logL_CII-0.27), 10.**(logL_CII+0.27), facecolor='lightgrey', linewidth=0, alpha=0.5)
    fit1                =   ax1.plot(10.**logSFR,10.**logL_CII,c='grey',ls='--',dashes=(1,4),lw=lw)

    # Powerlaw fit to model galaxies at z=6
    slope_models,intercept_models,slope_dev,inter_dev = aux.lin_reg_boot(np.log10(SFR),np.log10(L_CII_tot))

    # Making different line styles for the SFR-range covered by the simulations and 
    # the SFR-range where an extrapolation is necessary
    L_CII_model         =   10.**(slope_models*np.log10(xr)+intercept_models)
    xr2                 =   [min(SFR),max(SFR)]
    L_CII_model2        =   10.**(slope_models*np.log10(xr2)+intercept_models)
    # dex spread around relation:
    dex                 =   np.std(np.abs(np.log10(L_CII_tot)-(slope_models*np.log10(SFR)+intercept_models)))
    print('Dex: %s' % dex)
    if not twopanel:
        if plot_models:
                fit = ax1.plot(xr,L_CII_model,ls='--',color='purple',lw=1.8)
                fit = ax1.plot(xr2,L_CII_model2,ls='-',color='purple',lw=1.8)
                fit = ax1.plot(xr,L_CII_model,ls='--',color='purple',lw=1.8)
                fit = ax1.plot(xr2,L_CII_model2,ls='-',color='purple',lw=1.8)
    if plot_fits:
        # Powerlaw fit to Vallini+15, z~7 eq. 8
        sims                =   cl.sim(sim_paths)
        Zfit                =   np.mean(sims.Zmw())
        L_CII_V15           =   10.**(7.0 + 1.2*np.log10(xr) + 0.021*np.log10(Zfit) + \
                                0.012*np.log10(xr)*np.log10(Zfit) - 0.74*np.log10(Zfit)**2.)
        fit2                =   ax1.plot(xr,L_CII_V15,c='orange',ls='--',dashes=(6,4,1,4),lw=lw)
        print('Plotting Vallini at Z = '+str(Zfit))
        print('Slope of Vallini: '+str(1.2+0.012*np.log10(Zfit)))

        # PCA fit to L_[CII](SFR,Z)
        Zfit2               =   np.mean(Zsfr)
        L_CII_PCA           =   10.**(7.17+0.55*np.log10(xr)+0.23*np.log10(Zfit2))
        ax1.plot(xr,L_CII_PCA,c='purple',ls='--',dashes=(6,4,1,4,1,4),lw=lw)

        if len(galnames) > 1: 
            ax1.fill_between(xr, 10.**(np.log10(L_CII_model)-dex), 10.**(np.log10(L_CII_model)+dex),facecolor='lightgreen', linewidth=0, alpha=0.5)
            fit = ax1.plot(xr,L_CII_model,ls='--',color='purple',lw=1.8)
            fit = ax1.plot(xr2,L_CII_model2,ls='-',color='purple',lw=1.8)
            fit = ax1.plot(xr,L_CII_model,ls='--',color='purple',lw=1.8)
            fit = ax1.plot(xr2,L_CII_model2,ls='-',color='purple',lw=1.8)
        print('Slope of linear fit to z=6 galaxies: '+str(slope_models))
        print('and intercept: '+str(intercept_models))
        chi2_models         =   chisquare(slope_models*np.log10(SFR)+intercept_models, f_exp=np.log10(L_CII_tot))
        print('with chi^s: '+str(chi2_models[0]))
        dev                 =   np.log10(L_CII_tot) - (slope_models*np.log10(SFR)+intercept_models)
        rms_dex             =   np.sqrt(np.sum(dev**2./len(dev)))
        print('rms error of: '+str(rms_dex)+' dex')

        if len(galnames) > 1: 
            print('On average this far below observed relation:')
            L_CII_obs           =   10.**((np.log10(SFR)+7.06)/1.00)
            print(str(np.mean((L_CII_obs-L_CII_tot)/L_CII_obs)*100.)+' %')
            print(str(np.mean(np.log10(L_CII_obs)-np.log10(L_CII_tot)))+' dex')
            print('Power law fit this far above Vallini+15:')
            L_CII_V15           =   10.**(7.0 + 1.2*np.log10(min(SFR)) + 0.021*np.log10(Zfit) + \
                                    0.012*np.log10(min(SFR))*np.log10(Zfit) - 0.74*np.log10(Zfit)**2.)
            L_CII                =   10.**(slope_models*np.log10(min(SFR))+intercept_models)
            print('at min SFR : '+str(np.log10(L_CII)-np.log10(L_CII_V15))+' dex')
            L_CII_V15           =   10.**(7.0 + 1.2*np.log10(max(SFR)) + 0.021*np.log10(Zfit) + \
                                    0.012*np.log10(max(SFR))*np.log10(Zfit) - 0.74*np.log10(Zfit)**2.)
            L_CII                =   10.**(slope_models*np.log10(max(SFR))+intercept_models)
            print('at max SFR : '+str(np.log10(L_CII)-np.log10(L_CII_V15))+' dex')

        # Powerlaw fit to model galaxies at z=2 [Olsen+15]
        L_CII_model_z2      =   0.78e7*xr**1.27
        fit = ax1.plot(xr,L_CII_model_z2,'--k',lw=1.8,dashes=(15,10))

        # Legend for fits
        if plot_fits:
            ax1.plot(10.**logSFR, 10.**(logL_CII-10), '--',dashes=(1,4), lw=1.8, c='grey', label = u'Local starburst galaxies, [De Looze et al. 2014] $_{}$')
            ax1.plot(10.**logSFR, 10.**(logL_CII-10), '--', lw=1.8, c='grey', label = u'Local metal-poor dwarf galaxies, [De Looze et al. 2014] $_{}$')
            ax1.plot([1e20,1e30],[1e20,1e30],'--',dashes=(6,4,1,4),lw=1.8,c='orange',label=u'$z\sim7$ models with $Z$ = '+str.format("{0:.2f}",Zfit)+' [Vallini+15]')
            ax1.plot([1e20,1e30],[1e20,1e30],'--k',lw=1.8,dashes=(15,10),label = u'$z=2$ models [Olsen+15] $_{}$')#: log(L$_{\mathrm{[CII]}}$) = '+str.format("{0:.2f}",1.27)+'$\cdot$log(SFR) + ' + str.format("{0:.2f}",np.log10(0.78e7))+' [Olsen+15]')
            # ax1.plot([1e20,1e30],[1e20,1e30],'purple',lw=lw,dashes=(6,4,1,4,1,4),label = u'$z=6$ models, $Z$ = '+str.format("{0:.2f}",Zfit2)+' (this work)' )
            if len(galnames) > 1: ax1.plot([1e20,1e30],[1e20,1e30],'purple',lw=1.8,label = u'$z=6$ models: log(L$_{\mathrm{[CII]}}$) = '+str.format("{0:.2f}",slope_models)+'$\cdot$log(SFR) + ' + str.format("{0:.2f}" + ' (this work)',intercept_models))
            handles1, labels1     =   ax1.get_legend_handles_labels()
        if legend:
            ax1.legend(handles1,labels1,loc='upper left',fontsize=11,frameon=True,\
                numpoints=1,scatterpoints=1,handlelength=3)

    # ----------------------------------------------------------------------
    # Observations!
    # ======================================================================
    if twopanel:
        ax1 = plt.subplot(gs[0])
        ax1.set_xlim(xr)
        ax1.set_ylim(yr)
        ax1.set_ylabel('L$_{[\mathrm{CII}]}$ [L$_{\odot}$]',fontsize=fs_labels)
        ax1.set_xlabel('SFR [M$_{\odot}$ yr$^{-1}$]',fontsize=fs_labels)
        xlab                =   ax1.get_xticks()
        ylab                =   ax1.get_yticks()
        ax1.set_xticklabels(xlab,fontsize=fs_labels)
        ax1.set_yticklabels(ylab,fontsize=fs_labels)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        if plot_models:
            if len(galnames) > 1: 
                ax1.fill_between(xr, 10.**(np.log10(L_CII_model)-dex), 10.**(np.log10(L_CII_model)+dex),facecolor='lightgreen', linewidth=0, alpha=0.5)
                fit = ax1.plot(xr,L_CII_model,ls='--',color='purple',lw=1.8)
                fit = ax1.plot(xr2,L_CII_model2,ls='-',color='purple',lw=1.8)
        # else:
        # Powerlaw fit to z=0 metal-poor dwarf galaxies
        logL_CII            =  np.array([4,11])
        logSFR              =  -5.73+0.80*logL_CII
        ax1.fill_between(10.**logSFR, 10.**(logL_CII-0.37), 10.**(logL_CII+0.37), facecolor='lightgrey', linewidth=0, alpha=0.5,zorder=0)
        fit0                =   ax1.plot(10.**logSFR,10.**logL_CII,c='grey',ls='--',lw=lw,zorder=0)
        # Powerlaw fit to z=0 starburst galaxies
        logL_CII            =  np.array([4,11])
        logSFR              =  -7.06+1.00*logL_CII
        ax1.fill_between(10.**logSFR, 10.**(logL_CII-0.27), 10.**(logL_CII+0.27), facecolor='lightgrey', linewidth=0, alpha=0.5,zorder=0)
        fit1                =   ax1.plot(10.**logSFR,10.**logL_CII,c='grey',ls='--',dashes=(1,4),lw=lw,zorder=0)

    if plot_obs: add_observations_to_plot(slope_models,intercept_models,mark_reasons,mark_uplim,mark_det,z1=z1)

    # Legends for observations!
    if plot_models: ax1.scatter([1e20,1e30],[1e20,1e30],marker='o',s=64,color='lightseagreen',label=r'S$\mathrm{\'I}$GAME (this work)$_{}$',lw=2,edgecolor='black')
    handles2, labels2 = ax1.get_legend_handles_labels()
    # remove the errorbars
    handles2 = [h[0] if isinstance(h, mpl.container.ErrorbarContainer) else h for h in handles2]
    # Re-order:
    if plot_models: handles2 = [handles2[i] for i in [14,0,1,2,3,4,5,6,7,8,9,10,11,12,13,15]]
    if plot_models: labels2 = [labels2[i] for i in [14,0,1,2,3,4,5,6,7,8,9,10,11,12,13,15]]
    # use them in the legend
    if legend:
        ax1.legend(handles2, labels2, loc='upper left',fontsize=11,frameon=False,numpoints=1,borderpad=1)

    # Model galaxies
    ax1.xaxis.set_major_formatter(ScalarFormatter())
    if plot_models:
        plot = ax1.scatter(SFR,L_CII_tot,marker='o',edgecolor='black',c=np.log10(SFRsd),s=64,lw=2,cmap='viridis',zorder=200)
        cbar = fig.colorbar(plot,ax=ax1,pad=0)#,cax = cbar_ax)
        cbar.set_label(label=getlabel('lSFRsd'),size=13)#,weight='bold')

    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(1.)

    plt.subplots_adjust(left=0.07, right=0.97, bottom=0.1, top=0.97, wspace=0.2)
    if not twopanel:
        plt.subplots_adjust(left=0.12, right=0.95, bottom=0.1, top=0.95, wspace=0.2)
        if not plot_models:
            plt.subplots_adjust(left=0.12, right=0.825, bottom=0.1, top=0.95, wspace=0.13)

    plt.show(block=False)

    if twopanel: plt.savefig('plots/line_emission/line_SFR/CII_SFR/CII_SFR_'+z1+ext_DENSE+ext_DIFFUSE+'.pdf', format='pdf') # .eps for paper!

def OI_OIII_SFR(plotmodels=True,twopanel=True):
    '''
    Purpose
    ---------
    Inspect [OI]63micron AND [OIII]88micron emission (called by line_SFR() in analysis.py)
    '''

    lines                   =   ['[OI]','[OIII]','[CII]']
    wavelengths             =   np.array([63,88,122,158])

    models                      =   aux.find_model_results()
    for key in models.keys():
        exec(key + '=models[key].values')

    L_OIII                  =   L_OIII_GMC+L_OIII_DNG+L_OIII_DIG
    L_OI                    =   L_OI_GMC+L_OI_DNG+L_OI_DIG
    L_CII                   =   L_CII_GMC+L_CII_DNG+L_CII_DIG

    # L_GMCs,L_DNGs,L_DIGs,L_tots       =   [np.zeros([len(lines),len(galnames)]) for i in range(0,4)]

    print('L_OI/L_CII:')
    print('min: %s max: %s' % (np.min(L_CII/L_OI),np.max(L_CII/L_OI)))
    print('mean: %s' % np.mean(L_CII/L_OI))
    print('GMC fraction of L_[OI]:')
    print('min: %s max: %s' % (np.min(models['L_OI_GMC'].values/L_OI),np.max(models['L_OI_GMC'].values/L_OI)))
    print('mean: %s' % np.mean(models['L_OI_GMC'].values/L_OI))

    print('L_OIII/L_CII:')
    print('min: %s max: %s' % (np.min(L_CII/L_OIII),np.max(L_CII/L_OIII)))
    print('mean: %s' % np.mean(L_CII/L_OIII))
    print('GMC fraction of L_[OIII]:')
    print('min: %s max: %s' % (np.min(models['L_OIII_GMC'].values/L_OIII),np.max(models['L_OIII_GMC'].values/L_OIII)))
    print('mean: %s' % np.mean(models['L_OIII_GMC'].values/L_OIII))
    print('DIG fraction of L_[OIII]:')
    print('min: %s max: %s' % (np.min(models['L_OIII_DIG'].values/L_OIII),np.max(models['L_OIII_DIG'].values/L_OIII)))
    print('mean: %s' % np.mean(models['L_OIII_DIG'].values/L_OIII))

    plt.close('all')

    print('Make two panel plot')
    fig                     =   plt.figure(1,figsize = (6,14))
    ax1                     =   fig.add_subplot(2,1,1)
    xr                      =   axis_range(SFR,log='y',dex=1.5)
    xr                      =   [1,xr[1]]
    yr                      =   axis_range(L_OI,log='y',dex=1.5)
    yr                      =   [10**7.5,yr[1]]
    # Add 1 sigma dispersion around fit from De Looze 2014
    L_OI_L14_DG,dex         =   (np.log10(xr) + 6.23)/0.91,0.27
    ax1.fill_between(xr, 10.**(L_OI_L14_DG-dex), 10.**(L_OI_L14_DG+dex), facecolor='grey', linewidth=0, alpha=0.5)
    L_OI_L14_SB,dex         =   (np.log10(xr) + 6.05)/0.89,0.20
    ax1.fill_between(xr, 10.**(L_OI_L14_SB-dex), 10.**(L_OI_L14_SB+dex), facecolor='grey', linewidth=0, alpha=0.5)
    # Fit to our models:
    slope,intercept,r_value,p_value,std_err     =   scipy.stats.linregress(np.log10(SFR),np.log10(L_OI))
    powlaw                  =   10.**(slope*np.log10(xr)+intercept)
    simple_plot(add='y',xr=xr,yr=yr, ylog='y',xlog='y',\
        x1=xr, y1=10.**L_OI_L14_DG, ls1='--', lw1=2,  col1='grey', lab1=u'Local metal-poor dwarf galaxies, [De Looze et al. 2014] $_{}$',\
        x2=xr, y2=10.**L_OI_L14_SB, ls2=':', dashes2=(1,4), lw2=2,  col2='grey', lab2=u'Local starburst galaxies, [De Looze et al. 2014] $_{}$',\
        # x3=xr, y3=powlaw, ls3='-', lw3=2,  col3='black', \
        x4=SFR, y4=L_OI, ma4='o', scatter_color4='lightseagreen', mew4=2, ms4=64,\
        xlab='',xticks='n',ylab=getlabel('L_OI'),\
        legloc=[0.04,0.8])
    # pdb.set_trace()
    ax1                     =   fig.add_subplot(2,1,2)
    yr                      =   axis_range(L_OIII,log='y',dex=1)
    yr                      =   [10**6.3,yr[1]]
    L_OIII_L14_DG,dex       =   (np.log10(xr) + 6.71)/0.92,0.30
    ax1.fill_between(xr, 10.**(L_OIII_L14_DG-dex), 10.**(L_OIII_L14_DG+dex), facecolor='grey', linewidth=0, alpha=0.5)
    L_OIII_L14_SB,dex       =   (np.log10(xr) + 3.89)/0.69,0.23
    ax1.fill_between(xr, 10.**(L_OIII_L14_SB-dex), 10.**(L_OIII_L14_SB+dex), facecolor='grey', linewidth=0, alpha=0.5)
    # Fit to our models:
    if len(galnames) > 1:
        slope,intercept,slope_dev,inter_dev = aux.lin_reg_boot(np.log10(SFR),np.log10(L_OIII))
        powlaw                  =   10.**(slope*np.log10(xr)+intercept)
        xr1                     =   [np.min(models['SFR'].values),np.max(models['SFR'].values)]
        powlaw1                 =   10.**(slope*np.log10(xr1)+intercept)
        dex                     =   np.std(np.abs(np.log10(L_OIII)-(slope*np.log10(SFR)+intercept)))

    # Observation by Inoue+16
    L_OIII_I16              =   np.array([9.898411,2.083876,2.083876])*1e8 # magnification-corrected (~2)
    SFR_I16                 =   np.array([10.**2.54,10.**2.54-10.**(2.54-0.71),10.**2.54+10.**(2.54+0.17)]) # SED
    # Observation by Laporte+17
    L_OIII_L17              =   np.array([1.4,0.35,0.35])*1e8/2. # magnification-corrected (~2)
    SFR_L17                 =   np.array([20.4,9.5,17.6]) # SED
    # Observation by Carniani+17
    L_OIII_C17              =   np.array([1.8,0.2,0.2])*1e8 # clump I
    SFR_C17                 =   np.array([7]) # from comparing models to [OIII] detection [C17]

    simple_plot(add='y',legend='off',\
        x3=[1e6],y3=[1e6],ma3='^',fill3='y',lw3=1.2, ms3=4,lab3='$z=7.21$ galaxy [Inoue et al. 2016]',\
        x4=[1e6],y4=[1e6],ma4='s',fill4='y',lw4=1.2, ms4=4,lab4='$z=8.38$ galaxy [Laporte et al. 2017]',\
        x5=[1e6],y5=[1e6],ma5='d',fill5='y',lw5=1.2, ms5=4,lab5='$z=7.107$ galaxy [Carniani et al. 2017]')


    if len(galnames) > 1:
        SC = simple_plot(add='y',xr=xr,yr=yr, ylog='y',xlog='y',\
            x1=xr, y1=10.**L_OIII_L14_DG, ls1='--', lw1=2, col1='grey',\
            x2=xr, y2=10.**L_OIII_L14_SB, ls2=':', dashes2=(1,4), lw2=2, col2='grey',\
            x3=[SFR_I16[0]], y3=[L_OIII_I16[0]], ma3='^', lex3=[SFR_I16[1]], uex3=[SFR_I16[2]], \
            ley3=[L_OIII_I16[1]], uey3=[L_OIII_I16[2]], lw3=1.2, ms3=3,lab3='', \
            x4=[SFR_L17[0]], y4=[L_OIII_L17[0]], ma4='s', lex4=[SFR_L17[1]], uex4=[SFR_L17[2]], \
            ley4=[L_OIII_L17[1]], uey4=[L_OIII_L17[2]], lw4=1.2, ms4=2,lab4='',\
            x5=[SFR_C17[0]], y5=[L_OIII_C17[0]], ma5='d',fill5='y',lab5='',\
            lex5=[0.4], uex5=[0], ley5=[L_OIII_C17[1]], uey5=[L_OIII_C17[2]], lw5=1.2, ms5=2,\
            x6=xr, y6=[10.**(np.log10(powlaw)-dex), 10.**(np.log10(powlaw)+dex)], hatchstyle6='',col6='lightgreen',  alpha6=0.5,\
            x7=xr, y7=powlaw, ls7='--', lw7=2,  col7='purple',
            x8=xr1, y8=powlaw1, ls8='-', lw8=2,  col8='purple',
            x9=SFR, y9=L_OIII, ma9='o', scatter_color9='lightseagreen', mew9=2, ms9=64,\
            xlab=getlabel('SFR'),ylab=getlabel('L_OIII'),legloc='lower right',legend='on') # [0.05,0.8]
    else:
        SC = simple_plot(add='y',xr=xr,yr=yr, ylog='y',xlog='y',\
            x1=xr, y1=10.**L_OIII_L14_DG, ls1='--', lw1=2, col1='grey',\
            x2=xr, y2=10.**L_OIII_L14_SB, ls2=':', dashes2=(1,4), lw2=2, col2='grey',\
            x3=[SFR_I16[0]], y3=[L_OIII_I16[0]], ma3='^', lex3=[SFR_I16[1]], uex3=[SFR_I16[2]], \
            ley3=[L_OIII_I16[1]], uey3=[L_OIII_I16[2]], lw3=1.2, ms3=3,lab3='', \
            x4=[SFR_L17[0]], y4=[L_OIII_L17[0]], ma4='s', lex4=[SFR_L17[1]], uex4=[SFR_L17[2]], \
            ley4=[L_OIII_L17[1]], uey4=[L_OIII_L17[2]], lw4=1.2, ms4=2,lab4='',\
            x5=[SFR_C17[0]], y5=[L_OIII_C17[0]], ma5='d',fill5='y',lab5='',\
            lex5=[0.4], uex5=[0], ley5=[L_OIII_C17[1]], uey5=[L_OIII_C17[2]], lw5=1.2, ms5=2,\
            x9=SFR, y9=L_OIII, ma9='o', scatter_color9='lightseagreen', mew9=2, ms9=64,\
            xlab=getlabel('SFR'),ylab=getlabel('L_OIII'),legloc='lower right',legend='on') # [0.05,0.8]

    ax1                 =   plt.gca()

    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.13, top=0.95, hspace=0., wspace=0.)
    # plt.tight_layout()
    plt.show(block=False)
    plt.savefig('plots/line_emission/line_SFR/comparison/OI_OIII_SFR_'+z1+'.pdf',format='pdf',dpi=300)

def comp_ISM_phases(**kwargs):
    '''
    Purpose
    ---------
    Plotting function for comparing fractions of mass or line luminosity vs. SFR of Z.
    Called by ISM_line_contributions and ISM_line_efficiency() and ISM_mass_contributions() in analysis.py
    '''

    for key,val in zip(kwargs.keys(),kwargs.values()):
        exec(key + '=val')

    nGal                =   len(x1)

    # Starting figure
    ncols               =   2
    fontsize            =   11
    mpl.rcParams['xtick.labelsize'] = fontsize
    mpl.rcParams['ytick.labelsize'] = fontsize
    fig                 =   plt.figure(0,figsize = (14,5))

    # Left column
    x1_sort             =   np.sort(x1)
    dx                  =   (max(xr1)-min(xr1))/100.

    ax1                 =   fig.add_subplot(3,ncols,1)
    for i in range(0,nGal):
        # pdb.set_trace()
        y_sort              =   y1[0][x1.argsort()]
        simple_plot(add='y',xr=xr1,yr=yr1,xticklabels=[],\
            x1=[x1_sort[i]-dx,x1_sort[i]+dx],y1=[0,y_sort[i]],hatchstyle1='',col1='r',alpha1=0.6)
        if i == nGal-1:
            ax1                 =   plt.gca()
            ax1.fill_between([1e3,1e4], [-1e3,-1e3], [-1e4,-1e4], facecolor='r', lw=0, color='r', alpha=0.6,label='GMCs')
            handles, labels     =   ax1.get_legend_handles_labels()
            legend              =   ax1.legend(handles,labels,loc=[0.07,0.77],fontsize=fontsize*0.9, framealpha=0.,numpoints=1)
    ax1           =   fig.add_subplot(3,ncols,3)
    for i in range(0,nGal):
        y_sort              =   y1[1][x1.argsort()]
        simple_plot(add='y',xr=xr1,yr=yr1,xticklabels=[],ylab=ylab, lab_to_tick=1.1,\
            x2=[x1_sort[i]-dx,x1_sort[i]+dx],y2=[0,y_sort[i]],hatchstyle2='',col2='orange',alpha2=0.7)
        if i == nGal-1:
            ax1                 =   plt.gca()
            ax1.fill_between([1e3,1e4], [-1e3,-1e3], [-1e4,-1e4], facecolor='orange', lw=0, color='orange', alpha=0.7,label='Diffuse neutral gas')
            handles, labels     =   ax1.get_legend_handles_labels()
            legend              =   ax1.legend(handles,labels,loc=[0.07,0.77],fontsize=fontsize*0.9, framealpha=0.,numpoints=1)
    ax1           =   fig.add_subplot(3,ncols,5)
    for i in range(0,nGal):
        y_sort              =   y1[2][x1.argsort()]
        simple_plot(add='y',xr=xr1,yr=yr1,xlab=getlabel('SFR'),fontsize=12, lab_to_tick=1.1,\
            x3=[x1_sort[i]-dx,x1_sort[i]+dx],y3=[0,y_sort[i]],hatchstyle3='',col3='b',alpha3=0.5)
        if i == nGal-1:
            ax1                 =   plt.gca()
            ax1.fill_between([1e3,1e4], [-1e3,-1e3], [-1e4,-1e4], facecolor='b', lw=0, color='b', alpha=0.5,label='Diffuse ionized gas')
            handles, labels     =   ax1.get_legend_handles_labels()
            legend              =   ax1.legend(handles,labels,loc=[0.07,0.77],fontsize=fontsize*0.9, framealpha=0.,numpoints=1)

    # Right column
    x2_sort             =   np.sort(x2)
    dx                  =   (max(xr2)-min(xr2))/100.

    ax1                 =   fig.add_subplot(3,ncols,2)
    for i in range(0,nGal):
        y_sort              =   y1[0][x2.argsort()]
        simple_plot(add='y',xr=xr2,yr=yr1,xticklabels=[],\
            x1=[x2_sort[i]-dx,x2_sort[i]+dx],y1=[0,y_sort[i]],hatchstyle1='',col1='r',alpha1=0.6)
        if i == nGal-1:
            ax1                 =   plt.gca()
            ax1.fill_between([1e3,1e4], [-1e3,-1e3], [-1e4,-1e4], facecolor='r', lw=0, color='r', alpha=0.6,label='GMCs')
            handles, labels     =   ax1.get_legend_handles_labels()
            first_legend = ax1.legend(handles,labels,loc=[0.07,0.77],fontsize=fontsize*0.9, framealpha=0.,numpoints=1)
    ax1           =   fig.add_subplot(3,ncols,4)
    for i in range(0,nGal):
        y_sort              =   y1[1][x2.argsort()]
        simple_plot(add='y',xr=xr2,yr=yr1,xticklabels=[],ylab=ylab, lab_to_tick=1.1,\
            x1=[x2_sort[i]-dx,x2_sort[i]+dx],y1=[0,y_sort[i]],hatchstyle1='',col1='orange',alpha1=0.7)
        if i == nGal-1:
            ax1                 =   plt.gca()
            ax1.fill_between([1e3,1e4], [-1e3,-1e3], [-1e4,-1e4], facecolor='orange', lw=0, color='orange', alpha=0.7,label='Diffuse neutral gas')
            handles, labels     =   ax1.get_legend_handles_labels()
            first_legend = ax1.legend(handles,labels,loc=[0.07,0.77],fontsize=fontsize*0.9, framealpha=0.,numpoints=1)
    ax1           =   fig.add_subplot(3,ncols,6)
    for i in range(0,nGal):
        y_sort              =   y1[2][x2.argsort()]
        simple_plot(add='y',xr=xr2,yr=yr1,xlab=getlabel('Zmw'),fontsize=12, lab_to_tick=1.1,\
            x1=[x2_sort[i]-dx,x2_sort[i]+dx],y1=[0,y_sort[i]],hatchstyle1='',col1='b',alpha1=0.5)
        if i == nGal-1:
            ax1                 =   plt.gca()
            ax1.fill_between([1e3,1e4], [-1e3,-1e3], [-1e4,-1e4], facecolor='b', lw=0, color='b', alpha=0.5,label='Diffuse ionized gas')
            handles, labels     =   ax1.get_legend_handles_labels()
            first_legend = ax1.legend(handles,labels,loc=[0.07,0.77],fontsize=fontsize*0.9, framealpha=0.,numpoints=1)

    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.15,hspace=0,wspace=0.2)
    plt.show(block=False)

    return fig

#===============================================================================
""" Plot cloudy models """
#-------------------------------------------------------------------------------

def grid_parameters(histo_color='teal',FUV=5,ISM_phase='GMC',figsize=(10,7)):
    '''
    Purpose
    ---------
    Make histograms of galaxies with grid points on top

    What this function does
    ---------
    1) Makes 4-panel figure with histograms of [Mgmc, G0, Z, P_ext] in GMCs
    2) OR makes 4-panel figure with histograms of [nH, R, Z, Tk] in diffuse clouds

    Arguments
    ---------
    histo_color: color selection for histograms - str
    options:
        - '<a specific color>': all histograms will have this color
        - 'colsel': each galaxy will have a different color, using colsel from param_module.read_params

    '''

    plt.close('all')        # close all windows

    models                      =   aux.find_model_results()
    for key in models.keys():
        exec(key + '=models[key].values')

    if ISM_phase == 'GMC':

        print('\n Now looking at GMC grid parameters')
        plot_these          =   ['Mgmc','FUV','Z','P_ext']
        bins                =   80

        grid_params         =   pd.read_pickle('cloudy_models/GMC/grids/GMCgrid'+ext_DENSE+'_'+z1+'.param')
        grid_params['Mgmc'] =   grid_params['Mgmcs']
        grid_params['FUV']  =   grid_params['FUVs']
        grid_params['Z']    =   grid_params['Zs']
        grid_params['P_ext']=   grid_params['P_exts']

        fontsize            =   12
        mpl.rcParams['xtick.labelsize'] = fontsize
        mpl.rcParams['ytick.labelsize'] = fontsize

        plt.ion()
        fig                 =   plt.figure(1,figsize = figsize)
        panel               =   1
        if z1 == 'z6': ranges              =   [[3.5,6.],[0,7],[-4,0.5],[3,13]]
        if z1 == 'z2': ranges              =   [[3.5,6.],[2,7],[-2.2,1],[7,12.]]
        if z1 == 'z0': ranges              =   [[3.8,6.5],[-6.,5],[-2.2,1],[-2,10.]]
        for name in plot_these:
            # print(name)
            x_max       =   np.array([])
            x_min       =   np.array([])
            ax1         =   fig.add_subplot(2,2,panel)
            # Make histograms
            histos1     =   np.zeros([len(galnames),bins+2])
            histos2     =   np.zeros([len(galnames),bins+3])
            for i in range(0,len(galnames)):
                galname1        =   galnames[i]
                zred1           =   zreds[i]
                dat0            =   pd.read_pickle('sigame/temp/GMC/z'+'{:.2f}'.format(zred1)+'_'+galname1+'_GMC.gas')
                w               =   dat0['m'].values
                dat0['Mgmc']    =   dat0['m'].values
                # print('total GMC mass: '+str(np.sum(dat0['m'])))
                dat             =   dat0[name].values
                w               =   w[dat > 0]
                dat             =   dat[dat > 0]
                dg              =   (grid_params[name][1]-grid_params[name][0])
                dat             =   np.log10(dat)
                w               =   w[(dat > min(grid_params[name])-3.*dg) & (dat < max(grid_params[name])+3.*dg)]
                dat             =   dat[(dat > min(grid_params[name])-3.*dg) & (dat < max(grid_params[name])+3.*dg)]
                # remove data values = 0 !!
                i_nan           =   np.isnan(dat)
                w               =   w[i_nan == False]
                dat             =   dat[i_nan == False]
                if name == 'Mgmc': w = np.zeros(len(w))+1.
                dx              =   (max(grid_params[name])-min(grid_params[name]))*1.
                ranges.append([min(grid_params[name])-dx,max(grid_params[name])+dx])
                hist            =   np.histogram(dat,range=ranges[panel-1],bins=bins,weights=w)
                # Convert counts to percent:
                hist1           =   np.asarray(hist[0])
                hist2           =   np.asarray(hist[1])
                hist1           =   hist1*1./sum(hist1)*100.
                # add some zeros to bring histogram down
                wid             =   (hist2[1]-hist2[0])
                hist2           =   np.append([hist2],[hist2.max()+wid])
                hist2           =   np.append([hist2.min()-wid],[hist2])
                hist1           =   np.append([hist1],[0])
                hist1           =   np.append([0],[hist1])
                x_min           =   np.append(x_min,min(hist2[0:len(hist1)]+wid/2))
                x_max           =   np.append(x_max,max(hist2[0:len(hist1)]+wid/2))
                histos1[i,:]    =   hist1
                histos2[i,:]    =   hist2
            # Plot as background the 2 sigma distribution around the mean in each bin
            minhistos1,maxhistos1,meanhistos1       =   np.zeros(bins+2), np.zeros(bins+2), np.zeros(bins+2)
            for i in range(0,bins+2):
                meanhistos1[i]     =   np.mean(histos1[:,i])
                minhistos1[i]      =   meanhistos1[i]-2.*np.std(histos1[:,i])
                maxhistos1[i]      =   meanhistos1[i]+2.*np.std(histos1[:,i])
            # ax1.fill_between(histos2[0,0:len(hist1)], minhistos1, maxhistos1, facecolor='lightgreen', alpha=0.5, lw=0)

            # Now plot actual histograms
            for i in range(0,len(galnames)):
                if histo_color == 'teal': color = 'teal'
                if histo_color == 'colsel': color = colsel[i]
                hist2           =   histos2[i,:]
                hist1           =   histos1[i,:]
                ax1.plot(hist2[0:len(hist1)]+wid/2,hist1,ls='steps',color=color,label='G'+str(int(i+1)),alpha=0.7,lw=1)
            ax1.set_xlabel('log('+getlabel(name)+')')
            if name == 'Mgmc':
                ax1.set_ylabel('Number fraction [%]')
            else:
                ax1.set_ylabel('Mass fraction [%]')

            # Indicate grid points
            for grid_point in grid_params[name]:
                ax1.plot([grid_point,grid_point],[1e-3,1e3],'k--')

            # Now plot mean of histograms
            ax1.plot(hist2[0:len(hist1)]+wid/2,meanhistos1,ls='steps',color='blue',lw=1.5)

            # Fix axes
            ax1.set_yscale('log')
            ymin        =   10**(-1.)
            ax1.set_ylim([ymin,10.**(np.log10(max(maxhistos1))+(np.log10(max(maxhistos1))-np.log10(ymin))/4.)])
            dx                  =   (max(x_max)-min(x_min))/10.
            ax1.set_xlim(ranges[panel-1])

            if panel == 1:
                if histo_color == 'colsel': ax1.legend(loc='upper left')

            panel              +=  1

        plt.show(block=False)
        plt.savefig('plots/GMCs/grid_parameters/GMCgrid_points_on_histos'+ext_DENSE+'_'+z1+'.pdf',format='pdf') # .eps for paper!

    if ISM_phase == 'dif':

        print('\n Now looking at diffuse gas grid parameters')
        plot_these          =   ['nH','R','Z','Tk']
        bins                =   80
        ext_DIF1            =   '_'+str(FUV)+'UV'+ext_DIFFUSE
        grid_params         =   pd.read_pickle('cloudy_models/dif/grids/difgrid'+ext_DIF1+'_'+z1+'.param')
        grid_params['nH']   =   grid_params['nHs']
        grid_params['R']    =   grid_params['Rs']
        grid_params['Tk']   =   grid_params['Tks']
        grid_params['Z']    =   grid_params['Zs']

        plt.ion()
        fig                 =   plt.figure(2,figsize = figsize)
        panel               =   1
        if z1 == 'z6': ranges              =   [[-7,4],[-2.5,1],[-5,1],[0,7]]
        if z1 == 'z2': ranges              =   [[-7,2.2],[-1.7,1],[-6,1],[0,9]]
        if z1 == 'z0': ranges              =   [[-7,1],[-1.3,1.6],[-5,1],[1,8]]
        for name in plot_these:
            x_max       =   np.array([])
            x_min       =   np.array([])
            ax1         =   fig.add_subplot(2,2,panel)
            # Make histograms
            histos1     =   np.zeros([len(galnames),bins+2])
            histos2     =   np.zeros([len(galnames),bins+3])
            for i in range(0,len(galnames)):
                galname1        =   galnames[i]
                zred1           =   zreds[i]
                dat0            =   pd.read_pickle('sigame/temp/dif/z'+'{:.2f}'.format(zred1)+'_'+galname1+'_dif.gas')
                w               =   dat0['m']
                dat             =   dat0[name].values
                w               =   w[dat > 0]
                dat             =   dat[dat > 0]
                dg              =   (grid_params[name][1]-grid_params[name][0])
                dat             =   np.log10(dat)
                w               =   w[(dat > min(grid_params[name])-3.*dg) & (dat < max(grid_params[name])+3.*dg)]
                dat             =   dat[(dat > min(grid_params[name])-3.*dg) & (dat < max(grid_params[name])+3.*dg)]
                dx              =   (max(grid_params[name])-min(grid_params[name]))*1.
                ranges.append([min(grid_params[name])-dx,max(grid_params[name])+dx])
                hist            =   np.histogram(dat,range=ranges[panel-1],bins=bins,weights=w)
                # Convert counts to percent:
                hist1           =   np.asarray(hist[0])
                hist2           =   np.asarray(hist[1])
                hist1           =   hist1*1./sum(hist1)*100.
                # add some zeros to bring histogram down
                wid             =   (hist2[1]-hist2[0])
                hist2           =   np.append([hist2],[hist2.max()+wid])
                hist2           =   np.append([hist2.min()-wid],[hist2])
                hist1           =   np.append([hist1],[0])
                hist1           =   np.append([0],[hist1])
                x_min           =   np.append(x_min,min(hist2[0:len(hist1)]+wid/2))
                x_max           =   np.append(x_max,max(hist2[0:len(hist1)]+wid/2))
                histos1[i,:]    =   hist1
                histos2[i,:]    =   hist2
            # Plot as background the 2 sigma distribution around the mean in each bin
            minhistos1,maxhistos1,meanhistos1       =   np.zeros(bins+2), np.zeros(bins+2), np.zeros(bins+2)
            for i in range(0,bins+2):
                meanhistos1[i]     =    np.mean(histos1[:,i])
                minhistos1[i]      =   meanhistos1[i]-2.*np.std(histos1[:,i])
                maxhistos1[i]      =   meanhistos1[i]+2.*np.std(histos1[:,i])

            # Now plot actual histograms
            for galname,i in zip(galnames,range(0,len(galnames))):
                if histo_color == 'teal': color = 'teal'
                if histo_color == 'colsel': color = colsel[i]
                hist2           =   histos2[i,:]
                hist1           =   histos1[i,:]
                ax1.plot(hist2[0:len(hist1)]+wid/2,hist1,ls='steps',color=color,label='G'+str(int(i+1)),alpha=0.7,lw=1)
            ax1.set_xlabel('log('+getlabel(name)+')')
            ax1.set_ylabel('Mass fraction [%]')
            for grid_point in grid_params[name]:
                ax1.plot([grid_point,grid_point],[1e-3,1e3],'k--')

            # Now plot mean of histograms
            ax1.plot(hist2[0:len(hist1)]+wid/2,meanhistos1,ls='steps',color='blue',lw=1.5)

            ax1.set_yscale('log')
            ymin        =   10**(-1.2)
            ax1.set_ylim([ymin,10.**(np.log10(max(maxhistos1))+(np.log10(max(maxhistos1))-np.log10(ymin))/4.)])
            ax1.set_xlim(ranges[panel-1])

            if panel == 1:
                if histo_color == 'colsel': ax1.legend(loc='upper left')

            panel              +=  1
        plt.show(block=False)
        plt.savefig('plots/diffuse_gas/grid_parameters/difgrid_points_on_histos'+ext_DIFFUSE+'_'+z1+'.pdf',format='pdf') # .eps for paper!

#===============================================================================
""" Gas distribution plots """
#-------------------------------------------------------------------------------

#===============================================================================
""" Auxiliary plotting functions """
#-------------------------------------------------------------------------------

def axis_range(x,log=False,**kwargs):
    '''
    Purpose
    ---------
    Calculate a reasonable axis range
    '''

    if log:

        x       =   np.log10(x)
        xr      =   [np.min(x),np.max(x)]
        dx      =   (np.max(x)-np.min(x))/5.
        if kwargs.has_key('dex'):
            dx      =   kwargs['dex']
        xr      =   np.array([xr[0]-dx,xr[1]+dx])
        xr      =   10.**xr

        if len(x) == 1:
            xr      =   10.**np.array([x[0]-0.5,x[0]+0.5])

    else:

        xr      =   [np.min(x),np.max(x)]
        frac    =   1./5
        if kwargs.has_key('frac'): frac = kwargs['frac']
        dx      =   (np.max(x)-np.min(x))*frac
        xr      =   np.array([xr[0]-dx,xr[1]+dx])

    return xr

def save_obs(z1=z1):
    '''
    Purpose
    ---------
    Saves observations of line emission!
    '''


    if z1 == 'z2':
        authors                     =   ['Malhotra+17']

        # [CII] observations

        malhotra17                  =   {\
            'names':['Abell 2667a','Cosmic Horseshoe','Abell 2218b','The Clone',"8 O'clock Arc",'SMMJ14011','SDSS 090122+181432','Abell 2218','MS1512-cB58','SDSS J085137+333114','SGASJ122651+215220','SDSS134332+415503','SDSSJ120924+264052','SDSSJ091538+382658','CL 2244'],\
            'z':np.array([1.0334,2.3811,1.032,2.003,2.728,2.5653,2.2558,2.515,2.7265,1.6926,2.9233,2.0927,1.021,1.501,2.237]),\
            'SFR':np.array([36.9,32.8,54.7,56,477,866,1560,155,100,29,43.7,18,2.8,14.5,19]),\
            'SFR+':np.zeros(15),\
            'SFR-':np.zeros(15),\
            'SFR_uplim':[0,0,0,0,0,0,0,0,0,0,0,0,1,1,1],\
            # 'R':np.array([1.5999]),\
            'L_FIR':1e11*np.array([1.1,1,1.7,1.7,14.7,26.7,48.11,4.8,3.1,0.9,1.3,0.5,0.08,0.45,0.6]),\
            'L_FIR_uplim':np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,1,1])}
        F_CII                       =   1e-18*np.array([9.4,3.0,11.3,2.1,1.0,1.8,10.4,7.4,4.7,2.0,9.3,0.9,7.8,13.8,1.4]) # W m ^-2
        D_L                         =   np.array([6881.1,19220.9,6869.6,15565.7,22665.3,21040.1,17997.2,20541.1,22650.3,12659.5,24637.0,16422.4,6779.1,10917.4,17814.6])*1e6*pc2m # m
        mag                         =   np.array([17,24,6.1,28,12.3,3.6,8,22,30,27,40,40,58,25,20]) # magnification factor
        malhotra17['L_CII']         =   4.*np.pi*D_L**2*F_CII/Lsun/mag # Lsun
        # malhotra17['L_CII']         =   1e-3*malhotra17['L_FIR']*np.array([7.3,14.5,12.6,3.3,0.9,2.6,2.75,9.43,8.3,4.0,33.5,3.5,7.4,15.3,0])
        malhotra17['L_CII+']        =   malhotra17['L_CII']*np.array([1.2/9.4,0.7/3,1.4/11.3,0.3/2.1,0.3/1,0.5/1.8,0.4/10.4,1.1/7.4,0.4/4.7,0.6/2,0.8/9.3,0.3/0.9,1.1/7.8,0.6/13.6,0.4/1.4])
        malhotra17['L_CII-']        =   malhotra17['L_CII+']
        malhotra17['L_CII_uplim']   =   np.zeros(15)

        observations                =   pd.DataFrame({\
            'Malhotra+17':malhotra17})

        observations.to_pickle('sigame/temp/observations/CII_'+z1)

        # CI observations

        walter11                    =   {\
            'names':['SMMJ14011'],\
            'z':np.array([2.5653]),\
            'SFR':np.array([866]),\
            'SFR+':np.zeros(1),\
            'SFR-':np.zeros(1),\
            'SFR_uplim':[0]}
        walter11['L_CI2']            =   aux.Jy2solLum(3.1, f_CI2, 2.5653, 21040.1) # Lsun
        walter11['L_CI2+']           =   walter11['L_CI2']*0.3/3.1 # Lsun
        walter11['L_CI2-']           =   walter11['L_CI2']*0.3/3.1 # Lsun

        weiss05                     =   {\
            'names':['SMMJ14011'],\
            'z':np.array([2.5653]),\
            'SFR':np.array([866]),\
            'SFR+':np.zeros(1),\
            'SFR-':np.zeros(1),\
            'SFR_uplim':[0]}
        walter11['L_CI1']            =   aux.Jy2solLum(1.8, f_CI1, 2.5653, 21040.1) # Lsun
        walter11['L_CI1+']           =   walter11['L_CI1']*0.3/1.8 # Lsun
        walter11['L_CI1-']           =   walter11['L_CI1']*0.3/1.8 # Lsun


        observations                =   pd.DataFrame({\
            'Walter+11':walter11})
        observations.to_pickle('sigame/temp/observations/CI_'+z1)


    if z1 == 'z6':

        # List of authors (chronologically):
        # Bradac: October, Pentericci: september, Inoue: june, Knudsen: june, Maiolino: september, Willott: july, Capak: june,
        # Schaerer: february, Ota: september, Gonzalez: april, Ouchi: december, kanekar: july
        # authors                 =   ['Bradac+16','Pentericci+16','Inoue+16','Knudsen+16','Maiolino+15','Willott+15','Capak+15',\
            # 'Schaerer+15','Ota+14','González-López+14','Ouchi+13','Kanekar+13']

        smit17              =   {'names':['COS-3018555981','COS-2987030247'],\
            'z':np.array([6.8540,6.8076]),
            'SFR':np.array([10.**(np.log10(19)+(np.log10((19+19.2)/19)/2.)),10.**(np.log10(16)+(np.log10((16+22.7)/16)/2.))]),\
            'SFR+':np.array([19.2+19,22.7+16]),\
            'SFR-':np.array([19.2,22.7]),\
            'SFR_uplim':[2,2],\
            'R':np.array([1e-3,1e-3]),\
            'L_CII':np.array([4.7e8,3.6e8]),\
            'L_CII+':np.array([0.5e8,0.5e8]),\
            'L_CII-':np.array([0.5e8,0.5e8]),\
            'L_CII_uplim':[0,0],\
            'reason':'vallini'}
        # SFR surface density if enough data is available:
        smit17['SFRsd']     =   smit17['SFR']/(np.pi*smit17['R']**2)

        decarli17           =   {'names':['SDSS J0842+1218 comp','CFHQ J2100-1715 comp','PSO J231-20 comp','PSO J308-21 comp'],\
            'z':np.array([6.0656,6.0796,6.5900,6.2485]),
            'SFR':np.array([140,800,760,77])*1.5/1.6,\
            'SFR+':np.array([50,100,70,26])*1.5/1.6,\
            'SFR-':np.array([50,100,70,26])*1.5/1.6,\
            'SFR_uplim':[0,0,0,0],\
            'R':np.array([1e-3,1e-3,1e-3,1e-3]),\
            'L_CII':np.array([1.87e9,2.45e9,4.47e9,0.66e9]),\
            'L_CII+':np.array([0.24e9,0.42e9,0.53e9,0.13e9]),\
            'L_CII-':np.array([0.24e9,0.42e9,0.53e9,0.13e9]),\
            'L_CII_uplim':[0,0,0,0],\
            'reason':'vallini'}
        # SFR surface density if enough data is available:
        decarli17['SFRsd']  =   decarli17['SFR']/(np.pi*decarli17['R']**2)


        bradac16            =   {'names':['RXJ1347:1216'],\
            'z':np.array([6.7655]),\
            'SFR':np.array([8.5]),\
            'SFR+':np.array([5.8]),\
            'SFR-':np.array([1.0]),\
            'SFR_uplim':[0],\
            'R':np.array([1.5999]),\
            'L_CII':np.array([1.5e7]),\
            'L_CII+':np.array([0.2e7]),\
            'L_CII-':np.array([0.4e7]),\
            'L_CII_uplim':[0],\
            'reason':'vallini'}
        # SFR surface density if enough data is available:
        bradac16['SFRsd']   =   bradac16['SFR']/(np.pi*bradac16['R']**2)

        pentericci16            =   {'names':['COSMOS13679','NTTDF6345','UDS16291'],\
            'z':np.array([7.1453,6.701,6.6381]),\
            # 'SFR':np.array([23.9+6.2,25.0+5.7,15.8+6.6]),\
            'SFR':np.array([10.**(np.log10(23.9)+(np.log10((23.9+6.2)/23.9)/2.)),10.**(np.log10(25.)+(np.log10((25.+5.7)/25.)/2.)),10.**(np.log10(15.8)+(np.log10((15.8+6.6)/15.8)/2.))]),\
            'SFR+':np.array([23.9+6.2,25.0+5.7,15.8+6.6]),\
            'SFR-':np.array([23.9,25.0,15.8]),\
            # 'SFR_uplim':[1,1,1],\
            'SFR_uplim':[2,2,2],\
            'R':np.array([2.32335,2.41335,2.4264]),\
            'L_CII':10.**np.array([7.854062,8.247278,7.85406169]),\
            'L_CII_uplim':[0,0,0],\
            'reason':'vallini'}
        # error bars from listed S/N
        pentericci16['L_CII+']   =   pentericci16['L_CII']*(1.-1./np.array([4.5,6.1,4.5]))
        pentericci16['L_CII-']   =   pentericci16['L_CII']*(1.-1./np.array([4.5,6.1,4.5]))
        # SFR surface density if enough data is available:
        pentericci16['SFRsd']   =   pentericci16['SFR']/(np.pi*pentericci16['R']**2)

        inoue16            =   {'names':['SXDF-NB1006-2'],\
            'z':np.array([7.2120]),\
            'SFR':10.**np.array([2.54]),\
            'SFR+':10.**np.array([2.54+0.17])-10.**np.array([2.54]),\
            'SFR-':10.**np.array([2.54])-10.**np.array([2.54-0.71]),\
            'SFR_uplim':[0],\
            'L_CII':np.array([8.3355e7]),\
            'L_CII+':np.array([0]),\
            'L_CII-':np.array([0]),\
            'L_CII_uplim':[1],\
            'reason':'vallini'}
        # SFR surface density if enough data is available:
        # inoue16['SFRsd']   =   inoue16['SFR']/(np.pi*inoue16['R']**2)

        knudsen16               =   {'names':['A383-5.1', 'MS0451-H', 'A1689-zD1'],\
                'z':np.array([6.0274,6.703,7.6031]),\
                # 'SFR':np.array([3.2+0.5,0.4+0.07,12]),\
                'SFR':np.array([10.**(np.log10(3.2)+(np.log10((3.2+0.5)/3.2)/2.)),10.**(np.log10(0.4)+(np.log10((0.4+0.07)/0.4)/2.)),12]),\
                'SFR+':np.array([3.2+0.5,0.4+0.07,0]),\
                'SFR-':np.array([3.2,0.4,0]),\
                # 'SFR_uplim':[0,0,0],\
                'SFR_uplim':[2,2,0],\
                'L_CII':np.array([8.3e6,3e5,1.8e7]),\
                'L_CII+':np.array([3.1e6,0,0]),\
                'L_CII-':np.array([3.1e6,0,0]),\
                'L_CII_uplim':[0,1,1],\
                'reason':'other'}
        # SFR surface density if enough data is available:
        knudsen16['SFRsd']     =   np.array([10,knudsen16['SFR'][1]/(np.pi*0.6**2)])

        maiolino15              =   {'names':['BDF-3299','BDF-512','SDF-46975','BDF-3299-clump'],\
                'z':np.array([7.109,7.008,6.844,7.107]),\
                'SFR':np.array([5.7,6.0,15.4,5.7/3.5])*1.5/1.6,\
                'SFR+':np.array([0,0,0,0]),\
                'SFR-':np.array([0,0,0,0]),\
                'SFR_uplim':[0,0,0,1],\
                'R':np.array([0.57,0.57,0.64,1e-3]),\
                'L_CII':np.array([2e7,6e7,5.7e7,5.9e7]),\
                'L_CII+':np.array([0,0,0,0.8]),\
                'L_CII-':np.array([0,0,0,0.8]),\
                'L_CII_uplim':[1,1,1,0],\
                'reason':'vallini'}
        # SFR surface density if enough data is available:
        maiolino15['SFRsd']     =   np.array([maiolino15['SFR']/(np.pi*maiolino15['R']**2)])

        willott15                 =   {'names':['CLM1','WMH5'],\
                'z':np.array([6.1657,6.0695]),\

                'SFR':np.array([37,43]),\
                'SFR+':np.array([0,0]),\
                'SFR-':np.array([0,0]),\
                'SFR_uplim':[0,0],\
                'M_star':np.array([1.3e10,2.3e10]),\
                'R':np.array([3.2,0.74]),\
                'L_CII':np.array([2.4e8,6.6e8]),\
                'L_CII+':np.array([0.32e8,0.72e8]),\
                'L_CII-':np.array([0.32e8,0.72e8]),\
                'L_CII_uplim':[0,0],\
                'reason':'none'}
        # SFR surface density if enough data is available:
        willott15['SFRsd']     =   np.array([willott15['SFR']/(np.pi*willott15['R']**2)])
        # End of error bars:
        willott15['uL_CII']     =   willott15['L_CII']+willott15['L_CII+']
        willott15['lL_CII']     =   willott15['L_CII']-willott15['L_CII-']


        capak15                 =   {'names':['HZ1','HZ2','HZ3','HZ4','HZ5','HZ6','HZ7','HZ8','HZ9','HZ10'],\
            'z':np.array([5.6885,5.6697,5.5416,5.5440,5.3089,5.2928,5.2532,5.1533,5.5410,5.6566]),\
            'SFR':np.array([24,25,18,51,3,49,21,18,67,169]),\
            'SFR+':np.array([6,5,8,54,0,44,5,5,30,32]),\
            'SFR-':np.array([3,2,3,18,0,12,2,2,20,27]),\
            'SFR_uplim':[0,0,0,0,1,0,0,0,0,0],\
            'M_star':10.**np.array([10.47,10.23,10.23,9.67,0,10.17,9.86,9.77,9.86,10.39]),\
            'M_dyn':10.**np.array([9.8,10.7,10.4,10.4,0,10.6,10.8,10.8,10.7]),\
            'R':np.array([1.53,0.59,0.66,0.72,0.37,3.36,0.98,1.24,0.95,0]),\
            'L_CII':10.**np.array([8.40,8.56,8.67,8.98,7.2,9.15,8.74,8.41,9.21,9.13]),\
            'L_CII+':10.**(np.array([8.40,8.56,8.67,8.98,7.2,9.15,8.74,8.41,9.21,9.13])+np.array([0.32,0.41,0.28,0.22,0,0.17,0.24,0.18,0.09,0.13]))-10.**np.array([8.40,8.56,8.67,8.98,7.2,9.15,8.74,8.41,9.21,9.13]),\
            'L_CII-':10.**np.array([8.40,8.56,8.67,8.98,7.2,9.15,8.74,8.41,9.21,9.13])-10.**(np.array([8.40,8.56,8.67,8.98,7.2,9.15,8.74,8.41,9.21,9.13])-np.array([0.32,0.41,0.28,0.22,0,0.17,0.24,0.18,0.09,0.13])),\
            'L_CII_uplim':[0,0,0,0,1,0,0,0,0,0],\
            'reason':'none'}
        # SFR surface density if enough data is available:
        capak15['SFRsd']        =   capak15['SFR']/(np.pi*capak15['R']**2)
        capak15['SFRsd'][capak15['R'] == 0]        =   -1.
        # End of error bar:
        capak15['uL_CII']        =   capak15['L_CII']+capak15['L_CII+']
        capak15['lL_CII']        =   capak15['L_CII']-capak15['L_CII-']

        schaerer15                 =   {'names':['A1703-zD1','z8-GND-5296'],\
                'z':np.array([6.8,7.508]),\
                # 'SFR':np.array([9+13.8,23.4+113])*1/1.6,\
                'SFR':np.array([10.**(np.log10(9)+np.log10((9+13.8)/9)/2.),10.**(np.log10(23.4)+(np.log10((23.4+113)/23.4)/2.))])*1/1.6,\
                'SFR-':np.array([9,23.4])*1/1.6,\
                'SFR+':np.array([9+13.8,23.4+113])*1/1.6,\
                # 'SFR_uplim':[1,1],\
                'SFR_uplim':[2,2],\
                'R':np.array([4/2.,0.5]),\
                'L_CII':np.array([0.2833e8,3.56e8]),\
                'L_CII+':np.array([0,0]),\
                'L_CII-':np.array([0,0]),\
                'L_CII_uplim':[1,1],\
                'reason':'other'}
        # SFR surface density if enough data is available:
        schaerer15['SFRsd']     =   np.array([schaerer15['SFR']/(np.pi*schaerer15['R']**2)])

        ota14                   =   {'names':['IOK-1'],\
                'z':np.array([6.96]),\
                'SFR':np.array([10.**(np.log10(10)+np.log10((10+23.9)/10.)/2.)])*1.5/1.6,\
                'SFR+':np.array([10+23.9])*1.5/1.6,\
                'SFR-':np.array([10])*1.5/1.6,\
                # 'SFR_uplim':[1],\
                'SFR_uplim':[2],\
                'R':np.array([0.64]),\
                'L_CII':np.array([3.4e7]),\
                'L_CII+':np.array([0]),\
                'L_CII-':np.array([0]),\
                'L_CII_uplim':[1],\
                'reason':'other'}
        # SFR surface density if enough data is available:
        ota14['SFRsd']     =   np.array([ota14['SFR']/(np.pi*ota14['R']**2)])

        gonzalez14              =   {'names':['SDF J132415.7','SDF J132408.3'],\
            'z':np.array([6.541,6.554]),\
            # 'SFR':np.array([34+177.2,15+360.9]),\
            'SFR':np.array([10.**(np.log10(34)+(np.log10((34+177.2)/34)/2.)),10.**(np.log10(15.)+(np.log10((15.+360.9)/15.)/2.))]),\
            'SFR+':np.array([34+177.2,15+360.9]),\
            'SFR-':np.array([34,15]),\
            # 'SFR_uplim':[1,1],\
            'SFR_uplim':[2,2],\
            'R':10.**np.array([4,3.2]),\
            'L_CII':np.array([4.52e8,10.56e8]),\
            'L_CII+':np.array([0,0]),\
            'L_CII-':np.array([0,0]),\
            'L_CII_uplim':[1,1],\
            'reason':'none'}
        # SFR surface density if enough data is available:
        gonzalez14['SFRsd']     =   gonzalez14['SFR']/(np.pi*gonzalez14['R']**2)

        ouchi13                 =   {'names':['Himiko'],\
                'z':np.array([6.595]),\
                'SFR':np.array([100.])*1/1.6,\
                'SFR+':np.array([2.])*1/1.6,\
                'SFR-':np.array([2.])*1/1.6,\
                'SFR_uplim':[0],\
                'R':np.array([17/2.]),\
                'L_CII':np.array([0.54e8]),\
                'L_CII+':np.array([0]),\
                'L_CII-':np.array([0]),\
                'L_CII_uplim':[1],\
                'reason':'other'}
        # SFR surface density if enough data is available:
        ouchi13['SFRsd']     =   np.array([ouchi13['SFR']/(np.pi*ouchi13['R']**2)])

        kanekar13                 =   {'names':['HCM 6A'],\
                'z':np.array([6.56]),\
                'SFR':np.array([10])*1/1.6,\
                'SFR+':np.array([0]),\
                'SFR-':np.array([0]),\
                'SFR_uplim':[0],\
                'L_CII':np.array([0.64e8]),\
                'L_CII+':np.array([0]),\
                'L_CII-':np.array([0]),\
                'L_CII_uplim':[1],\
                'reason':'none'}

        # maiolino05              =   {'names':['SDSS J1148+52511'],\
        #         'z':np.array([6.42]),\
        #         'SFR':np.array([])*1.5/1.6,\
        #         'SFR+':np.array([0,0,0]),\
        #         'SFR-':np.array([0,0,0]),\
        #         'SFR_uplim':[0,0,0],\
        #         'R':np.array([0.57,0.57,0.64]),\
        #         'L_CII':np.array([4.4e9]),\
        #         'L_CII+':np.array([0,0,0]),\
        #         'L_CII-':np.array([0,0,0]),\
        #         'L_CII_uplim':[1,1,1],\
        #         'reason':'vallini'}
        # SFR surface density if enough data is available:
        # maiolino15['SFRsd']     =   np.array([maiolino15['SFR']/(np.pi*maiolino15['R']**2)])


        # SFR surface density if enough data is available:
        # kanekar13['SFRsd']     =   np.array([kanekar13['SFR'].values/(np.pi*kanekar13['R']**2)])

        # Collect in ONE dataframe:
        observations        =   pd.DataFrame({'Smit+17':smit17,'Decarli+17':decarli17,\
            'Bradac+16':bradac16,'Pentericci+16':pentericci16,'Inoue+16':inoue16,\
            'Knudsen+16+17':knudsen16,'Maiolino+15':maiolino15,\
            'Willott+15':willott15,'Capak+15':capak15,'Schaerer+15':schaerer15,\
            'Ota+14':ota14,'Gonzalez-Lopez+14':gonzalez14,'Ouchi+13':ouchi13,\
            'Kanekar+13':kanekar13})

        observations.to_pickle('sigame/temp/observations/observations_'+z1)

def add_observations_to_plot(slope_models,intercept_models,mark_reasons=False,mark_uplim=False,mark_det=False,z1='z6',MW=True,ms_scaling=1,alpha=1,line='[CII]'):
    ''''
    Purpose
    ---------
    Adds observatinos to plot
    '''

    arrwidth            =   1.5         # width of arrow
    capthick            =   ms_scaling*3           # size of arrow head
    errwidth            =   1.8         # witch of errorbars

    # Get axis for this figure:
    ax1                 =   plt.gca()

    CII_obs             =   pd.read_pickle('sigame/temp/observations/observations'+'_'+z1)

    if z1 == 'z6':
    	# List of authors (chronologically):
    	# Bradac: October, Pentericci: september, Inoue: june, Knudsen: june, Maiolino: september, Willott: july, Capak: june,
    	# Schaerer: february, Ota: september, Gonzalez: april, Ouchi: december, kanekar: july
    	CII_obs             =   CII_obs[['Kanekar+13','Ouchi+13','Gonzalez-Lopez+14','Ota+14','Schaerer+15','Capak+15',\
    	                    'Willott+15','Maiolino+15','Knudsen+16+17','Inoue+16','Pentericci+16','Bradac+16','Decarli+17','Smit+17']]

    	# And let's decide which author gets which symbol:
    	obs_col             =   'darkgrey'
    	markers             =   {'Kanekar+13':'d','Ouchi+13':'*','Gonzalez-Lopez+14':'>','Ota+14':'^','Schaerer+15':'v','Capak+15':'D',\
    	                    'Willott+15':'x','Maiolino+15':'s','Knudsen+16+17':'<','Inoue+16':'h','Pentericci+16':'+','Bradac+16':'p','Decarli+17':'h','Smit+17':'s'}
    	ms                  =   {'Kanekar+13':7,'Ouchi+13':9,'Gonzalez-Lopez+14':8,'Ota+14':8,'Schaerer+15':8,'Capak+15':7,\
    	                    'Willott+15':7,'Maiolino+15':7,'Knudsen+16+17':8,'Inoue+16':5,'Pentericci+16':9,'Bradac+16':9,'Decarli+17':9,'Smit+17':5}
    	mews                =   {'Kanekar+13':1.5,'Ouchi+13':1.5,'Gonzalez-Lopez+14':1.5,'Ota+14':1.5,'Schaerer+15':1.5,'Capak+15':1.5,\
    	                    'Willott+15':3,'Maiolino+15':1.5,'Knudsen+16+17':1.5,'Inoue+16':1.5,'Pentericci+16':3,'Bradac+16':1.5,'Decarli+17':1.5,'Smit+17':1.5}

    if z1 == 'z2':
        CII_obs             =   CII_obs[['Malhotra+17']]

        # And let's decide which author gets which symbol:
        obs_col             =   'dimgrey'
        markers             =   {'Malhotra+17':'s'}
        ms                  =   {'Malhotra+17':4}
        mews                =   {'Malhotra+17':1.5}

    # Plot with different symbols
    n_det               =   np.zeros(len(CII_obs.keys()))
    n_det_above         =   np.zeros(len(CII_obs.keys()))
    n_det_below         =   np.zeros(len(CII_obs.keys()))
    n_nondet            =   np.zeros(len(CII_obs.keys()))
    n_nondet_below      =   np.zeros(len(CII_obs.keys()))
    n_tot               =   np.zeros(len(CII_obs.keys()))
    zred_nondet         =   np.array([])
    zred_det            =   np.array([])
    i                   =   0
    for author in CII_obs.keys():
        obs                 =   CII_obs[author]
        n_tot[i]            =   len(obs['L_CII_uplim'])
        n_det[i]            =   len([ind for ind, uplim in enumerate(obs['L_CII_uplim']) if uplim == 0])
        n_nondet[i]         =   len([ind for ind, uplim in enumerate(obs['L_CII_uplim']) if uplim == 1])
        print(author)
        # if author == 'Smit+17': pdb.set_trace()
        if n_det[i]>0:
            SFR_lower_limits            =   [obs['SFR'][j] for j in range(0,len(obs['SFR'])) if obs['L_CII_uplim'][j] == 0]
            z                           =   [obs['z'][j] for j in range(0,len(obs['SFR'])) if obs['L_CII_uplim'][j] == 0]
            zred_det                    =   np.append(zred_det,np.array(z))
            print('Detected z: ',z)
            L_CII_lower_limits           =   [obs['L_CII'][j]-obs['L_CII-'][j] for j in range(0,len(obs['L_CII'])) if obs['L_CII_uplim'][j] == 0]
            L_CII_upper_limits           =   [obs['L_CII'][j]+obs['L_CII+'][j] for j in range(0,len(obs['L_CII'])) if obs['L_CII_uplim'][j] == 0]
            L_CII_models_prediction      =   10.**(slope_models*np.log10(SFR_lower_limits)+intercept_models) # according to models
            n_det_above[i]              =   len(np.array([L_CII_lower_limits[j] for j in range(0,len(L_CII_lower_limits)) if (L_CII_lower_limits[j] > L_CII_models_prediction[j])]))
            n_det_below[i]              =   len(np.array([L_CII_lower_limits[j] for j in range(0,len(L_CII_lower_limits)) if (L_CII_upper_limits[j] < L_CII_models_prediction[j])]))
        if n_nondet[i]>0:
            SFR_lower_limits            =   [obs['SFR'][j] for j in range(0,len(obs['SFR'])) if obs['L_CII_uplim'][j] == 1]
            z                           =   [obs['z'][j] for j in range(0,len(obs['SFR'])) if obs['L_CII_uplim'][j] == 1]
            print('Not detected z: ',z)
            zred_nondet                 =   np.append(zred_nondet,np.array(z))
            # pdb.set_trace()
            L_CII_upper_limits           =   [obs['L_CII'][j] for j in range(0,len(obs['L_CII'])) if obs['L_CII_uplim'][j] == 1]
            L_CII_models_prediction      =   10.**(slope_models*np.log10(SFR_lower_limits)+intercept_models) # according to models
            n_nondet_below[i]           =   len(np.array([L_CII_upper_limits[j] for j in range(0,len(L_CII_upper_limits)) if (L_CII_upper_limits[j] < L_CII_models_prediction[j])]))
        i                   +=  1
    print('\nTotal number of detections:')
    print(sum(n_det))
    print('Min and max z: '+str(min(zred_det))+' '+str(max(zred_det)))
    print('\nTotal number of non-detections:')
    print(sum(n_nondet))
    print('Min and max z: '+str(min(zred_nondet))+' '+str(max(zred_nondet)))
    print('total: '+str(sum(n_tot)))


    print('\nTotal number detections with lower errorbars above our relation:')
    print(sum(n_det_above))
    print('\nTotal number detections with upper errorbars below our relation:')
    print(sum(n_det_below))
    print('\nTotal number non-detections with upper limits below our relation:')
    print(sum(n_nondet_below))

    print('\nAverage factor below Capak and Willot at high SFR end:')
    SFR_capak_willott               =   np.append(CII_obs['Capak+15']['SFR'],CII_obs['Willott+15']['SFR'])
    L_CII_capak_willott              =   np.append(CII_obs['Capak+15']['L_CII'],CII_obs['Willott+15']['L_CII'])
    L_CII_uplim_capak_willott        =   np.append(CII_obs['Capak+15']['L_CII_uplim'],CII_obs['Willott+15']['L_CII_uplim'])
    SFR_capak_willott               =   SFR_capak_willott[L_CII_uplim_capak_willott == 0]
    L_CII_capak_willott              =   L_CII_capak_willott[L_CII_uplim_capak_willott == 0]
    L_CII_models_predictions         =   10.**(slope_models*np.log10(SFR_capak_willott)+intercept_models)
    print(np.mean(L_CII_capak_willott/L_CII_models_predictions))
    print('min and max: '+str(min(L_CII_capak_willott/L_CII_models_predictions))+' '+str(max(L_CII_capak_willott/L_CII_models_predictions)))

    i                   =   0
    for author in CII_obs.keys():

        obs                 =   CII_obs[author]
        ms1                 =   ms_scaling*ms[author]

        if mark_reasons:
            if obs['reason'] == 'vallini': col_reason = 'green'
            if obs['reason'] == 'littleHI': col_reason = 'blue'
            if obs['reason'] == 'other': col_reason = 'turquoise'
            if obs['reason'] != 'none': ax1.plot(obs['SFR'],obs['L_CII'],marker='o', fillstyle='full', mew=0, ms=20, color=col_reason, markeredgecolor = col_reason, linestyle="None",alpha=alpha,zorder=100)
            # pdb.set_trace()

        # Mark with color?
        if mark_uplim:
            if np.sum(obs['L_CII_uplim']) > 0:
                index       =   [ind for ind, uplim in enumerate(obs['L_CII_uplim']) if uplim == 1]
                ax1.plot(obs['SFR'][index],obs['L_CII'][index],marker='o', fillstyle='full', mew=0, ms=20, color='pink', markeredgecolor = 'pink', linestyle="None",alpha=alpha,zorder=100)
        if mark_det:
            if np.sum(obs['L_CII_uplim']) < len(obs['L_CII_uplim']):
                index       =   [ind for ind, uplim in enumerate(obs['L_CII_uplim']) if uplim == 0]
                ax1.plot(obs['SFR'][index],obs['L_CII'][index],marker='o', fillstyle='full', mew=0, ms=20, color='cyan', markeredgecolor = 'cyan', linestyle="None",alpha=alpha,zorder=100)

        label                   =   CII_obs.keys()[i]
        if CII_obs.keys()[i] == 'Gonzalez-Lopez+14': label = u'Gonzàlez-Lòpez+14'
        if CII_obs.keys()[i] == 'Bradac+16': label = r'Bradac'+r'$\check{}$'+'+16' # 'Brada'+r'$\v{c}$'+'+16' #'Brada'++'+16'

        # Label it, but plot far away:
        if np.sum(obs['L_CII_uplim']) < len(obs['L_CII_uplim']):
            ax1.plot([1e10],[1e10],marker=markers[author], fillstyle='full', mew=mews[author], ms=ms1, color=obs_col, markeredgecolor = obs_col, linestyle="None", label=label,alpha=alpha,zorder=100)
        if np.sum(obs['L_CII_uplim']) == len(obs['L_CII_uplim']):
            ax1.plot([1e10],[1e10],marker=markers[author], fillstyle='none', mew=mews[author], ms=ms1, color=obs_col, markeredgecolor = obs_col, linestyle="None", label=label,alpha=alpha,zorder=100)


        # # Upper limits or range on SFR?
        for j in range(0,len(obs['SFR_uplim'])):
            SFR_uplim           =   obs['SFR_uplim'][j]
            if SFR_uplim == 1:
                ax1.errorbar(obs['SFR'][j],obs['L_CII'][j],\
                    color=obs_col, markeredgecolor = obs_col, marker=markers[author], fillstyle='none', mew=mews[author], ms=ms1,\
                    xerr=arrow_length(obs['SFR'][j]), xuplims=[True]*np.sum(obs['SFR_uplim'][j]),\
                    linestyle="None",elinewidth=errwidth,capthick=capthick,alpha=alpha,zorder=100)
            if SFR_uplim == 2:
                ax1.plot([obs['SFR-'][j],obs['SFR+'][j]],[obs['L_CII'][j],obs['L_CII'][j]],\
                    color=obs_col, markeredgecolor = obs_col, mew=mews[author], ms=ms1,\
                    linestyle='-',lw=errwidth,alpha=alpha,zorder=0)
                ax1.plot(obs['SFR'][j],obs['L_CII'][j], color=obs_col, marker=markers[author], fillstyle='none', mew=mews[author], ms=ms1,alpha=alpha,zorder=100)
                print(author,obs['SFR'][j])
        # Upper limits on L_CII?
        if np.sum(obs['L_CII_uplim']) > 0:
            index       =   [ind for ind, uplim in enumerate(obs['L_CII_uplim']) if uplim == 1]
            ax1.errorbar(obs['SFR'][index],obs['L_CII'][index],\
                color=obs_col, markeredgecolor = obs_col, marker=markers[author], fillstyle='none', mew=mews[author], ms=ms1,\
                yerr=arrow_length(obs['L_CII'][index]), uplims=[True]*sum(obs['L_CII_uplim']),\
                linestyle="None",elinewidth=errwidth,capthick=capthick,alpha=alpha,zorder=100)


        # Plot [CII] detections with error bars
        if np.sum(obs['L_CII_uplim']) < len(obs['L_CII_uplim']):
            index       =   [ind for ind, uplim in enumerate(obs['L_CII_uplim']) if uplim == 0]
            ax1.errorbar(obs['SFR'][index],obs['L_CII'][index],\
                color=obs_col, markeredgecolor = obs_col, marker=markers[author], fillstyle='full', mew=mews[author], ms=ms1,\
                yerr=[np.array(obs['L_CII-'][index]).tolist(),np.array(obs['L_CII+'][index]).tolist()],\
                linestyle="None",elinewidth=errwidth,capthick=0,alpha=alpha,zorder=100)

        # Add error bars on SFR
        if np.sum(obs['SFR_uplim']) < len(obs['SFR_uplim']):
            index       =   [ind for ind, uplim in enumerate(obs['SFR_uplim']) if uplim == 0]
            ax1.errorbar(obs['SFR'][index],obs['L_CII'][index],\
                color=obs_col, markeredgecolor = obs_col, marker=markers[author], fillstyle='full', mew=mews[author], ms=ms1,\
                xerr=[np.array(obs['SFR-'][index]).tolist(),np.array(obs['SFR+'][index]).tolist()],\
                linestyle="None",elinewidth=errwidth,capthick=0,alpha=alpha,zorder=0)

        # White-filled symbols for upper limits:
        if np.sum(obs['L_CII_uplim']) > 0:
            index       =   [ind for ind, uplim in enumerate(obs['L_CII_uplim']) if uplim == 1]
            ax1.errorbar(obs['SFR'][index],obs['L_CII'][index],\
                color='white', markeredgecolor = obs_col, marker=markers[author], fillstyle='full', mew=mews[author], ms=ms1,\
                linestyle="None",elinewidth=errwidth,capthick=capthick,alpha=alpha,zorder=100)
            ax1.plot(obs['SFR'][index],obs['L_CII'][index], marker=markers[author], color=obs_col, fillstyle='none', mew=mews[author], ms=ms1, lw=0,alpha=alpha,zorder=100)


        i                   +=  1

    # MW
    if line == '[CII]':
        SFR_MW              =   1.9                     # Chomiuk+11
        L_CII_MW            =   1e41/1e7/Lsun           # Pineda+14
        if MW: ax1.errorbar(SFR_MW,L_CII_MW,marker='*',ms=12,label='Milky Way',c='black',linestyle="None")

def getlabel(foo):
    '''
    Purpose
    ---------
    Gets label for plots
    '''

    if foo == 'x': return 'x position [kpc]'
    if foo == 'y': return 'y position [kpc]'
    if foo == 'z': return 'y position [kpc]'
    if foo == 'vx': return 'v$_x$ [km s$^{-1}$]'
    if foo == 'vy': return 'v$_y$ [km s$^{-1}$]'
    if foo == 'vz': return 'v$_z$ [km s$^{-1}$]'
    if foo == 'nH': return '$n_{\mathrm{H}}$ [cm$^{-3}$]'
    if foo == 'lnH': return 'log($n_{\mathrm{H}}$ [cm$^{-3}$])'
    if foo == 'nHmw': return r'$\langle n_{\mathrm{H}}\rangle_{\mathrm{mass}}$'+' [cm$^{-3}$]'
    if foo == 'nH_pdr': return 'H density of PDR gas [cm$^{-3}$]'
    if foo == 'R_pdr': return 'Size of PDR gas [pc]'
    if foo == 'Rgmc': return 'R$_{\mathrm{GMC}}$ [pc]'
    if foo == 'lRgmc': return 'log(R$_{\mathrm{GMC}}$ [pc])'
    if foo == 'f_HI': return 'f$_{\mathrm{[HI]}}$'
    # if foo == 'f_HI1': return 'f$_{\mathrm{[HI]}}$ before'
    if foo == 'f_H2': return 'f$_{\mathrm{mol}}$'
    # if foo == 'f_H21': return 'f$_{\mathrm{mol}}$ before'
    if foo == 'Tk': return '$T_{\mathrm{k}}$ [K]'
    if foo == 'Z': return '$Z$ [Z$_{\odot}$]'
    if foo == 'lZ': return 'log($Z$ [Z$_{\odot}$])'
    if foo == 'Zmw': return r"$\langle Z'\rangle_{\mathrm{mass}}$"
    if foo == 'Zsfr': return r"$\langle Z'\rangle_{\mathrm{SFR}}$"
    if foo == 'lZsfr': return r"log($\langle Z'\rangle_{\mathrm{SFR}}$ [$Z_{\odot}$])"
    if foo == 'SFR': return 'SFR [M$_{\odot}$yr$^{-1}$]'
    if foo == 'lSFR': return 'log(SFR [M$_{\odot}$yr$^{-1}$])'
    if foo == 'sSFR': return 'sSFR [yr$^{-1}$]'
    if foo == 'SFRsd': return '$\Sigma$$_{\mathrm{SFR}}$ [M$_{\odot}$/yr/kpc$^{2}$]'
    if foo == 'lSFRsd': return 'log($\Sigma$$_{\mathrm{SFR}}$ [M$_{\odot}$/yr kpc$^{-2}$])'
    if foo == 'h': return 'Smoothing length $h$ [kpc]'
    if foo == 'm': return 'Total gas mass [M$_{\odot}$]'
    if foo == 'lm': return 'log(Total gas mass [M$_{\odot}$])'
    if foo == 'Ne': return 'Electron fraction'
    if foo == 'ne': return 'n$_{e}$ [cm$^{-3}$]'
    if foo == 'Mgmc': return '$m_{\mathrm{GMC}}$ [M$_{\odot}$]'
    if foo == 'm_mol': return '$m_{\mathrm{mol}}$ [M$_{\odot}$]'
    if foo == 'M_dust': return 'M$_{\mathrm{dust}}$ [M$_{\odot}$]'
    if foo == 'M_star': return 'M$_{\mathrm{*}}$ [M$_{\odot}$]'
    if foo == 'M_ISM': return 'M$_{\mathrm{ISM}}$ [M$_{\odot}$]'
    if foo == 'lM_ISM': return 'log(M$_{\mathrm{ISM}}$ [M$_{\odot}$])'
    if foo == 'g0': return "$G_{0}$ [Habing]"
    if foo == 'CR': return "$\zeta_{\mathrm{CR}}$ [s$^{-1}$]"
    if foo == 'P_ext': return "$P_{\mathrm{ext}}$ [K cm$^{-3}$]"
    if foo == 'lP_ext': return "log($P_{\mathrm{ext}}$ [K cm$^{-3}$])"
    if foo == 'lP_extmw': return r"log($\langle P_{\mathrm{ext}}\rangle_{\mathrm{mass}}$)"
    if foo == 'age': return "Age [Myr]"
    if foo == 'lage': return "log(Age [Myr])"
    if foo == 'C': return "C mass fraction I think?"
    if foo == 'O': return "O mass fraction I think?"
    if foo == 'Si': return "Si mass fraction I think?"
    if foo == 'Fe': return "Fe mass fraction I think?"
    if foo == 'FUV': return "G$_0$ [0.6 Habing]"
    if foo == 'lFUV': return "log(G$_0$ [0.6 Habing])"
    if foo == 'FUVmw': return r"$\langle$G$_{\mathrm{0}}\rangle_{\mathrm{mass}}$ [0.6 Habing]"
    if foo == 'FUV_amb': return "G$_0$ (ambient) [0.6 Habing]"
    if foo == 'nH_DNG': return "H density of DNG [cm$^{-3}$]"
    if foo == 'dr_DNG': return "Thickness of DNG layer [pc]"
    if foo == 'm_DIG': return "m$_{\mathrm{DIG}}$ [M$_{\odot}$]"
    if foo == 'nH_DIG': return "n$_{\mathrm{H,DIG}}$ [cm$^{-3}$]"
    if foo == 'R': return "$R$ [kpc]"
    if foo == 'vel_disp_gas': return r"$\sigma_{\mathrm{v}}$ of gas [km s$^{-1}$]"
    if foo == 'sigma_gas': return r"$\sigma_{\mathrm{v,\perp}}$ of gas [km s$^{-1}$]"
    if foo == 'sigma_star': return r"$\sigma_{\mathrm{v,\perp}}$ of star [km s$^{-1}$]"
    if foo == 'surf_gas': return "$\Sigma_{\mathrm{gas}}$ [M$_{\odot}$ kpc$^{-2}$]"
    if foo == 'surf_star': return "$\Sigma_{\mathrm{*}}$ [M$_{\odot}$ kpc$^{-2}$]"
    if foo == 'L_CII': return 'L$_{\mathrm{[CII]}}$ [L$_{\odot}$]'
    if foo == 'lL_CII': return 'log(L$_{\mathrm{[CII]}}$ [L$_{\odot}$])'
    if foo == 'L_OI': return 'L$_{\mathrm{[OI]}\,63\mu\mathrm{m}}$ [L$_{\odot}$]'
    if foo == 'lL_OI': return 'log(L$_{\mathrm{[OI]}\,63\mu\mathrm{m}}$ [L$_{\odot}$])'
    if foo == 'L_OIII': return 'L$_{\mathrm{[OIII]}\,88\mu\mathrm{m}}$ [L$_{\odot}$]'
    if foo == 'lL_OIII': return 'log(L$_{\mathrm{[OIII]}\,88\mu\mathrm{m}}$ [L$_{\odot}$])'
    if foo == 'S_CII': return 'S$_{\mathrm{[CII]}}$ [mJy]'
    if foo == 'x_e': return 'Electron fraction [H$^{-1}$]'
    if foo == 'f_CII': return '(mass of carbon in CII state)/(mass of carbon in CIII state) [%]'
    if foo == 'f_ion': return 'Ionized gas mass fraction [%]'
    if foo == 'f_gas': return 'Gas mass fraction M$_{\mathrm{gas}}$/(M$_{\mathrm{gas}}$+M$_{\mathrm{*}}$) [%]'

def arrow_length(n):
    '''
    Purpose
    ---------
    '''

    arrow   =   10.**(np.log10(n))-10.**(np.log10(n)-0.15)
    return arrow

def trim_df(file,skip,maxcol):
    '''
    Purpose
    ---------
    Comment columns to trim a dataframe
    '''

    script_out  =    open('sigame/temp/trim_df.txt','w')
    i           =   0
    save        =   'n'
    with open(file) as f:
        for line in f:                   # loop over the rows
            if i<skip:
                line1 = line
            if i>=skip:                      # skipping lines
                fields  =   line.split('\t')        # parse the columns
                if len(fields) > maxcol:
                    save        =   'y'
                    fields[maxcol-1] = fields[maxcol-1]+' #'
                line1   =   '\t'.join(fields)
            i   +=   1
            script_out.write(line1)
    script_out.close()
    if save == 'y': file    =   'sigame/temp/trim_df.txt'          # There were too many columns so save in trim_df.txt
    return(file)


