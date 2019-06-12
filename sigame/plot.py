# coding=utf-8
"""
Module: plot
"""

# Import other SIGAME modules
import sigame.global_results as glo
import sigame.aux as aux
import sigame.galaxy as gal

# Import other modules
# from __future__ import division
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as FuncAnimation
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import numpy as np
import pdb as pdb
from scipy.interpolate import RegularGridInterpolator
import matplotlib.ticker as ticker
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
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, ScalarFormatter, MaxNLocator
import sympy as sy
from sympy.solvers import solve
from argparse import Namespace


#===============================================================================
"""  Load parameters """
#-------------------------------------------------------------------------------

params                      =   aux.load_parameters()
g                           =   globals()
for key,val in params.items():
    exec(key + '=val',g)

#===============================================================================
"""  Basic plotting """
#-------------------------------------------------------------------------------

def simple_plot(**kwargs):
    '''A function to standardize all plots

    Plots that can be created this way:
        - 1. errorbar plot (with x and/or y errorbars)
        - 2. line
        - 3. histogram
        - 4. markers
        - 5. bar
        - 6. hexbin
        - 7. contour plot
        - 8. scatter plot
        - 9. hatched/filled region

    The '1' below can be replaced by '2', '3', '4' etc for several plotting options in the same figure.


    Parameters
    ----------

    add : bool
        If True add to existing axis object, otherwise new figure+axes will be created, default: False

    fig : int
        Figure number, default: 0

    figsize : tuple
        Figure size, default: (8,6)

    figname : str
        If defined, a figure will be saved with this path+name, default = not defined (no figure saved)

    figtype : str
        Format of figure, default: 'png'

    figres : float
        The dpi of the saved figure, default: 1000

    fontsize : float
        Fontsize of tick labels, default: 15

    x1 : list or numpy array
        x values

    y1 : list or numpy array
        y values

    xr : list
        x range

    yr : list
        y range

    xlog : bool
        If True: x-axis will be in log units

    ylog : bool
        If True: y-axis will be in log units

    fill1 : str
        If defined, markers will be used ('y' for filled markers, 'n' for open markers), default: 'y'

    ls1 : str
        Linestyle, default: 'None' (does markers by default)

    ma1 : str
        Marker type, default: 'x'

    ms1 : int/float
        Marker size, default: 5

    mew1 : int/float
        Marker edge width, default: 2

    col1 : str
        Color of markers/lines, default: 'k'

    ecol1 : str
        Edgecolor, default: 'k'

    lab1 : str
        Label for x1,y1 points, default: ''

    alpha1 : float
        Transparency fraction, default: 1.1

    dashes1 : str
        Custom-made dashes/dots, default = ''

    legend : bool
        Whether to plot legend or not, default: False

    leg_fs : float
        Legend fontsize, default: not defined (same as fontsize for labels/tick marks)

    legloc : str or list of coordinates
        Legend location, default: 'best'

    cmap1 : str
        Colormap for contour plots, default: 'viridis'

    xlab : str
        x axis label, default: no label

    ylab : str
        y axis label, default: no label

    title : str
        Plot title, default: no title

    xticks: bool
        Whether to put x ticks or not, default: True

    yticks: bool
        Whether to put y ticks or not, default: True

    lab_to_tick : int/float
        If axis labels should be larger than tick marks, say how much here, default: 1.0

    lex1,uex1: list or numpy array
        Lower and upper errorbars on x1, default: None, **options**:
        If an element in uex1 is 0 that element will be plotted as upper limit in x

    ley1,uey1: list or numpy array
        lower and upper errorbars on y1, default: None, **options**:
        If an element in uey1 is 0 that element will be plotted as upper limit in y

    histo1 : bool
        Whether to make a histogram of x1 values, default: False

    histo_real1 : bool
        Whether to use real values for histogram or percentages on y axis, default: False

    bins1 : int
        Number of bins in histogram, default: 100

    weights1 : list or numpy array
        weights to histogram, default: np.ones(len(x1))

    hexbin1 : bool
        If True, will make hexbin contour plot, default: False

    contour_type1 : str
        If defined, will make contour plot, default: not defined, **options**:
        plain: use contourf on colors alone, optionally with contour levels only (no filling),
        hexbin: use hexbin on colors alone,
        median: use contourf on median of colors,
        mean: use contourf on mean of colors,
        sum: use contourf on sum of colors

    barwidth1 : float
        If defined, will make bar plot with this barwidth

    scatter_color1 : list or numpy array
        If defined, will make scatter plot with this color, default: not defined (will not do scatter plot)

    colormin1 : float
        Minimum value for colorbar in scatter plot, default: not defined (will use all values in scatter_color1)

    lab_colorbar : str
        Label for colorbar un scatter plot, default: not defined (will not make colorbar)

    hatchstyle1 : str
        If defined, make hatched filled region, default: not defined (will not make hatched region), **options**:
        if set to '', fill with one color,
        otherwise, use '/' '//' '///'' etc.

    text : str
        If defined, add text to figure with this string, default: not defined (no text added)

    textloc : list
        Must be specified in normalized axis units, default: [0.1,0.9]

    textbox : bool
        If True, put a box around text, default: False

    fontsize_text : int/float
        Fontsize of text on plot, default: 0/7 * fontsize

    grid : bool
        Turn on grid lines, default: False

    SC_return : bool
        Return scatter plot object or not, default = False

    '''

    # Set fontsize
    if mpl.rcParams['ytick.labelsize'] == 'medium':
        fontsize            =   15
    if 'fontsize' in kwargs:
        fontsize            =   kwargs['fontsize']
        mpl.rcParams['ytick.labelsize'] = fontsize
        mpl.rcParams['xtick.labelsize'] = fontsize
    else:
        fontsize            =   15
    lab_to_tick         =   1.
    if 'lab_to_tick' in kwargs: lab_to_tick = kwargs['lab_to_tick']
    textcol             =   'black'
    if 'textcol' in kwargs: textcol = kwargs['textcol']
    fontsize_text       =   fontsize*0.7

    # Get axes object
    if 'add' in kwargs:
        ax1                 =   plt.gca()
    else:
        fig                 =   0                                       # default figure number
        if 'fig' in kwargs: fig = kwargs['fig']
        figsize             =   (8,6)                                   # slightly smaller figure size than default
        if 'figsize' in kwargs: figsize = kwargs['figsize']
        fig                 =   plt.figure(fig,figsize=figsize)
        ax1                 =   fig.add_subplot(1,1,1)
    # if kwargs.has_key('aspect'): ax1.set_aspect(kwargs['aspect'])

    if 'plot_margin' in kwargs:
        plt.subplots_adjust(left=kwargs['plot_margin'], right=1-kwargs['plot_margin'], top=1-kwargs['plot_margin'], bottom=kwargs['plot_margin'])
        # pdb.set_trace()

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
    zorder0             =   100
    fillstyle0          =   'full'

    # Set axis settings
    xlab,ylab           =   '',''
    if 'xlab' in kwargs:
        ax1.set_xlabel(kwargs['xlab'],fontsize=fontsize*lab_to_tick)
    if 'ylab' in kwargs:
        ax1.set_ylabel(kwargs['ylab'],fontsize=fontsize*lab_to_tick)
    if 'title' in kwargs:
        ax1.set_title(kwargs['title'],fontsize=fontsize*lab_to_tick)
    if 'histo' in kwargs:
        ax1.set_ylabel('Number fraction [%]',fontsize=fontsize*lab_to_tick)

    # Set aspect here before colorbar
    # if kwargs.has_key('aspect'):
    #     ax1.set_aspect(kwargs['aspect'])

    # Add lines/points to plot
    for i in range(1,20):
        done            =   'n'
        if 'x'+str(i) in kwargs:
            if 'x'+str(i) in kwargs: x = kwargs['x'+str(i)]
            if 'y'+str(i) in kwargs: y = kwargs['y'+str(i)]
            # If no x values, make them up
            if not 'x'+str(i) in kwargs: x = np.arange(len(y))+1
            ls              =   ls0
            lw              =   lw0
            mew             =   mew0
            ma              =   ma0
            col             =   col0
            mfc             =   col0
            ecol            =   ecol0
            ms              =   ms0
            lab             =   lab0
            ls              =   ls0
            fill            =   fill0
            alpha           =   alpha0
            cmap            =   cmap0
            bins            =   bins0
            zorder          =   zorder0
            fillstyle       =   fillstyle0
            if 'ls'+str(i) in kwargs:
                if kwargs['ls'+str(i)] != 'None': ls = kwargs['ls'+str(i)]
                if kwargs['ls'+str(i)] == 'None': ls = 'None'
            if 'lw'+str(i) in kwargs: lw = kwargs['lw'+str(i)]
            if 'lw' in kwargs: lw = kwargs['lw'] # or there is a general keyword for ALL lines...
            if 'mew'+str(i) in kwargs: mew = kwargs['mew'+str(i)]
            if 'ma'+str(i) in kwargs: ma = kwargs['ma'+str(i)]
            if 'ms'+str(i) in kwargs: ms = kwargs['ms'+str(i)]
            if 'col'+str(i) in kwargs: col, mfc = kwargs['col'+str(i)], kwargs['col'+str(i)]
            if 'ecol'+str(i) in kwargs: ecol = kwargs['ecol'+str(i)]
            if 'lab'+str(i) in kwargs: lab = kwargs['lab'+str(i)]
            if 'lab'+str(i) in kwargs: legend = 'on' # do make legend
            legend          =   False
            if 'legend' in kwargs: legend = kwargs['legend']
            if 'ls'+str(i) in kwargs: ls = kwargs['ls'+str(i)]
            if 'alpha'+str(i) in kwargs: alpha = kwargs['alpha'+str(i)]
            if 'cmap'+str(i) in kwargs: cmap = kwargs['cmap'+str(i)]
            if 'zorder'+str(i) in kwargs: zorder = kwargs['zorder'+str(i)]
            if 'fill'+str(i) in kwargs:
                fill                = kwargs['fill'+str(i)]
                if kwargs['fill'+str(i)] == 'y': fillstyle  = 'full'
                if kwargs['fill'+str(i)] == 'n': fillstyle, mfc, alpha  = 'none', 'None', None


            # ----------------------------------------------
            # 1. Errorbar plot
            # Errorbars/arrows in x AND y direction
            if 'lex'+str(i) in kwargs:
                if 'ley'+str(i) in kwargs:
                    for x1,y1,lex,uex,ley,uey in zip(x,y,kwargs['lex'+str(i)],kwargs['uex'+str(i)],kwargs['ley'+str(i)],kwargs['uey'+str(i)]):
                        ax1.errorbar(x1,y1,color=col,linestyle="None",fillstyle=fillstyle,xerr=[[lex],[uex]],yerr=[[ley],[uey]],elinewidth=lw,capsize=0,\
                            capthick=0,marker=kwargs['ma'+str(i)],label=kwargs['lab'+str(i)])
            # Errorbars/arrows in x direction
            if 'lex'+str(i) in kwargs:
                # print('>> Adding x errorbars!')
                for x1,y1,lex,uex in zip(x,y,kwargs['lex'+str(i)],kwargs['uex'+str(i)]):
                    if uex > 0: # not upper limit, plot errobars
                        ax1.errorbar(x1,y1,color=col,linestyle="None",fillstyle=fillstyle,xerr=[[lex],[uex]],elinewidth=lw,capsize=0,\
                            capthick=0,marker=kwargs['ma'+str(i)])
                    if uex == 0: # upper limit, plot arrows
                        ax1.errorbar(x1,y1,color=col,xerr=lex,\
                           xuplims=True,linestyle="None",fillstyle=fillstyle,linewidth=lw,mew=0,capthick=lw*2)
            # Errorbars/arrows in y direction
            if 'ley'+str(i) in kwargs:
                # print('>> Adding y errorbars!')
                for x1,y1,ley,uey in zip(x,y,kwargs['ley'+str(i)],kwargs['uey'+str(i)]):
                    if uey > 0: # not upper limit, plot errorbars
                        ax1.errorbar(x1,y1,color=col,linestyle='None',fillstyle=fillstyle,yerr=[[ley],[uey]],elinewidth=lw,\
                            capsize=0,capthick=0,marker=ma)
                    if uey == 0: # upper limit, plot arrows
                        ax1.errorbar(x1,y1,color=col,yerr=ley,\
                           uplims=True,linestyle="None",fillstyle=fillstyle,linewidth=lw,mew=0,capthick=lw*2)
                    continue

            # ----------------------------------------------
            # 2. Line connecting the dots
            if 'y'+str(i) in kwargs:

                if type(kwargs['y'+str(i)]) == str: y = ax1.get_ylim()
                if 'dashes'+str(i) in kwargs:
                    # print('>> Line plot!')
                    ax1.plot(x,y,linestyle=ls,color=col,lw=lw,label=lab,dashes=kwargs['dashes'+str(i)],zorder=zorder)
                    continue
                else:
                    if 'ls'+str(i) in kwargs:
                        # print('>> Line plot!')
                        ax1.plot(x,y,linestyle=ls,color=col,lw=lw,label=lab,zorder=zorder)
                        continue

            # ----------------------------------------------
            # 3. Histogram
            if 'histo'+str(i) in kwargs:
                # print('>> Histogram!')
                if ls == 'None': ls = '-'
                weights             =   np.ones(len(x))
                if 'bins'+str(i) in kwargs: bins = kwargs['bins'+str(i)]
                if 'weights'+str(i) in kwargs: weights = 'weights'+str(i) in kwargs
                if 'histo_real'+str(i) in kwargs:
                    make_histo(x,bins,col,lab,percent=False,weights=weights,lw=lw,ls=ls)
                else:
                    make_histo(x,bins,col,lab,percent=True,weights=weights,lw=lw,ls=ls)
                continue

            # ----------------------------------------------
            # 4. Marker plot
            if 'fill'+str(i) in kwargs:
                # print('>> Marker plot!')
                ax1.plot(x,y,linestyle='None',color=col,marker=ma,mew=mew,ms=ms,fillstyle=fillstyle,alpha=alpha,markerfacecolor=mfc,zorder=zorder,label=lab)
                continue

            # ----------------------------------------------
            # 5. Bar plot
            if 'barwidth'+str(i) in kwargs:
                # print('>> Bar plot!')
                plt.bar(x,y,width=kwargs['barwidth'+str(i)],color=col,alpha=alpha)
                continue

            # ----------------------------------------------
            # 6. Hexbin contour bin
            if 'hexbin'+str(i) in kwargs:
                # print('>> Hexbin contour plot!')
                bins                =   300
                if 'bins'+str(i) in kwargs: bins = kwargs['bins'+str(i)]
                if 'alpha'+str(i) in kwargs: alpha = kwargs['alpha'+str(i)]
                if 'col'+str(i) in kwargs:
                    colors          =   kwargs['col'+str(i)]
                    CS              =   ax1.hexbin(x, y, C=colors, gridsize=bins, cmap=cmap, alpha=alpha)
                else:
                    CS              =   ax1.hexbin(x, y, gridsize=bins, cmap=cmap, alpha=alpha)
                continue

            # ----------------------------------------------
            # 7. Contour map

            if 'contour_type'+str(i) in kwargs:
                CS                  =   make_contour(i,fontsize,kwargs=kwargs)

            if 'colorbar'+str(i) in kwargs:
                if kwargs['colorbar'+str(i)]:
                    if only_one_colorbar == 1: pad = 0
                    if only_one_colorbar < 0: pad = 0.03
                    ax2 = ax1.twinx()
                    ax2.get_xaxis().set_visible(False)
                    ax2.get_yaxis().set_visible(False)
                    divider = make_axes_locatable(ax2)
                    cax = divider.append_axes("right", size="5%", pad=pad)
                    cbar                    =   plt.colorbar(CS,cax=cax)
                    cbar.set_label(label=kwargs['lab_colorbar'+str(i)],size=fontsize-5)   # colorbar in it's own axis
                    cbar.ax.tick_params(labelsize=fontsize-5)
                    only_one_colorbar       =   -1
                    plt.axes(ax1)

            # ----------------------------------------------
            # 8. Scatter plot (colored according to a third parameter)
            if 'scatter_color'+str(i) in kwargs:
                # print('>> Scatter plot!')
                SC              =   ax1.scatter(x,y,marker=ma,lw=mew,s=ms,c=kwargs['scatter_color'+str(i)],cmap=cmap,alpha=alpha,label=lab,edgecolor=ecol,zorder=zorder)
                if 'colormin'+str(i) in kwargs: SC.set_clim(kwargs['colormin'+str(i)],max(kwargs['scatter_color'+str(i)]))
                if 'lab_colorbar' in kwargs:
                    if only_one_colorbar > 0:
                        cbar                    =   plt.colorbar(SC,pad=0)
                        cbar.set_label(label=kwargs['lab_colorbar'],size=fontsize-2)   # colorbar in it's own axis
                        cbar.ax.tick_params(labelsize=fontsize-2)
                        only_one_colorbar       =   -1
                continue

            # ----------------------------------------------
            # 8. Filled region
            if 'hatchstyle'+str(i) in kwargs:
                # print('>> Fill a region!')
                from matplotlib.patches import Ellipse, Polygon
                if kwargs['hatchstyle'+str(i)] != '': ax1.add_patch(Polygon([[x[0],y[0]],[x[0],y[1]],[x[1],y[1]],[x[1],y[0]]],closed=True,fill=False,hatch=kwargs['hatchstyle'+str(i)],color=col),zorder=zorder)
                if kwargs['hatchstyle'+str(i)] == '': ax1.fill_between(x,y[0],y[1],facecolor=col,color=col,alpha=alpha,lw=0,zorder=zorder)
                continue

    # Log or not log?
    if 'xlog' in kwargs:
        if kwargs['xlog']: ax1.set_xscale('log')
    if 'ylog' in kwargs:
        if kwargs['ylog']: ax1.set_yscale('log')

    # Legend
    if legend:
        legloc          =   'best'
        if 'legloc' in kwargs: legloc = kwargs['legloc']
        frameon         =   not 'frameon' in kwargs or kwargs['frameon']          # if "not" true that frameon is set, take frameon to kwargs['frameon'], otherwise always frameon=True
        handles1, labels1     =   ax1.get_legend_handles_labels()
        leg_fs          =   fontsize#int(fontsize*0.7)
        if 'leg_fs' in kwargs: leg_fs = kwargs['leg_fs']
        leg = ax1.legend(loc=legloc,fontsize=leg_fs,numpoints=1,scatterpoints = 1,frameon=frameon)
        leg.set_zorder(zorder)

    # Add text to plot
    if 'text' in kwargs:
        textloc             =   [0.1,0.95]
        if 'textloc' in kwargs: textloc = kwargs['textloc']
        fontsize_text       =   fontsize
        if 'textfs' in kwargs: fontsize_text=kwargs['textfs']
        if 'textbox' in kwargs:
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

    if 'grid' in kwargs: ax1.grid()

    if 'xticks' in kwargs:
        if kwargs['xticks']:
            ax1.set_xticks(kwargs['xticks'])
            ax1.set_xticklabels(str(_) for _ in kwargs['xticks'])
        else:
            ax1.set_xticks(kwargs['xticks'])
            ax1.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    if 'xticklabels' in kwargs: ax1.set_xticklabels(kwargs['xticklabels'])

    if 'yticks' in kwargs:
        if kwargs['yticks']:
            ax1.set_yticks(kwargs['yticks'])
            ax1.set_yticklabels(str(_) for _ in kwargs['yticks'])
        else:
            ax1.set_yticks(kwargs['yticks'])
            ax1.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    if 'yticklabels' in kwargs: ax1.set_yticklabels(kwargs['yticklabels'])

    if 'xr' in kwargs: ax1.set_xlim(kwargs['xr'])
    if 'yr' in kwargs: ax1.set_ylim(kwargs['yr'])

    # plt.tight_layout()

    # Save plot if figure name is supplied
    if 'figres' in kwargs:
        dpi = kwargs['figres']
    else:
        dpi = 1000
    if 'figname' in kwargs:
        figname = kwargs['figname']
        figtype = 'png'
        if 'figtype' in kwargs: figtype = kwargs['figtype']
        plt.savefig(figname+'.'+figtype, format=figtype, dpi=dpi) # .eps for paper!

    if 'show' in kwargs:
        plt.show(block=False)


    # restoring defaults
    # mpl.rcParams['xtick.labelsize'] = u'medium'
    # mpl.rcParams['ytick.labelsize'] = u'medium'

    if 'fig_return' in kwargs:
        return fig

def make_contour(i,fontsize,kwargs):
    '''Makes contour plot (called by simple_plot)

    Parameters
    ----------
    contour_type: str
        Method used to create contour map (see simple_plot)

    '''

    # print('Contour plot!')

    ax1                 =   plt.gca()

    linecol0            =   'k'
    cmap0               =   'viridis'
    alpha0              =   1.1
    nlev0               =   10
    only_one_colorbar   =   1

    # Put on regular grid!
    if 'y'+str(i) in kwargs:

        y               =   kwargs['y'+str(i)]
        x               =   kwargs['x'+str(i)]
        colors          =   kwargs['col'+str(i)]
        linecol         =   linecol0
        if 'linecol'+str(i) in kwargs: linecol = kwargs['linecol'+str(i)]
        cmap            =   cmap0
        if 'cmap'+str(i) in kwargs: cmap = kwargs['cmap'+str(i)]
        alpha           =   alpha0
        if 'alpha'+str(i) in kwargs: alpha = kwargs['alpha'+str(i)]
        nlev            =   nlev0
        if 'nlev'+str(i) in kwargs: nlev = kwargs['nlev'+str(i)]

        if kwargs['contour_type'+str(i)] == 'plain':

            if cmap == 'none':
                print('no cmap')
                CS = ax1.contour(x,y,colors, nlev, colors=linecol)
                plt.clabel(CS, fontsize=9, inline=1)

            if 'colormin'+str(i) in kwargs:
                # print('Colormap with a minimum value')
                CS = ax1.contourf(x,y,colors, nlev, cmap=cmap)
                ax1.contourf(x,y,colors, levels=kwargs['colormin'+str(i)], colors='k')

            else:
                if 'alpha'+str(i) in kwargs:
                    print('with alpha')
                    CS = ax1.contourf(x,y,colors, nlev, cmap=cmap, alpha=kwargs['alpha'+str(i)])
                if not 'alpha'+str(i) in kwargs:
                    # print('without alpha')
                    CS = ax1.contourf(x,y,colors, nlev, cmap=cmap)#, lw=0, antialiased=True)

        if kwargs['contour_type'+str(i)] == 'hexbin':
            CS              =   ax1.hexbin(x, y, C=colors, cmap=cmap)

        if kwargs['contour_type'+str(i)] == 'mesh':
            CS              =   ax1.pcolormesh(x,y,colors, cmap=cmap)

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
            if 'nlev'+str(i) in kwargs: nlev0 = kwargs['nlev'+str(i)]
            CS               =   ax1.contourf(gridx, gridy, z.T, nlev0, cmap=cmap)
            mpl.rcParams['contour.negative_linestyle'] = 'solid'
            # CS               =   ax1.contour(gridx, gridy, z.T, 5, colors='k')
            # plt.clabel(CS, inline=1, fontsize=10)
            if 'colormin'+str(i) in kwargs: CS.set_clim(kwargs['colormin'+str(i)],max(z.reshape(lx*ly,1)))
            if 'colormin'+str(i) in kwargs: print(kwargs['colormin'+str(i)])
            CS.cmap.set_under('k')

    # plt.subplots_adjust(left=0.13, right=0.94, bottom=0.14, top=0.95)

    return CS

def histos(**kwargs):
    '''Makes histograms of all (particles in all) galaxies in the sample on the same plot.

    Parameters
    ---------
    bins : int/float
        Number of bins, default: 100

    add : bool
        If True, add to an existing plot, default: False

    one_color : bool
        If True, use only one color for all lines, default: True

    fs_labels : int/float
        Fontsize, default: 15

    '''

    GR                      =   glo.global_results()

    # set label sizes
    mpl.rcParams['xtick.labelsize'] = fs_labels
    mpl.rcParams['ytick.labelsize'] = fs_labels

    # Ask a lot of questions!!
    data_type       =   input('For which data type? [default: sim] '+\
                        '\n gmc for Giant Molecular Clouds'+\
                        '\n sim for raw simulation data (gas/stars/dark matter)'+\
                        '\n dng for Diffuse Neutral Gas'+\
                        '\n dig for Diffuse Ionized Gas'+\
                        '...? ')
    if data_type == '': data_type =   'sim'
    data_type = data_type.upper()

    if data_type == 'SIM':
        sim_type        =   input('\nGas or star or dark matter (dm)? [default: gas] ... ')
        if sim_type == '': sim_type =   'gas'

    # Start plotting (fignum = 1: first plot)
    if not add:
        plt.close('all')
        plt.ion()
    redo        =   'y'
    fignum      =   1
    while redo == 'y':
        if fignum >1:
            quant        =   input('\nOver what quantity? [default: m]... ')
            if quant == '': quant =   'm'
        histos1     =   np.zeros([len(GR.galnames),bins+2])
        histos2     =   np.zeros([len(GR.galnames),bins+3])
        igal        =   0
        Ngal        =   0
        indices     =   []
        for gal_index in range(len(GR.galnames)): #TEST
            zred,galname        =   GR.zreds[gal_index],GR.galnames[gal_index]
            gal_ob              =   gal.galaxy(gal_index=gal_index)

            if data_type == 'SIM': dat0 = aux.load_temp_file(gal_ob=gal_ob,sim_type=sim_type)
            if data_type == 'GMC': dat0 = gal_ob.particle_data.get_raw_data(data='ISM')[data_type]
            if data_type in ['DNG','DIG']:
                dat0 = gal_ob.particle_data.get_raw_data(data='ISM')['dif']
                if data_type == 'DNG': dat0 = dat0[dat0['m_DNG'] > dat0['m_DIG']]
                if data_type == 'DIG': dat0 = dat0[dat0['m_DIG'] > dat0['m_DNG']]

            # Choose what to make histogram over and start figure
            if gal_index == 0:
                print('\nOver what quantity? Options:')
                keys = ''
                for key in dat0.keys(): keys = keys + key + ', '

                quant           =   input('[default: m]... ')
                if quant == '': quant =   'm'

                weigh           =   input('\nMass or number-weighted (m vs n)? [default: n] ... ')
                if weigh == '': weigh =   'n'

                logx            =   input('\nLogarithmix x-axis? [default: y] ... ')
                if logx == '': logx =   'y'

                logy            =   input('\nLogarithmix y-axis? [default: y] ... ')
                if logy == '': logy =   'y'

                if add:
                    print('\nadding to already existing figure')
                    fig         =   plt.gcf()
                    ax1         =   fig.add_subplot(add[0],add[1],add[2])
                else:
                    print('\ncreating new figure')
                    fig         =   plt.figure(fignum,figsize=(8,6))
                    ax1         =   fig.add_subplot(1,1,1)

            # Weigh the data (optional) and calculate histogram
            if quant == 'm_mol': dat0['m_mol'] = dat0['f_H2'].values*dat0['m'].values
            dat         =   dat0[quant].values.squeeze()
            if weigh == 'm': w           =   dat0['m']
            if weigh == 'n': w           =   1./len(dat0)
            if data_type == 'SIM':
                if quant == 'nH': dat = dat/(mH*1000.)/1e6 # Hydrogen only per cm^3
            if logx == 'y':
                if quant == 'Z': dat[dat == 0] = 1e-30 # to avoid crash if metallicity is zero
                dat = np.log10(dat)
                i_nan   =   np.isnan(dat)
                if weigh == 'm':  w       =   w[i_nan == False]
                dat     =   dat[i_nan == False]
            # print('min and max: %s and %s ' % (np.min(dat[dat > -100]),dat.max()))
            if logy == 'n':
                if weigh == 'm':  w       =   w[dat > -10.**(20)]
                dat     =   dat[dat > -10.**(20)]
            if logy == 'y':
                if weigh == 'm':  w       =   w[dat > -20]
                dat     =   dat[dat > -20]
            if weigh == 'n':    hist        =   np.histogram(dat,bins=bins)
            if weigh == 'm':    hist        =   np.histogram(dat,bins=bins,weights=w)
            if 'f_HI' in quant:
                print('Particles are above 0.9: %s %%' % (1.*len(dat[dat > 0.9])/len(dat)*100.))
                print('Particles are below 0.1: %s %%' % (1.*len(dat[dat < 0.1])/len(dat)*100.))
            if 'f_H2' in quant:
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
                ax1.plot(hist2[0:len(hist1)]+wid/2,hist1,ls='-',ds='steps',color=col[igal],label='G'+str(int(igal+1)))

            igal             +=  1
            Ngal             +=  1
            indices.append(gal_index)

        histos1             =   histos1[0:Ngal,:]
        histos2             =   histos2[0:Ngal,:]

        if one_color:

            # Plot as background the 2 sigma distribution around the mean in each bin
            minhistos1,maxhistos1,meanhistos1       =   np.zeros(bins+2), np.zeros(bins+2), np.zeros(bins+2)
            for i in range(0,bins+2):
                meanhistos1[i]     =   np.mean(histos1[:,i])
                minhistos1[i]      =   meanhistos1[i]-2.*np.std(histos1[:,i])
                maxhistos1[i]      =   meanhistos1[i]+2.*np.std(histos1[:,i])
            ax1.fill_between(hist2[0:len(hist1)]+wid/2, minhistos1, maxhistos1, facecolor='lightgreen', alpha=0.5, lw=0)

            # Now plot actual histograms
            for i in range(Ngal):
                # pdb.set_trace()
                hist2           =   histos2[i,:]
                hist1           =   histos1[i,:]
                ax1.plot(hist2[0:len(hist1)]+wid/2,hist1,ls='-',ds='steps',color='teal',label='G'+str(int(indices[i]+1)),alpha=0.7,lw=1)

            # Now plot mean of histograms
            if Ngal > 1: ax1.plot(hist2[0:len(hist1)]+wid/2,meanhistos1,ls='-',ds='steps',color='blue',lw=1)
        # if logx == 'y':     ax1.set_xscale('log')

        # labels and ranges
        xl          =   getlabel(quant)
        if logy    == 'y': xl = getlabel('l'+quant)
        ax1.set_xlabel(xl,fontsize=fs_labels)
        if weigh     == 'n': ax1.set_ylabel('Number fraction [%]',fontsize=fs_labels)
        if weigh     == 'm': ax1.set_ylabel('Mass fraction [%]',fontsize=fs_labels)
        ax1.set_ylim([max(hist1)/1e4,max(hist1)*10.])

        if not add:
            fig.canvas.draw()

            # axes ranges
            if xlim: ax1.set_xlim(xlim)
            if ylim:
                if logy == 'y':
                    ax1.set_ylim([10.**ylim[0],10.**ylim[1]])
                else:
                    ax1.set_ylim(ylim)
            fig.canvas.draw()

            if logy    == 'y': ax1.set_yscale('log')

            savefig         =   input('Save figure? [default: n] ... ')
            if savefig == '': savefig = 'n'
            if savefig == 'y':
                if not os.path.exists('plots/histos/'):
                    os.makedirs('plots/histos/')
                name            =   input('Figure name? plots/histos/... ')
                if name == '':
                    name = galname + '_' + data_type + '_' + quant
                plt.savefig('plots/histos/'+name+'.png', format='png', dpi=250) # .eps for paper!

            # New figure?
            if add:
                redo = 'n'
            else:
                redo        =   input('plot another quantity? [default: n] ... ')
                if redo == '': redo='n'
                if redo == 'n':
                    # restoring defaults
                    mpl.rcParams['xtick.labelsize'] = u'medium'
                    mpl.rcParams['ytick.labelsize'] = u'medium'
                    # break
                fignum      +=1
                changex, changey  =   'n','n'

#===============================================================================
""" Global gas properties in simulation """
#-------------------------------------------------------------------------------

def SFR_Mstar(**kwargs):
    '''Plot of SFR vs stellar mass for one or more redshift groups

    Parameters
    ----------

    color : str
        What to color-code the galaxies by, default: age

    Example
    -------
    >>> import sigame as si
    >>> si.plot.SFR_Mstar(color='age')
    '''

    # color galaxies by...
    color = 'age'

    plt.close('all')

    redshift        =   float(z1.replace('z',''))

    # Get models
    GR              =   glo.global_results()

    # Get M_star range
    xr              =   axis_range(GR.M_star,log=True)

    # Get observed MS SFR-Mstar relation
    MS =   aux.MS_SFR_Mstar(Mstars=xr,redshift=redshift)

    if color == 'age':
        if z1 == 'z0': colors,lab = GR.mw_age/1000.,'Mass-weighted stellar age [Gyr]'
        if z1 == 'z6': colors,lab = GR.mw_age,'Mass-weighted stellar age [Myr]'

    # Plot MS and models
    simple_plot(add='y',
                xlab=getlabel('M_star'), ylab=getlabel('SFR'),
                xlog='y', ylog='y', xr=xr, legloc='upper left', frameon=False,
                x1=GR.M_star, y1=GR.SFR, ma1=galaxy_marker,
                scatter_color1=colors, ms1=80, alpha1=1, mew1=0,
                lab1='$z\sim$'+str(int(redshift))+' model galaxies (this work)$_{}$',
                zorder1=100, lab_colorbar=lab,
                x2=xr, y2=MS, lw2=2, col2='k',
                lab2='$z\sim$'+str(int(redshift))+' main sequence [Speagle+14]$_{}$',ls2='--',
                legend=True)

    # more updated SFMS at z~6
    from astropy.cosmology import FlatLambdaCDM

    cosmo = FlatLambdaCDM(H0=hubble*100.,
                          Om0=omega_m,
                          Ob0=1-omega_m-omega_lambda)

    age = cosmo.age(redshift).value # Gyr
    MS_Iyer = 10.**((0.80-0.017*age)*np.log10(xr)-(6.487-0.039*age))

    simple_plot(add='y',
                x3=xr, y3=MS_Iyer, col3='orange',lw3=2, ls3='--',
                lab3='$z\sim$'+str(int(redshift))+' main sequence [Iyer+18]$_{}$',
                legend=True)

    # Spread around MS
    simple_plot(add='y',\
        x3=xr,y3=10.**(np.log10(MS)-0.2),col3='k',lw3=1,ls3='--',\
        x4=xr,y4=10.**(np.log10(MS)+0.2),col4='k',lw4=1,ls4='--',\
        x5=xr,y5=10.**(np.log10(MS)-3.*0.2),col5='k',lw5=1,ls5=':',\
        x6=xr,y6=10.**(np.log10(MS)+3.*0.2),col6='k',lw6=1,ls6=':',legend=False)

    if z1 == 'z6':
        # Add z ~ 6 LBGs and LAEs
        columns         =   ['ID','age(M)','mass(G)','SFR_Lya','SFR_UV','E(B-V)']
        J16             =   pd.read_table('Tables/Observations/SFR_Mstar/Jiang16.txt',names=columns,skiprows=1,sep=r'\s*',engine='python')
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
        ax1             =   plt.gca()
        simple_plot(add='y',x1=M_star_J13[age_J16 > 30],y1=SFR_J16[age_J16 > 30],fill1='y',ma1='x',col1='r',ms1=10,mew1=2,lab1='Old $z\sim6$ LBGs/LAEs [Jiang+16]',\
                x2=M_star_J13[age_J16 < 30],y2=SFR_J16[age_J16 < 30],fill2='y',ma2='+',col2='b',ms2=11,mew2=2,lab2='Young $z\sim6$ LBGs/LAEs [Jiang+16]',legloc='upper left', legend=True)

    plt.tight_layout()
    plt.show(block=False)
    if not os.path.exists('plots/galaxy_sims/SFR_Mstar/'):
        os.makedirs('plots/galaxy_sims/SFR_Mstar/')
    plt.savefig('plots/galaxy_sims/SFR_Mstar/M_SFR_sample_'+z1+'.png', dpi=200, format='png') # .eps for paper!


#===============================================================================
""" Line emission plotting """
#-------------------------------------------------------------------------------

def line_SFR(**kwargs):
    '''Plots line luminosity against SFR (together with observations)

    Parameters
    ----------
    line: str
        Line to look at, default: 'CII'

    '''

    plt.close('all')

    set_mpl_params()

    # Redshift sample that we're at:
    z2                      =   z1.replace('z','')

    # handle default values and kwargs
    args                    =   dict(line='CII',colorbar='SFRsd',extract_from='regions',save=True,plot_DeLooze14=True,one_symbol_only=False)
    args                    =   aux.update_dictionary(args,kwargs)
    for key,val in args.items():
        exec('globals()["' + key + '"]' + '=val')

    # construct global results and galaxy
    GR                      =   glo.global_results()
    zreds                   =   getattr(GR,'zreds')
    SFRs                    =   getattr(GR,'SFR')
    SFRsds                  =   getattr(GR,'SFRsd')
    line_lums               =   getattr(GR,'L_'+line)

    # Start figure
    simple_plot(fignum=0,xlab=getlabel('SFR'),ylab='L$_{\mathrm{%s}}$ [L$_{\odot}$]' % aux.line_name(line))
    ax1                 =   plt.gca()
    xr                  =   axis_range(SFRs,log='y',dex=[0.5,1])
    yr                  =   axis_range(line_lums,log='y',dex=[0.5,2])
    ax1.set_xlim(xr)
    ax1.set_ylim(yr)
    ax1.set_xscale('log')
    ax1.set_yscale('log')

    # Add MW?
    if z1 == 'z0' and line == 'CII':
        SFR_MW              =   1.9 # Chomiuk and Povich 2011
        L_tot_MW            =   10.**41*1e-7/Lsun # Pineda+14
        simple_plot(add='y',\
            x1=SFR_MW,y1=L_tot_MW,col1='k',fill1='y',ma1='x',ms1=10,lw=2,lab1='MW [Pineda+14]')

    # Plot [CII]-SFR relation by de Looze et al 2014
    if plot_DeLooze14:
        logLcii_range       =   np.log10([1e-3,1e10])
        logSFR_range        =   np.log10([1e-3,1e4])
        # z=0 data (de Looze 2014)
        # Powerlaw fit to z=0 metalpoor dwarfs
        logSFR_dwarf        =  -5.73+0.80*logLcii_range
        # Powerlaw fit to z=0 starburst galaxies
        logSFR_SB           =  -7.06+1.00*logLcii_range
        simple_plot(add='y',\
            x2=10.**logSFR_dwarf,y2=[10.**(logLcii_range-0.37),10.**(logLcii_range+0.37)], hatchstyle2='',col2='lightgrey',alpha2=0.5,zorder2=0,\
            x3=10.**logSFR_SB,y3=[10.**(logLcii_range-0.27),10.**(logLcii_range+0.27)], hatchstyle3='',col3='lightgrey',alpha3=0.5,zorder3=0,\
            x4=10.**logSFR_dwarf,y4=10.**logLcii_range,col4='grey',ls4='--',lab4='$z=0$ metal-poor dwarfs [de Looze 14]',zorder4=0,\
            x5=10.**logSFR_SB,y5=10.**logLcii_range,col5='grey',ls5='--',dashes5=(1,4),lab5='$z=0$ starburst galaxies [de Looze 14]',zorder5=0)

    # Add power law fit if possible
    if len(zreds) > 1:
        slope,intercept,slope_dev,inter_dev = aux.lin_reg_boot(np.log10(SFRs),np.log10(line_lums))
        fit                 =   10.**(slope*np.log10(xr)+intercept)
        simple_plot(add='y',\
            x1=xr,y1=fit,col1=redshift_colors[int(z2)],ls1='--',lw1=2,lab1='Power law fit',zorder1=10)

    # Color-code galaxies 
    if colorbar == 'SFRsd':
        simple_plot(add='y',\
            x1=SFRs,y1=line_lums,scatter_color1=np.log10(SFRsds),ecol1='k',lw1=2,ma1='o',ms1=64,lab1=r'S$\mathrm{\'I}$GAME at z$\sim$'+z2+' (this work)',zorder1=200,\
            lab_colorbar=getlabel('lSFRsd'))
    if colorbar == 'none':
        simple_plot(add='y',\
            x1=SFRs,y1=line_lums,fill1='y',col1=redshift_colors[int(z2)],alpha1=0.7,lw1=2,ma1='o',ms1=8,lab1=r'S$\mathrm{\'I}$GAME at z$\sim$'+z2+' (this work)',zorder1=200)

    # Add legend
    ax1                 =   plt.gca()
    handles, labels     =   ax1.get_legend_handles_labels()
    fs_legend           =   mpl.rcParams['ytick.labelsize']*0.8
    ax1.legend(handles[::-1], labels[::-1],loc='upper left',fontsize=fs_legend)


    if save:
        plt.tight_layout()
        plt.show(block=False)
        plt.savefig('plots/line_emission/line_SFR/'+line+'_SFR_'+z1+ext_DENSE+ext_DIFFUSE+'.pdf',format='pdf')

def OI_OIII_SFR(plotmodels=True,twopanel=True):
    '''
    Purpose
    ---------
    Inspect [OI]63micron AND [OIII]88micron emission (called by line_SFR() in analysis.py)
    '''

    lines                   =   ['[OI]','[OIII]','[CII]']
    wavelengths             =   np.array([63,88,122,158])

    models                      =  glo.global_results()

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
    simple_plot(add='y',xr=xr,yr=yr, ylog='y',xlog='y',fontsize=13,\
        x1=xr, y1=10.**L_OI_L14_DG, ls1='--', lw1=2,  col1='grey', lab1=u'Local metal-poor dwarf galaxies, [De Looze et al. 2014] $_{}$',\
        x2=xr, y2=10.**L_OI_L14_SB, ls2=':', dashes2=(1,4), lw2=2,  col2='grey', lab2=u'Local starburst galaxies, [De Looze et al. 2014] $_{}$',\
        # x3=xr, y3=powlaw, ls3='-', lw3=2,  col3='black', \
        x4=SFR, y4=L_OI, ma4='o', scatter_color4='lightseagreen', mew4=2, ms4=64,\
        xlab='',xticks='n',ylab=getlabel('L_OI'),\
        legloc=[0.04,0.8],legend=True)
    # pdb.set_trace()
    ax1                     =   fig.add_subplot(2,1,2)
    yr                      =   axis_range(L_OIII,log='y',dex=1)
    yr                      =   [10**5.8,yr[1]]
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

def map_line(**kwargs):
    """Makes moment0 map of line using datacubes.
    Parameters
    ----------
    gal_index: int
        Galaxy index, default: 0
    line : str
        Line to use for map, default: 'CII'
    R_max : float
        Maximum radius for moment0 map in kpc, default: 15
    ISM_dc_phase : str
        ISM phase(s) to map, default: 'tot' (sum of all)
    units : str
        Units; 'Wm2' for W/m^2, 'Jykms' for Jy*km/s, default: 'Jykms'
    convolve : bool
        Whether moment0 map should be convolved by corresponding Herschel beam, default: True
    min_fraction : float
        Fraction of maximum in image used as lower limit for colorbar, default: 1/1e6
    """

    for key,val in kwargs.items():
        exec('globals()["' + key + '"]' + '=val')

    plt.close('all')

    # Initialize galaxy object and plot moment0 map
    gal_ob              =   gal.galaxy(gal_index=gal_index)
    mom0                =   gal_ob.datacube.get_moment0_map(**kwargs)

    min_value               =   np.max(mom0)*min_fraction
    mom0[mom0 < min_value]  =   min_value
    if Iunits == 'Wm2_sr': lab = aux.line_name(line) + r' log(F$_{\nu}$ [$\,$W/m$^2\,$sr$^{-1}$])'
    if Iunits == 'Jykms': lab = aux.line_name(line) + r' log(F$_{\nu}$ [$\,$Jy km s$^{-1}$ per pixel])'

    # Size of pixels in steradians
    pix_arcsec      =   np.tan(x_res_pc/1000./gal_ob.ang_dist_kpc)*60*60*360./(2*np.pi)
    pix_sr          =   aux.arcsec2_to_sr(pix_arcsec**2)
    if Iunits == 'Wm2_sr':   mom0        =   aux.Jykm_s_to_W_m2(line,self.gal_ob.zred,mom0)/pix_sr # W/m^2/sr

    simple_plot(plot_margin=0.15,xr=[-R_max,R_max],yr=[-R_max,R_max],\
        x1=aux.get_x_axis_kpc(),y1=aux.get_x_axis_kpc(),col1=np.log10(mom0),\
        colorbar1=True,lab_colorbar1=lab,\
        aspect='equal',\
        contour_type1='plain',nlev1=100,xlab='x [kpc]',ylab='y [kpc]',title='G%s' % (gal_ob.gal_index+1),\
        textfs=9,textcol='white',**kwargs)

    if convolve:
        FWHM_arcsec         =   aux.get_Herschel_FWHM(line)
        FWHM_kpc            =   np.arctan(FWHM_arcsec/60./60./360.*2.*np.pi)*gal_ob.ang_dist_kpc
        ax1                 =   plt.gca()
        patches             =   [Wedge((R_max*0.8,-0.8*R_max), FWHM_kpc/2., 0, 360, width=0.05)]
        patches_col         =   PatchCollection(patches, alpha=1, edgecolors='g',linestyle='-')
        ax1.add_collection(patches_col)

    if not os.path.exists('plots/maps/'):
        os.mkdir('plots/maps/')
    plt.savefig(('plots/maps/%s_%s_%s_%s.png' % (z1, line, gal_ob.name, ISM_dc_phase)),format='png',dpi=300)
    plt.show(block=False)

#===============================================================================
""" Plot cloudy models """
#-------------------------------------------------------------------------------

def grid_parameters_checkParam(histo_color='teal', FUV=0.002, ISM_phase='GMC',figsize=(10,7)):

    '''
    Make histograms of galaxies to check what parameters to be used for dif_cloud_grid.py and GMC_cloud_grid.py to make cloudy grids.

    Parameters
    ----------
    ISM_phase: str
        if 'GMC': make 4-panel figure with histograms of [Mgmc, G0, Z, P_ext] in GMCs
        if 'dif': make 4-panel figure with histograms of [nH, R, Z, Tk] in diffuse clouds

    histo_color: str
        color selection for histograms
        options:
            - '<a specific color>': all histograms will have this color
            - 'colsel': each galaxy will have a different color, using colsel from param_module.read_params
    '''

    plt.close('all')        # close all windows

    # handle default values and kwargs
    # args                    =   dict(region_size=16.8, max_regions_per_gal=30,gal_index=0,ISM_phase='GMC',line1='CII',line2='NII_205',extract_from='regions')
    # args                    =   aux.update_dictionary(args,kwargs)
    # for key in args: exec(key + '=args[key]')

    # construct global results and galaxy
    GR                      =   glo.global_results()


    if ISM_phase == 'GMC':

        print('\n Now looking at GMC grid parameters')
        plot_these          =   ['Mgmc','FUV','Z','P_ext']
        bins                =   80
        fontsize            =   12
        mpl.rcParams['xtick.labelsize'] = fontsize
        mpl.rcParams['ytick.labelsize'] = fontsize

        plt.ion()
        fig                 =   plt.figure(1,figsize = figsize)
        panel               =   1
        if z1 == 'z6':
                ranges = [[3.5,6.],[0,7],[-4,0.5],[3,13]]
        if z1 == 'z2':
                ranges = [[3.5,6.],[-5,7],[-2.2,1],[0,12.]]
        if z1 == 'z0':
                ranges = [[3.8,6],[-8.,5],[-1.5,1],[-2,12.]]
        for name in plot_these:
            x_max       =   np.array([])
            x_min       =   np.array([])
            ax1         =   fig.add_subplot(2,2,panel)
            # Make histograms
            histos1     =   np.zeros([GR.N_gal,bins+2])
            histos2     =   np.zeros([GR.N_gal,bins+3])
            n_histos    =   0
            for gal_index in range(0,GR.N_gal):
                gal_ob          =   gal.galaxy(gal_index=gal_index)
                dat0            =   gal_ob.particle_data.get_raw_data(data='ISM')['GMC']
                w               =   dat0['m'].values
                dat0['Mgmc']    =   dat0['m'].values
                dat             =   dat0[name].values
                w               =   w[dat > 0]
                dat             =   dat[dat > 0]
                dat             =   np.log10(dat)
                i_nan           =   np.isnan(dat)
                w               =   w[i_nan == False]
                dat             =   dat[i_nan == False]
                if name == 'Mgmc': w = np.zeros(len(w))+1.
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
                histos1[n_histos,:]    =   hist1
                histos2[n_histos,:]    =   hist2
                n_histos        +=   1
            histos1         =   histos1[0:n_histos,:]
            histos2         =   histos2[0:n_histos,:]
            # Plot as background the 2 sigma distribution around the mean in each bin
            minhistos1,maxhistos1,meanhistos1       =   np.zeros(bins+2), np.zeros(bins+2), np.zeros(bins+2)
            for i in range(0,bins+2):
                meanhistos1[i]     =   np.mean(histos1[:,i])
                minhistos1[i]      =   meanhistos1[i]-2.*np.std(histos1[:,i])
                maxhistos1[i]      =   meanhistos1[i]+2.*np.std(histos1[:,i])
            # ax1.fill_between(histos2[0,0:len(hist1)], minhistos1, maxhistos1, facecolor='lightgreen', alpha=0.5, lw=0)
            # Now plot actual histograms
            for i in range(0,n_histos):
                if histo_color == 'teal': color = 'teal'
                if histo_color == 'colsel': color = colsel[i]
                hist2           =   histos2[i,:]
                hist1           =   histos1[i,:]
                ax1.plot(hist2[0:len(hist1)]+wid/2,hist1,ls='-',ds='steps',color=color,label='G'+str(int(i+1)),alpha=0.7,lw=1)
            ax1.set_xlabel('log('+getlabel(name)+')')
            if name == 'Mgmc':
                ax1.set_ylabel('Number fraction [%]')
            else:
                ax1.set_ylabel('Mass fraction [%]')

            # Now plot mean of histograms
            ax1.plot(hist2[0:len(hist1)]+wid/2,meanhistos1,ls='-',ds='steps',color='blue',lw=1.5)

            # Fix axes
            ax1.set_yscale('log')
            ymin        =   10**(-1.)
            ax1.set_ylim([ymin,10.**(np.log10(max(maxhistos1))+(np.log10(max(maxhistos1))-np.log10(ymin))/4.)])
            ax1.set_ylim([0.1,100])
            dx                  =   (max(x_max)-min(x_min))/10.
            ax1.set_xlim(ranges[panel-1])

            if panel == 1:
                if histo_color == 'colsel': ax1.legend(loc='upper left')

            panel              +=  1

        plt.show(block=False)
        savepath = 'plots/GMCs/grid_parameters_checkParam/'
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        plt.savefig('plots/GMCs/grid_parameters_checkParam/GMC_histos'+ext_DENSE+'_'+z1+'.pdf',format='pdf')

    if ISM_phase == 'dif':

        print('\n Now looking at diffuse gas grid parameters')
        plot_these          =   ['nH','R','Z','Tk']
        bins                =   80
        ext_DIF1            =   '_'+str(FUV)+'UV'+ext_DIFFUSE
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
            histos1     =   np.zeros([GR.N_gal,bins+2])
            histos2     =   np.zeros([GR.N_gal,bins+3])
            n_histos    =   0
            for gal_index in range(0,GR.N_gal):
                gal_ob          =   gal.galaxy(gal_index=gal_index)
                dat0            =   gal_ob.particle_data.get_raw_data(data='ISM')['dif']
                w               =   dat0['m'].values
                dat             =   dat0[name].values
                w               =   w[dat > 0]
                dat             =   dat[dat > 0]
                dat             =   np.log10(dat)
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
                histos1[n_histos,:]    =   hist1
                histos2[n_histos,:]    =   hist2
                n_histos        +=   1
            histos1         =   histos1[0:n_histos,:]
            histos2         =   histos2[0:n_histos,:]
            # Plot as background the 2 sigma distribution around the mean in each bin
            minhistos1,maxhistos1,meanhistos1       =   np.zeros(bins+2), np.zeros(bins+2), np.zeros(bins+2)
            for i in range(0,bins+2):
                meanhistos1[i]     =    np.mean(histos1[:,i])
                minhistos1[i]      =   meanhistos1[i]-2.*np.std(histos1[:,i])
                maxhistos1[i]      =   meanhistos1[i]+2.*np.std(histos1[:,i])

            # Now plot actual histograms
            for i in range(0,n_histos):
                if histo_color == 'teal': color = 'teal'
                if histo_color == 'colsel': color = colsel[i]
                hist2           =   histos2[i,:]
                hist1           =   histos1[i,:]
                ax1.plot(hist2[0:len(hist1)]+wid/2,hist1,ls='-',ds='steps',color=color,label='G'+str(int(i+1)),alpha=0.7,lw=1)
            ax1.set_xlabel('log('+getlabel(name)+')')
            ax1.set_ylabel('Mass fraction [%]')

            # Now plot mean of histograms
            ax1.plot(hist2[0:len(hist1)]+wid/2,meanhistos1,ls='-',ds='steps',color='blue',lw=1.5)

            ax1.set_yscale('log')
            ymin        =   10**(-1.2)
            ax1.set_ylim([ymin,10.**(np.log10(max(maxhistos1))+(np.log10(max(maxhistos1))-np.log10(ymin))/4.)])
            ax1.set_xlim(ranges[panel-1])

            if panel == 1:
                if histo_color == 'colsel': ax1.legend(loc='upper left')

            panel              +=  1
        # plt.suptitle('Grid points (dashed lines) on histograms (solid lines) in diffuse gas')
        plt.show(block=False)
        savepath = 'plots/GMCs/grid_parameters_checkParam/'
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        plt.savefig('plots/GMCs/grid_parameters_checkParam/dif_histos'+ext_DIFFUSE+'_'+z1+'.pdf', format='pdf')

def grid_parameters(histo_color='teal',FUV=0.1,ISM_phase='GMC',figsize=(10,7)):
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
    ISM_phase: ISM phase in cloudy models, can be either 'GMC' or 'dif' - str
    default: 'GMC'
    histo_color: color selection for histograms - str
    options:
        - '<a specific color>': all histograms will have this color
        - 'colsel': each galaxy will have a different color, using colsel from param_module.read_params
    '''

    plt.close('all')        # close all windows

    # handle default values and kwargs
    # args                    =   dict(region_size=16.8, max_regions_per_gal=30,gal_index=0,ISM_phase='GMC',line1='CII',line2='NII_205',extract_from='regions')
    # args                    =   aux.update_dictionary(args,kwargs)
    # for key in args: exec(key + '=args[key]')

    # construct global results and galaxy
    GR                      =   glo.global_results()

    if ISM_phase == 'GMC':

        print('\n Now looking at GMC grid parameters')
        plot_these          =   ['Mgmc','FUV','Z','P_ext']
        bins                =   80
        grid_params         =   pd.read_pickle(d_cloud_models+ 'GMC/grids/'+ 'GMCgrid'+ext_DENSE+'_'+z1+'.param')
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
        if z1 == 'z2': ranges              =   [[3.5,6.],[-5,7],[-2.2,1],[0,12.]]
        if z1 == 'z0': ranges              =   [[3.8,6],[-8.,5],[-1.5,1],[-2,12.]]
        for name in plot_these:
            x_max       =   np.array([])
            x_min       =   np.array([])
            ax1         =   fig.add_subplot(2,2,panel)
            # Make histograms
            histos1     =   np.zeros([GR.N_gal,bins+2])
            histos2     =   np.zeros([GR.N_gal,bins+3])
            n_histos    =   0
            for gal_index in range(0,GR.N_gal):
                gal_ob          =   gal.galaxy(gal_index=gal_index)
                dat0            =   aux.load_temp_file(gal_ob=gal_ob,ISM_phase=ISM_phase) #self.particle_data.get_raw_data(data='sim')['gas']
                # dat0            =   gal_ob.particle_data.get_raw_data(data='ISM')['GMC']
                w               =   dat0['m'].values
                dat0['Mgmc']    =   dat0['m'].values
                dat             =   dat0[name].values
                w               =   w[dat > 0]
                dat             =   dat[dat > 0]
                dg              =   (grid_params[name][1]-grid_params[name][0])
                dat             =   np.log10(dat)
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
                histos1[n_histos,:]    =   hist1
                histos2[n_histos,:]    =   hist2
                n_histos        +=   1
            histos1         =   histos1[0:n_histos,:]
            histos2         =   histos2[0:n_histos,:]
            # Plot as background the 2 sigma distribution around the mean in each bin
            minhistos1,maxhistos1,meanhistos1       =   np.zeros(bins+2), np.zeros(bins+2), np.zeros(bins+2)
            for i in range(0,bins+2):
                meanhistos1[i]     =   np.mean(histos1[:,i])
                minhistos1[i]      =   meanhistos1[i]-2.*np.std(histos1[:,i])
                maxhistos1[i]      =   meanhistos1[i]+2.*np.std(histos1[:,i])
            # ax1.fill_between(histos2[0,0:len(hist1)], minhistos1, maxhistos1, facecolor='lightgreen', alpha=0.5, lw=0)
            # Now plot actual histograms
            for i in range(0,n_histos):
                if histo_color == 'teal': color = 'teal'
                if histo_color == 'colsel': color = colsel[i]
                hist2           =   histos2[i,:]
                hist1           =   histos1[i,:]
                ax1.plot(hist2[0:len(hist1)]+wid/2,hist1,ls='-',ds='steps',color=color,label='G'+str(int(i+1)),alpha=0.7,lw=1)
            ax1.set_xlabel('log('+getlabel(name)+')')
            if name == 'Mgmc':
                ax1.set_ylabel('Number fraction [%]')
            else:
                ax1.set_ylabel('Mass fraction [%]')

            # Indicate grid points
            for grid_point in grid_params[name]:
                ax1.plot([grid_point,grid_point],[1e-3,1e3],'k--')

            # Now plot mean of histograms
            ax1.plot(hist2[0:len(hist1)]+wid/2,meanhistos1,ls='-',ds='steps',color='blue',lw=1.5)

            # Fix axes
            ax1.set_yscale('log')
            ymin        =   10**(-1.)
            ax1.set_ylim([ymin,10.**(np.log10(max(maxhistos1))+(np.log10(max(maxhistos1))-np.log10(ymin))/4.)])
            ax1.set_ylim([0.1,100])
            dx                  =   (max(x_max)-min(x_min))/10.
            ax1.set_xlim(ranges[panel-1])

            if panel == 1:
                if histo_color == 'colsel': ax1.legend(loc='upper left')

            panel              +=  1

        plt.show(block=False)
        savepath = 'plots/GMCs/grid_parameters/'
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        plt.savefig('plots/GMCs/grid_parameters/GMCgrid_points_on_histos'+ext_DENSE+'_'+z1+'.pdf',format='pdf')

    if ISM_phase == 'dif':

        print('\n Now looking at diffuse gas grid parameters')
        plot_these          =   ['nH','R','Z','Tk']
        bins                =   80
        ext_DIF1            =   '_'+str(FUV)+'UV'+ext_DIFFUSE
        grid_params         =   pd.read_pickle(d_cloudy_models+ 'dif/grids/' + 'difgrid'+ext_DIF1+'_'+z1+'.param')
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
            histos1     =   np.zeros([GR.N_gal,bins+2])
            histos2     =   np.zeros([GR.N_gal,bins+3])
            n_histos    =   0
            for gal_index in range(0,GR.N_gal):
                gal_ob          =   gal.galaxy(gal_index=gal_index)
                dat0            =   aux.load_temp_file(gal_ob=gal_ob,ISM_phase=ISM_phase) #self.particle_data.get_raw_data(data='sim')['gas']
                w               =   dat0['m'].values
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
                histos1[n_histos,:]    =   hist1
                histos2[n_histos,:]    =   hist2
                n_histos        +=   1
            histos1         =   histos1[0:n_histos,:]
            histos2         =   histos2[0:n_histos,:]
            # Plot as background the 2 sigma distribution around the mean in each bin
            minhistos1,maxhistos1,meanhistos1       =   np.zeros(bins+2), np.zeros(bins+2), np.zeros(bins+2)
            for i in range(0,bins+2):
                meanhistos1[i]     =    np.mean(histos1[:,i])
                minhistos1[i]      =   meanhistos1[i]-2.*np.std(histos1[:,i])
                maxhistos1[i]      =   meanhistos1[i]+2.*np.std(histos1[:,i])

            # Now plot actual histograms
            for i in range(0,n_histos):
                if histo_color == 'teal': color = 'teal'
                if histo_color == 'colsel': color = colsel[i]
                hist2           =   histos2[i,:]
                hist1           =   histos1[i,:]
                ax1.plot(hist2[0:len(hist1)]+wid/2,hist1,ls='-',ds='steps',color=color,label='G'+str(int(i+1)),alpha=0.7,lw=1)
            ax1.set_xlabel('log('+getlabel(name)+')')
            ax1.set_ylabel('Mass fraction [%]')
            for grid_point in grid_params[name]:
                ax1.plot([grid_point,grid_point],[1e-3,1e3],'k--')

            # Now plot mean of histograms
            ax1.plot(hist2[0:len(hist1)]+wid/2,meanhistos1,ls='-',ds='steps',color='blue',lw=1.5)

            ax1.set_yscale('log')
            ymin        =   10**(-1.2)
            ax1.set_ylim([ymin,10.**(np.log10(max(maxhistos1))+(np.log10(max(maxhistos1))-np.log10(ymin))/4.)])
            ax1.set_xlim(ranges[panel-1])

            if panel == 1:
                if histo_color == 'colsel': ax1.legend(loc='upper left')

            panel              +=  1
        # plt.suptitle('Grid points (dashed lines) on histograms (solid lines) in diffuse gas')
        plt.show(block=False)
        savepath = 'plots/dif/grid_parameters/'
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        plt.savefig('plots/dif/grid_parameters/difgrid_points_on_histos'+ext_DENSE+'_'+z1+'.pdf',format='pdf')

        # fig.savefig('plots/dif/grids/grid_points_on_histos_paper.png', dpi=fig.dpi)

#===============================================================================
""" Auxiliary plotting functions """
#-------------------------------------------------------------------------------

def set_mpl_params():
    mpl.rcParams['xtick.labelsize']=13
    mpl.rcParams['ytick.labelsize']=13

    mpl.rcParams['legend.loc']='best'
    mpl.rcParams['legend.numpoints']=1
    mpl.rcParams['legend.fontsize']=13

    mpl.rcParams['lines.linewidth']=1.5
    mpl.rcParams['lines.markersize']=5

    mpl.rcParams['font.size']=13
    # mpl.rcParams['font.weight']='bold'

    mpl.rcParams['figure.figsize']=(15,15)
    mpl.rcParams['figure.titlesize']=15
    # mpl.rcParams['figure.titleweight']='bold'
    mpl.rcParams['figure.subplot.wspace']=0.5
    mpl.rcParams['figure.subplot.hspace']=0.4

    mpl.rcParams['axes.titlesize']=15
    mpl.rcParams['axes.titlepad']=10
    # mpl.rcParams['axes.titleweight']='bold'
    mpl.rcParams['axes.linewidth']=1.5
    mpl.rcParams['axes.labelsize']=13
    mpl.rcParams['axes.labelpad']=10
    # mpl.rcParams['axes.labelweight']='bold'

    mpl.rcParams['image.cmap']='seismic'

def delete_old_plots():

    plt.close('all')        # close all windows

def axis_range(x,log=False,**kwargs):
    '''Calculates a reasonable axis range

    Parameters
    ----------

    x : np.array()
        Must be given, array of values

    log : bool
        If True, takes the log of the values in x, default: False

    dex : float
        Dex above and below maximum values in x, effective when log == True, default: max range in x divided by 5

    fraction : float
        Fraction of maximum range in x to go above and below on the axis, effective when log == False, default: 1/5.

    '''

    if log:

        x       =   np.log10(x)
        xr      =   [np.min(x),np.max(x)]
        dx      =   (np.max(x)-np.min(x))/5.
        if 'dex' in kwargs:
            dx      =   kwargs['dex']
        if type(dx) == int or type(dx) == float or type(dx) == np.float64: dx = [dx,dx]
        xr      =   np.array([xr[0]-dx[0],xr[1]+dx[1]])
        xr      =   10.**xr

        if len(x) == 1:
            xr      =   10.**np.array([x-0.5,x+0.5])

    else:

        xr      =   [np.min(x),np.max(x)]
        frac    =   1./5
        if 'frac' in kwargs: frac = kwargs['frac']
        if type(frac) == int or type(frac) == float or type(frac) == np.float64: frac = [frac,frac]
        dx1     =   (np.max(x)-np.min(x))*frac[0]
        dx2     =   (np.max(x)-np.min(x))*frac[1]
        xr      =   np.array([xr[0]-dx1,xr[1]+dx2])

    return xr

def add_CII_observations_to_plot1(slope_models,intercept_models,mark_reasons=False,mark_uplim=False,mark_det=False,z1='z6',MW=True,ms_scaling=1,alpha=1,line='[CII]'):
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

    CII_obs             =   pd.read_pickle('sigame/temp/observations/observations_CII_'+z1)

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

def add_line_ratio_obs(ratio='CII_NII',zred_sample='lowz'):
    '''
    Purpose
    ---------
    Plots observations of the relevant line ratio against redshift

    Arguments
    ---------
    ratio - str
    Options:
        - 'CII_NII': plots the [CII]/[NII]205 line ratio
        - 'OIII_NII': plots the [OIII]88/[NII]122 line ratio

    zred_sample - str
    Options:
        - 'highz': adds observations at high redshift
        - 'lowz': adds observations at low redshift
    '''

    markers             =   ['o','x','D','v','s','^','*']
    colors              =   [u'fuchsia',u'darkcyan',u'purple',u'indigo',u'blueviolet',\
                            u'darkred',u'lightgrey']
    if (ratio == 'CII_NII'):
        name                =   '[CII]/[NII]205'
        num                 =   'CII'
        denom               =   'NII'
        data                =   np.load('sigame/temp/observations/observations_CII_NII205_'+zred_sample+'.npy', allow_pickle=True).item()
    if (ratio == 'OIII_NII'):
        name                =   '[OIII]88/[NII]122'
        num                 =   'OIII'
        denom               =   'NII'
        data                =   np.load('sigame/temp/observations/observations_OIII88_NII122_'+zred_sample+'.npy', allow_pickle=True).item()

    ax                  =   plt.gca()
    authors             =   data.keys()

    # print('Now plotting the %s ratio against redshift' % (name) )
    all_x_obs           =   []
    for i,author in enumerate(authors):
        color               =   colors[i]
        marker              =   markers[i]
        ratio_val           =   data[author][ratio]
        zred                =   data[author]['zred']
        # Calculate error propagation
        uperror             =   data[author][ratio+'+']
        loerror             =   data[author][ratio+'-']
        error               =   np.vstack((loerror,uperror))
        uplim               =   data[author][ratio+'_uplim']
        lolim               =   data[author][ratio+'_lolim']
        ax.errorbar(zred, ratio_val, yerr=error, lolims=lolim, uplims=uplim, c=color, marker=marker, lw=0.5, ls='None', capsize=4, capthick=0.5)
        all_x_obs           =   np.append(all_x_obs,zred)

    # Add labels for legend:
    for i,author in enumerate(authors):
        # print(author)
        color               =   colors[i]
        marker              =   markers[i]
        ax.plot(1e9,1e9,\
            c=color,marker=marker,lw=0.5,ls='None',\
            label=author)

    ax.xaxis.set_ticks_position('both'); ax.yaxis.set_ticks_position('both')
    ax.set_xlabel('z',fontsize=11); ax.set_ylabel(name,fontsize=11)

    return(all_x_obs)

def getlabel(foo):
    '''Gets axis labels for plots
    '''


    if foo == 'z': return 'z'
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
    if foo == 'f_neu': return 'f$_{\mathrm{neu}}$'
    if foo == 'Tk': return '$T_{\mathrm{k}}$ [K]'
    if foo == 'Z': return '$Z$ [Z$_{\odot}$]'
    if foo == 'lZ': return 'log($Z$ [Z$_{\odot}$])'
    if foo == 'Zmw': return r"$\langle Z'\rangle_{\mathrm{mass}}$"
    if foo == 'Zsfr': return r"$\langle Z'\rangle_{\mathrm{SFR}}$"
    if foo == 'Zstar': return r"$\langle Z'\rangle_{\mathrm{stars}}$"
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
    if foo == 'L_NII_122': return 'L$_{\mathrm{[NII122]}}$ [L$_{\odot}$]'
    if foo == 'lL_NII_122': return 'log(L$_{\mathrm{[NII122]}}$ [L$_{\odot}$])'
    if foo == 'L_NII_205': return 'L$_{\mathrm{[NII205]}}$ [L$_{\odot}$]'
    if foo == 'lL_NII_205': return 'log(L$_{\mathrm{[NII205]}}$ [L$_{\odot}$])'
    if foo == 'S_CII': return 'S$_{\mathrm{[CII]}}$ [mJy]'
    if foo == 'x_e': return 'Electron fraction [H$^{-1}$]'
    if foo == 'f_CII': return '(mass of carbon in CII state)/(mass of carbon in CIII state) [%]'
    if foo == 'f_ion': return 'Ionized gas mass fraction [%]'
    if foo == 'f_neu': return 'Neutral gas mass fraction [%]'
    if foo == 'f_gas': return 'Gas mass fraction M$_{\mathrm{gas}}$/(M$_{\mathrm{gas}}$+M$_{\mathrm{*}}$) [%]'
    if foo == 'f_CII_neu': return 'f_${CII,neutral}$ [%]'

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
