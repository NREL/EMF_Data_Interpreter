#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 13:09:25 2023

@author: Liz Wachs
@author: Timothy Simon 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

Transport_Fuel_type_dict = {'Final Energy|Transportation|Biomass Solids': 'biomass solids',
                            'Final Energy|Transportation|Biomass Liquids': 'biomass liquids',
                            'Final Energy|Transportation|Biogas': 'biogas',
                            'Final Energy|Transportation|Electricity': 'electricity',
                            'Final Energy|Transportation|Gas': 'gas',
                            'Final Energy|Transportation|Hydrogen': 'hydrogen',
                            'Final Energy|Transportation|Oil': 'oil',
                            'Final Energy|Transportation|Synthetic Gas': 'synthetic gas',
                            'Final Energy|Transportation|Synthetic Liquids': 'synthetic liquids'
                            }

Transport_Fuel_list = list(Transport_Fuel_type_dict.keys())




def elim_nan(df):
    """
    Parameters
    ----------
    df : dataframe

    Returns
    -------
    df : dataframe with nan values replaced with zero
        DESCRIPTION.
    """
    df = df.fillna(np.nan)
    df = df.fillna(0)
    return df


def emf(dff, dfb, scen, var1, var2, var3, nm):
    """
    pivots dataframe and limits by the scenario specified.

    Parameters
    ----------
    dff : the dataframe that is being assessed.  Basically this is the the sector 
    that you want to run through the function
    dfb : the "master dataframe" which is going to refer to the primary sector 
    you want to focus on. What this does is it only includes models in the output 
    that are included in this sector for whatever scenario you choose.  For example, 
    if dfb is defined as the transportation dataframe (dfb_t) and you are modeling 
    the electricity sector (dfb_e), only models that include data for whatever 
    scenario you are review within the transportation sector will be included in the output.
    scen : the scenario that you are running.
    var1 - var3 - the variables that you are included )(refer to the power point slide on variables for an example.

    nm : name for final output.

    Returns
    -------
    dffb : pivoted dataframe for scenario.

    """
    dffb = pd.DataFrame()
    # c1-c3 are categories of energy, so the coding here is for biomass solids, liquids and biogas
    c1_add = [] #could just use append but this is fine and easier
    c2_add = []
    c3_add = []
    yr = []
    mods = []
    set1 = []
    set2 = []
    set3 = []

    for mod in list(dfb['model'].unique()):
        year = list(dfb['year'].loc[dfb['model']==mod].loc[dfb['scenario']==scen].unique())
        set1 = list(dff['value'].loc[dff['model']==mod].loc[dff['variable']==var1].loc[dff['scenario']==scen].loc[dff['year'].isin(year)])
        set2 = list(dff['value'].loc[dff['model']==mod].loc[dff['variable']==var2].loc[dff['scenario']==scen].loc[dff['year'].isin(year)])
        set3 = list(dff['value'].loc[dff['model']==mod].loc[dff['variable']==var3].loc[dff['scenario']==scen].loc[dff['year'].isin(year)])

        if len(set1) == 0:
            set1 = [0]*len(year)
        if len(set2) == 0:
            set2 = [0]*len(year)
        if len(set3) == 0:
            set3 = [0]*len(year)

        c1_add = c1_add + set1
        c2_add = c2_add + set2
        c3_add = c3_add + set3
        yr = yr + year
        mods = mods + [mod]*len(year)

        set1=[]
        set2=[]
        set3=[]
        year = []

    dffb['model'] = mods
    dffb['year'] = yr
    dffb['Biomass Solids'] = c1_add
    dffb['Biomass Liquids'] = c2_add
    dffb['Biogas'] = c3_add
    dffb['Bioenergy '+nm] = dffb['Biomass Solids'] + dffb['Biomass Liquids'] + dffb['Biogas']

    return dffb

def figs(data, nmm, title):
    """
    
    Parameters
    ----------
    data : the new dataframe being assessed which can be the output from the emf function.
    nmm : the column name for what you are plotting in the dataframe, this was determined from the first function
    title : title for figure

    Returns
    -------
    fig : line chart from 2020-2050 in 5 year increments in EJ/yr with differentiation of different models; 
    can also be a line chart of percentage of EJ from bioenergy - in that case nmm = 'ratio'

    """
    count = 0
    colors = sns.color_palette('hls', len(data['model'].unique()))
    colors = plt.cm.tab20(np.linspace(0, 1, len(data['model'].unique())))
    
    for mod in list(data['model'].unique()):
        if data.loc[data['model'] == mod, nmm].sum() > 0:
            plt.plot(data['year'].loc[data['model'] == mod], data[nmm].loc[data['model']==mod], label = mod, c = colors[count])
            count = count+1
    
    if nmm == 'ratio':
        plt.title(title + ' Bioenergy Ratio')
        plt.ylabel('% Transportation Bioenergy')
    else:
        plt.title(title + ' (EJ/yr)')
        plt.ylabel('EJ/yr')
    plt.xlabel('Year')
    
    plt.xlim(2020, 2050)
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc = "center left", borderaxespad=0)
    fig = plt.gcf()
    return fig


def heatmap_for_var(df, var, output_dir = '', poss_years = [2020, 2025, 2030, 2035, 2040, 2045, 2050], file_opt = '.svg'):
    """
    Create a heatmap with every model and scenario's energy use for the variable category. Defaults to 5 year increments from 
    2020 to 2050. Defaults to save as a .svg vectorized file. If a larger graphic size is desired, that should be adjusted before 
    this function is run.

    Parameters
    ----------
    df : The dataframe used has been pivoted from the default format. It is a multiindex with model, scenario and year as the
    index levels, and each variable is a column, with the values filling the columns.
    var : This comes from the variable column in the EMF csv.
    output_dir : string with the filepath this should be saved to. The default is ''.
    poss_years : list of years to be included in the heatmap. 
               DESCRIPTION. The default is [2020, 2025, 2030, 2035, 2040, 2045, 2050].
    file_opt : string, the type of file used to store the heatmap output 
        optional  DESCRIPTION. The default is '.svg'.

    Returns
    -------
    None. Heatmap is saved to file. 

    """
    df_to_plot = pd.DataFrame(df[var].copy())
    df_to_plot = df_to_plot.dropna(axis = 0)
    df_to_plot = df_to_plot.reset_index()
    df_to_plot = df_to_plot.pivot(index = ['model', 'scenario'], columns = 'year', values = var)
    df_to_plot = df_to_plot[df_to_plot.columns.intersection(poss_years)]
    #df_to_plot = df_to_plot.dropna(axis = 1)
    svm = sns.heatmap(df_to_plot, cmap = "YlGnBu", cbar_kws={'label': 'EJ/yr'})
    figure = svm.get_figure()
    figure.tight_layout()
    var = var.replace('|', '_')
    figure.savefig(output_dir + 'svm_' + var + file_opt)
    figure.clf()
    return

def heatmap_for_scen(df, scen, output_dir = '', var = 'Final Energy|Transportation|All Biomass', 
                     poss_years = [2020, 2025, 2030, 2035, 2040, 2045, 2050], file_opt = '.svg'):
    """
Create heatmaps for a specified scenario, with each model that includes the scenario along the y axis and the years on the x-axis.

    Parameters
    ----------
    df : The dataframe used has been pivoted from the default format. It is a multiindex with model, scenario and year as the
    index levels, and each variable is a column, with the values filling the columns.
    scen : scenario (string) specified by the user.
    output_dir : string with the filepath this should be saved to. The default is ''.
    var : TYPE, This comes from the variable column in the EMF csv.
        DESCRIPTION. The default is 'Final Energy|Transportation|All Biomass'.
    poss_years : list of years to include on heatmap.
        DESCRIPTION. The default is [2020, 2025, 2030, 2035, 2040, 2045, 2050].
    file_opt : string, the type of file used to store the heatmap output 
        optional  DESCRIPTION. The default is '.svg'.    

    Returns
    -------
    None. Heatmap is saved to file. 

    """
    df_to_plot = pd.DataFrame(df[var].copy())
    df_to_plot = df_to_plot.dropna(axis = 0)
    df_to_plot = df_to_plot.reset_index()
    df_to_plot = df_to_plot.pivot(index = ['model', 'scenario'], columns = 'year', values = var)
    df_to_plot = df_to_plot.xs(scen, level = 1)
    df_to_plot = df_to_plot[df_to_plot.columns.intersection(poss_years)]
    df_to_plot = df_to_plot.loc[(df_to_plot.sum(axis=1) != 0)]
    if len(df_to_plot.index) > 1:
        svm = sns.heatmap(df_to_plot, cmap = "YlGnBu", cbar_kws={'label': 'EJ/yr'}, yticklabels = 1)
        figure = svm.get_figure()
        figure.tight_layout()
        var = var.replace('|', '_')
        scen = scen.replace('.', '')
        figure.savefig(output_dir + 'svm_' + var + scen + file_opt)
        figure.clf()
    return



def find_elements(df, variable = 'Final Energy|Transportation|All Biomass', element_type = 'scenario'):
    """
    Create list of elements in a given category. The default gives a list of every scenario (element_type) 
    included in the dataframe being studied for the variable desired (All Biomass in Transport). 
    
    Parameters
    ----------
    df : dataframe 
    variable : For EMF data, it is usually one of the values in the 'variable' column. 
        optional DESCRIPTION. The default is 'Final Energy|Transportation|All Biomass'.
    element_type : string entry corresponding to a column in original dataframe. The default is 'scenario'.

    Returns
    -------
    elements : list of elements, for example a list of all scenarios that are included in the Biomass for transportation.

    """

    df2 = pd.DataFrame(df[variable].copy())
    df2 = df2.dropna(axis = 0)
    df2 = df2.reset_index()
    elements = list(df2[element_type].unique())
    return elements

def prep_df_for_bar_chart(reshaped_df, scen, column_list = Transport_Fuel_list):
    """
    
    Parameters
    ----------
    reshaped_df : Datafrane that has been pivoted from the default format. It is a multiindex with model, scenario and year as the
    index levels, and each variable is a column, with the values filling the columns.
    scen : scenario being studied
    column_list : list of columns to include in bar chart. The default is Transport_Fuel_list.

    Returns
    -------
    dft : dataframe to use for bar chart function. 

    """
    dft = reshaped_df[reshaped_df.index.get_level_values(1) == scen]
    dft = dft[dft.columns[dft.columns.isin(Transport_Fuel_list)]]
    dft = dft.rename(columns = Transport_Fuel_type_dict)
    dft = dft.dropna(axis = 0, how = 'all')
    dft['bioenergy'] = dft.filter(regex = 'bio').sum(axis = 1)
    dft = dft.reset_index()
    return dft




def plot_clustered_stacked(dfstack, scen, title_text = ' Transport Fuel Split for Years 2020, 2035 and 2050',
                           H="/"):    
    """
    Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot. 
    labels is a list of the names of the dataframe, used for the legend
    title is a string for the title of the plot
    H is the hatch used for identification of the different dataframe
    This is amended from: https://stackoverflow.com/questions/22787209/how-to-have-clusters-of-stacked-bars/22845857#22845857
    
    Parameters
    ----------
    dfstack : This dataframe should be in the format that is given by running the prep_df_for_bar_chart function. columns are 
    model, year, and then the categories of fuel use with values that can be added together. They correspond to a particular scenario.
    scen : scenario used for the df being plotted.
    H : tick mark used to delineate different models
        DESCRIPTION. The default is "/".

    Returns
    -------
    None. The bar chart appears and can be saved by the user.

    """
    
    title = scen + title_text
    axe = plt.figure(figsize = (20,12))
    axe = plt.subplot(111)

    yearlist = [2020, 2035, 2050]
    dfstack = dfstack[dfstack['year'].isin(yearlist)]
    dfstack = dfstack.loc[:, (dfstack != 0).any(axis = 0)]
#    dfstack = dfstack.drop(['biogas', 'biomass liquids'])

    stack_key = dfstack.model.unique()
    n_mod = len(stack_key)
    n_yr = len(yearlist)
    n_fuel = len(dfstack.columns) - 2

    dstack = {}
    for name in stack_key:
        dstack[name] = dfstack[dfstack['model'] == name]
        dstack[name] = dstack[name].drop('model', axis = 1)
        dstack[name].set_index('year', inplace = True)
        dstack[name].index.names = ['Index']

    for names in stack_key : # for each data frame
        axe = dstack[names].plot(kind="bar",
                      linewidth=0, width = 0.9,
                      stacked=True,
                      ax=axe,
                      legend=False,
                      grid=False,)  # make bar plots
                  
    h,l = axe.get_legend_handles_labels() # get the handles we want to modify
    for i in range(0, n_mod * n_fuel, n_fuel): # len(h) = n_fuel * n_mod
        for j, pa in enumerate(h[i:i+n_fuel]):
            for rect in pa.patches: # for each index
                rect.set_x(rect.get_x() + 1 / float(n_mod + 1) * i / float(n_fuel))
                rect.set_hatch(H * int(i / n_fuel)) #edited part     
                rect.set_width(1 / float(n_mod + 1))
           
    axe.set_xticks((np.arange(0, 2 * n_yr, 2) + 1 / float(n_mod + 1)) / 2.)
    axe.set_xticklabels(yearlist, rotation = 0)
    axe.set_xlabel('Years')
    axe.set_ylabel('EJ/yr')
    axe.set_title(title)
    
    n=[]        
    for i in range(n_mod):
        n.append(axe.bar(0, 0, color="gray", hatch=H * i))

    l1 = axe.legend(h[:n_fuel], l[:n_fuel], loc=2)

    labels = stack_key
    l2 = plt.legend(n, labels, loc=[1.01, 0.1]) 

    axe.add_artist(l1)
    plt.tight_layout()
    return



def stacked_bar_by_model(df, scen, yrs = [2020, 2035, 2050], title = ' Sectoral Split for Years 2020, 2035, 2050'):
    """
    Takes a dataframe (df) with total bioenergy for each sector (buildings, electricity, transport and industry) and
    creates a stacked bar plot with the breakdown of total bioenergy by sector for the years selected.
    df: should have 

    Parameters
    ----------
    df : dataframe. This one has six columns, named model, year, electricity, transport, buildings and industry. 
    It needs to be based on the scenario mentioned. The custom pivot can be used for each of those sectors.
    scen : string - scenario being visualized - the dataframe needs to be constructed with this scenario.
    yrs : list of years to include
        DESCRIPTION. The default is [2020, 2035, 2050].
    title : string describing the visualization
        DESCRIPTION. The default is ' Sectoral Split for Years 2020, 2035, 2050'.

    Returns
    -------
    None. The bar chart appears and can be saved by the user.

    """

    models = list(df['model'].unique())
    width = 0.25
    
    col_to_use = 'year'
    var_to_use = yrs  
           
    b = {}
    e = {}
    t = {}
    ind = {}
    
    
    for i in var_to_use:
        b[i] = np.array(list(df['buildings'].loc[df[col_to_use]==i]))
        e[i] = np.array(list(df['electricity'].loc[df[col_to_use]==i]))
        t[i] = np.array(list(df['transport'].loc[df[col_to_use]==i]))
        ind[i] = np.array(list(df['industry'].loc[df[col_to_use]==i]))

  
    x_axis = np.arange(len(df['model'].unique()))
    label = models
    xlabels = 'Models'
    newb = b
    newe = e
    newt = t
    newi = ind
    
    fig, ax = plt.subplots() 
    q = 0
    for i in yrs:
        ax.bar(x_axis + q*width + 0.05*q, newb[i], width, label = 'Buildings', color = 'blue') #, edgecolor = 'k')
        ax.bar(x_axis + q*width + 0.05*q, newe[i], width, bottom= newb[i], label = 'Electricity', color = 'orange') #, edgecolor = 'k')
        ax.bar(x_axis + q*width + 0.05*q, newt[i], width, bottom = newb[i]+newe[i], label = 'Transport', color = 'green') #, edgecolor = 'k')
        ax.bar(x_axis + q*width + 0.05*q, newi[i], width, bottom = newb[i]+newe[i]+newt[i], label = 'Industry', color = 'red') #, edgecolor = 'k')
        q = q + 1

    ax.set_xticks(x_axis+width, list(label), rotation = 'vertical')
    ax.set_ylabel('EJ/yr')
    ax.set_xlabel(xlabels)
    ax.set_title(scen+title)
    color_scheme = [Line2D([0], [0], color = 'blue', lw=4), Line2D([0], [0], color = 'orange', lw=4), 
                    Line2D([0], [0], color = 'green', lw=4), Line2D([0], [0], color = 'red', lw=4), 
                    Line2D([0], [0], color = 'gray', lw=4), 
                    Line2D([0], [0], color = 'violet', lw=4), Line2D([0], [0], color = 'pink', lw=4)]
    ax.legend(color_scheme, ['Buildings', 'Electricity', 'Transport', 'Industry'],bbox_to_anchor=(1.04, 1), borderaxespad=0)

    plt.show()
    return

