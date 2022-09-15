#!/usr/bin/env python
# coding: utf-8
# %%
#Author: Fernando Prado Araujo

# %%
# ---------------- importing reduced LAICPMS data files

    # - LAICPMS data processed using SILLS for data reduction

# --- import modules

import os
import glob
import pandas as pd
import numpy as np

# --- set working directories
os.chdir("..")

root_dir = os.getcwd()
base_dir = root_dir + "/Data/4-LAICPMS/"


# %%
# --- look into working directory and create a list with selected files - .xls
os.chdir(base_dir)
txt_files = glob.glob('*.xls')

print(txt_files)
len(txt_files)


# %%
# --- create a single dataframe with all files in the list

df_compo_raw = pd.DataFrame()
df_error_raw = pd.DataFrame()
df_LOD_raw = pd.DataFrame()
df_host_compo_raw = pd.DataFrame()

    # -- loop in the file list, get relevant data and append to combined dataframe
for file in txt_files:
    print(file)
    df = pd.read_excel(file, skiprows = (0,1,2,3,4,5,6,7,9), na_values=None) #skiprows in the start of the file, before actual data
    blank_df = df.loc[df.isnull().all(1)] #creates a dataframe showing where blank lines are located

        # - here is a condition to remove everything after the relevant data (multidata input file, only first part is required)
    if len(blank_df) > 0: #check if there are indeed blank lines in the df
        composition_index = blank_df.index[0]  #locate index of the composition summary (first blank line)
        error_index = blank_df.index[2]  #locate index of the error summary (third blank line)
        lod_index = blank_df.index[4]  #locate index of the detection limits summary (fifth blank line)
        host_index = blank_df.index[12]  #locate index of the host composition summary (third blank line)
        
        df_compo = df[:composition_index]  #filters data from start of dataframe up to the first blank line
        df_error = df[(composition_index+5):error_index]
        df_lod = df[(error_index+5):lod_index]
        df_host_compo = df[(lod_index+13):host_index]
        
    df_compo_raw = pd.concat((df_compo_raw,df_compo),sort=False,ignore_index = True)
    df_error_raw = pd.concat((df_error_raw,df_error),sort=False,ignore_index = True)
    df_LOD_raw = pd.concat((df_LOD_raw,df_lod),sort=False,ignore_index = True)
    df_host_compo_raw = pd.concat((df_host_compo_raw,df_host_compo),sort=False,ignore_index = True)

    
to_process = df_compo_raw

    #work source File columns into relevant columns
to_process['Sample'] = to_process['File'].replace(('.csv','BU','19A7b','Ar7','22A-3-','22C-3-','20B-1-1','A','B','C','D','E','J'),
                                          ('','','19-A-7b-','F-r7','22-A-3a-','22C-1-','20B-2-1','-A-','-B-','-C-','-D-','-E-','-J-'), 
                                          regex=True)

to_process['Sample'] = to_process['Sample'].replace(('--','19-11b','32-A-II','192','-Mbs'),
                                          ('-','19-E-11b','32-AII','19(2)-A-','-1-Mbs'), 
                                          regex=True)

to_process['Sample'] = to_process['Sample'].replace(('19\(2\)-2-','19\(2\)2-'),
                                          ('19(2)-B-2-','19(2)-A-2-'), 
                                          regex=True)
            
to_process[['sample','piece','field','analysis']
        ] = to_process['Sample'].str.split(pat='-', 
                                         n=3,  
                                         expand=True)


    #Add Host name column to dataframe

wyl = ((to_process['sample'] == '32') & (to_process['piece'] == 'AII')) | (to_process['analysis'] == 'wyl')

qtz = ((to_process['sample'] == '22') & ((to_process['piece'] == 'A')|(to_process['piece'] == 'B'))) | ((to_process['sample'] == '19') & (to_process['field'] == '11b'))

bts = ((to_process['sample'] == '20') & ((to_process['piece'] == 'B')|(to_process['piece'] == 'C'))) | (((to_process['sample'] == '4')|(to_process['sample'] == '04')) & ((to_process['field'] == '3')|(to_process['field'] == '6')|(to_process['field'] == '7'))) | ((to_process['sample'] == '19') & (to_process['field'] == '11'))|(to_process['analysis'] == 'B-st')|((to_process['sample'] == '22') & (to_process['piece'] == 'C')) 

trl = ((to_process['sample'] == '20') & ((to_process['piece'] == 'J')|(to_process['piece'] == 'E')))| (((to_process['sample'] == '4')|(to_process['sample'] == '04')) & ((to_process['field'] == '5')|(to_process['field'] == '8')))|((to_process['sample'] == '19') & ((to_process['piece'] == 'A')|(to_process['piece'] == 'B')))|(to_process['analysis'] == 'trl')|(to_process['analysis'] == 'Trl')

aug = ((to_process['sample'] == '19(2)') & (to_process['piece'] == 'A'))

brz = ((to_process['sample'] == '32') & ((to_process['piece'] == 'A')|(to_process['piece'] == 'B'))) | (to_process['sample'] == '2a')

rsm = ((to_process['sample'] == '19') & (to_process['field'] == 'r7') & ((to_process['analysis'] == 'M')|(to_process['analysis'] == 'M2'))) | ((to_process['sample'] == '19(2)') & (to_process['analysis'] == 'M'))

mbs = ((to_process['sample'] == '20') & (to_process['piece'] == 'D')) | (to_process['analysis'] == 'mbs')

scz = (to_process['analysis'] == 'scz')


df['Host'] = np.nan


to_process.loc[wyl, 'Host'] = 'wyllieite'
to_process.loc[qtz, 'Host'] = 'quartz'
to_process.loc[bts, 'Host'] = 'bertossaite'
to_process.loc[aug, 'Host'] = 'augelite'
to_process.loc[brz, 'Host'] = 'brazilianite'
to_process.loc[trl, 'Host'] = 'trolleite'

to_process.loc[rsm, 'Host'] = 'rosemaryite'
to_process.loc[mbs, 'Host'] = 'montebrasite'
to_process.loc[scz, 'Host'] = 'scorzalite'

    
# print(df_raw.columns)

# df_raw.to_csv(base_dir+'Buranga_data_LAICPMS_combined.csv',index=False)

to_process


# %%
df_data = to_process.drop(['File','Sample','sample', 'piece', 'field', 'analysis', 'Time', 'Info','Sb121', "Host"], axis=1)

#separate data from metadata - only do calculation in data afterwards
df_metadata = to_process[['sample', 'piece', 'field', 'analysis', 'Time', 'Info', "Host"]]

#rename dataframe columns for simplification
df_data.columns = ['Li', 'B', 'Na', 'Mg', 'Al', 'P', 'K', 'Ca43','Ca44', 'Ti', 'Mn', 
                   'Fe56', 'Fe57', 'Rb', 'Nb', 'Sn','Cs', 'Ta', 'W','Si','Sr','Ba']

df_data[['Ca','Fe']] = np.nan

print(df_data.columns)


# df_data.to_csv(base_dir+'Buranga_LAICPMS_data.csv',index=False)

df_data


# %%
df_reference = pd.read_csv(root_dir + '/Data/_Oxides_mass.csv', 
                    encoding = "ANSI", 
                   index_col = 0)


df_reference.columns = ['H', 'Li', 'Be', 'B', 'C', 'F', 'Na', 'Mg', 'Al',
       'Si', 'P', 'S', 'Cl', 'K', 'Ca', 'Sc', 'Ti', 'Ti2O3',
       'V', 'Cr', 'Mn', 'Fe2O3', 'Fe', 'Ni', 'Co', 'Cu', 'Zn',
       'Ga', 'Ge', 'As', 'Rb', 'Sr', 'Y', 'Zr', 'Nb',
       'Mo', 'Sn', 'Sb', 'Cs', 'Ba', 'La', 'Ce', 'Ce2O3',
       'Pr', 'Nd', 'Sm', 'Eu', 'EuO', 'Gd', 'Tb', 'Dy',
       'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W',
       'Au', 'Pb', 'Th', 'U']


reference_oxides = df_reference[(df_reference.columns) & (df_data.columns)]



    # --- create an array with molecular weights of oxides
    
molar_mass = reference_oxides.iloc[4,:]



print(reference_oxides.columns, len(reference_oxides.columns), '\n\n', molar_mass)

reference_oxides


# %%
df_data_bdl = df_data.copy()

#replace values starting with "<" (i.e. below detection limit) to value/sqrt(value)
#replace negative values with 0
# for col_name, col_values in df_data_bdl.iteritems():
#     for index, value in enumerate(col_values):
#         if type(value) is str:
#             value = value.replace('<', '')
#             value = float(value)
#             if value < 0:
#                 df_data_bdl[col_name][index] = 0
#             else:
#                 df_data_bdl[col_name][index] = (value/(math.sqrt(2)))
#         elif value < 0:
#             df_data_bdl[col_name][index] = 0    
            
            
df_data_bdl = df_data.replace({'<': np.nan},regex=True) #remove values below detection limit from dataset

df_data_bdl[df_data_bdl <= 1e-5] = np.nan #remove values below quantification limit (i.e., too low) from dataset


#add columns in the dataframe using iloc characters --> min_count=1 allows to sum columns containing NaN
df_data_bdl.loc[:,'Ca'] = df_data_bdl.loc[:,['Ca43','Ca44']].sum(axis=1, min_count=1)
df_data_bdl.loc[:,'Fe'] = df_data_bdl.loc[:,['Fe56','Fe57']].sum(axis=1, min_count=1)

df_data_bdl = df_data_bdl.drop(['Ca43','Ca44', 'Fe56', 'Fe57'], axis=1)

# df_data_bdl.to_csv(base_dir+'Buranga_data_LAICPMS_bdl_new.csv',index=False)

df_data_bdl


# %%
# --- new dataframe with molar concentrations (measured ppm divided by molar weights * 1000) 
df_mol = df_data_bdl.div((molar_mass*1000), axis='columns')

# --- or new dataframe in ppm
# df_mol = df_data_bdl.copy()

df_mol.loc[:,'NbTa'] = df_mol.loc[:,['Nb','Ta']].sum(axis=1, min_count=1)
df_mol["Nb/Ta"] = df_mol['Nb'] / (df_mol['Ta'].replace(0, np.nan))
df_mol["Li/Na"] = df_mol['Li'] / df_mol['Na']
df_mol["K/Na"] = df_mol['K'] / df_mol['Na']
df_mol["K/Cs"] = df_mol['K'] / df_mol['Cs']
df_mol["K/Rb"] = df_mol['K'] / df_mol['Rb']
df_mol["Cs/Na"] = df_mol['Cs'] / df_mol['Na']
df_mol["Rb/Na"] = df_mol['Rb'] / df_mol['Na']
df_mol["Rb/Cs"] = df_mol['Rb'] / df_mol['Cs']

df_mol["Ta#"] = (df_mol['Ta'] / (df_mol['NbTa']))*100
df_mol["Mn#"] = (df_mol['Mn'] / (df_mol.loc[:,['Mn','Fe']].sum(axis=1, min_count=1)))*100

df_mol["Na%"] = df_data_bdl['Na'] / 10000
df_mol["Li%"] = df_data_bdl['Li'] / 10000
df_mol["K%"] = df_data_bdl['K'] / 10000
df_mol["Rb%"] = df_data_bdl['Rb'] / 10000
df_mol["Cs%"] = df_data_bdl['Cs'] / 10000

df_mol["P%"] = df_data_bdl['P'] / 10000
df_mol["B%"] = df_data_bdl['B'] / 10000

df_mol.loc[:,'diva'] = df_mol.loc[:,['Mg','Ca','Fe','Mn','Sr', 'Ba']].sum(axis=1, min_count=1)
df_mol.loc[:,"diva%"] = (df_data_bdl.loc[:,['Mg','Ca','Fe','Mn','Sr', 'Ba']].sum(axis=1, min_count=1)) / 10000

df_mol.loc[:,'Alkalis'] = df_mol.loc[:,['Li','Na','K','Rb','Cs']].sum(axis=1, min_count=1)
df_mol.loc[:,'Alkalis%'] = df_mol.loc[:,['Li%','Na%','K%','Rb%','Cs%']].sum(axis=1, min_count=1)


#combine calculated data with metadata
df_mol[['sample', 'piece', 'field', 'analysis', 'Host', 'Time', 'Info']] = df_metadata[['sample', 'piece', 'field', 'analysis', 'Host', 'Time', 'Info']]

df_mol['Info'] = df_mol['Info'].fillna('')
# df_mol = df_mol.fillna(0.0)

print(df_mol.columns)

df_mol


# %%
# df_hosts = df_mol.copy()

df_hosts = df_mol[df_mol['Info'].str.contains("matri|rutile")]

df_FI = df_mol[~df_mol['Info'].str.contains("matri|rutile")]

    # --- Dataframe for FI data

df_FI = df_FI[['sample', 'piece', 'field', 'analysis', 'Host',
               'Li', 'B', 'Na', 'Mg', 'Al', 'Si', 'P', 'K', 
               'Ti', 'Mn','Rb', 
               'Sr', 'Nb', 'Sn', 'Cs', 'Ba', 'Ta', 'W', 'Ca',
               'Fe', 'NbTa', 'Nb/Ta', 
               'Li/Na', 'K/Na', 'K/Cs', 'K/Rb', 'Cs/Na', 'Rb/Na', 'Rb/Cs',
               'Ta#','Mn#',
               'Na%', 'Li%', 'K%', 'Rb%','Cs%', 'B%', 'P%',
               'diva', 'diva%', 'Alkalis','Alkalis%',
               'Info']]


df_FI['sample'] = 'BU18FA' + df_FI['sample'].astype(str)
df_FI['sample'] = df_FI['sample'].astype(str).replace(('4'),('04'),regex=True)
df_FI['sample'] = df_FI['sample'].replace(('2a'),('02a'),regex=True)
df_FI['sample'] = df_FI['sample'].replace(('00'),('0'),regex=True)

df_FI['field'] = 'ff' + df_FI['field'].astype(str).replace(('.0'),(''),regex=True)
df_FI['field'] = df_FI['field'].astype(str).replace(('ffC'),('incC'),regex=True)


df_FI['analysis'] = df_FI['analysis'].astype(str).replace(('\.0','b'),('',''),regex=True)
df_FI['analysis'] = 'fi0' + df_FI['analysis'].astype(str)


    # --- Dataframe for hosts data

df_hosts = df_hosts[['sample', 'piece', 'field', 'analysis', 'Host',
                     'Li', 'B', 'Na', 'Mg', 'Al', 'Si', 'P', 'K', 
                     'Ti', 'Mn','Rb', 
                     'Sr', 'Nb', 'Sn', 'Cs', 'Ba', 'Ta', 'W', 'Ca',
                     'Fe', 'NbTa', 'Nb/Ta', 
                     'Li/Na', 'K/Na','K/Cs', 'K/Rb', 'Cs/Na', 'Rb/Na', 'Rb/Cs',
                     'Ta#','Mn#',
                     'Na%', 'Li%', 'K%', 'Rb%','Cs%', 'B%', 'P%',
                     'Alkalis','Alkalis%',
                     'Time', 'Info']]

df_hosts['sample'] = 'BU18FA' + df_hosts['sample'].astype(str)
df_hosts['sample'] = df_hosts['sample'].astype(str).replace(('4'),('04'),regex=True)
df_hosts['sample'] = df_hosts['sample'].replace(('2a'),('02a'),regex=True)
df_hosts['sample'] = df_hosts['sample'].replace(('00'),('0'),regex=True)

df_hosts['field'] = 'ff' + df_hosts['field'].astype(str).replace(('.0'),(''),regex=True)
df_hosts['field'] = df_hosts['field'].astype(str).replace(('ffC'),('incC'),regex=True)
df_hosts['field'] = df_hosts['field'].astype(str).replace(('ffr7'),('r7'),regex=True)

df_hosts['analysis'] = df_hosts['analysis'].astype(str).replace(('\.0','b'),('',''),regex=True)


df_FI.to_csv(base_dir+'Buranga_FI_LAICPMS_mol.csv',index=False)
df_hosts.to_csv(base_dir+'Buranga_host_LAICPMS_mol.csv',index=False)

# print(df_FI.columns)

# df_FI


# %%
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

df_plot = df_hosts.copy()

# df_plot = df_trl.query('sample == "BU18FA20"')


host_list = ['wyllieite', 'trolleite', 'bertossaite', 'augelite',
             'montebrasite','rosemaryite','scorzalite', 
             'brazilianite', 'quartz']

elem_list = ['Li', 'B', 'Na', 'Mg',
       'K', 'Ti', 'Mn', 'Rb',
       'Sr', 'Nb', 'Sn', 'Cs', 'Ba', 'Ta', 'W', 'Ca', 'Fe']

# fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16,5))

# Histogram + kernel density function curve

# ax = sns.displot(df_plot, x="K%", hue="Host", multiple='stack', kde=True)#col="Host"
# sns.kdeplot(data=df_plot, x="K", hue="Host")





# Boxplot + data on top

x = 'Host'



# ax = sns.violinplot(x=x, y=y, data=df_plot, showfliers = False, hue="sample")
for elem in elem_list:
    fig = plt.figure(figsize =(30, 12))

    sns.set_style("whitegrid")

    y = elem
    ax = sns.stripplot(data=df_plot, 
                       x=x, y=y,
                       color='k', order = host_list, size=8, jitter=True, alpha=0.8)
    ax = sns.boxplot(data=df_plot, 
                     x=x, y=y,
                     showfliers = False, order = host_list)
    ax.set(yscale="log")


    for i,box in enumerate(ax.artists):
        box.set_edgecolor('black')
        box.set_facecolor('lightgrey')



# # Binary distribution map

# sns.displot(df_plot, x="K%", y="Na%", hue="Host", kind="kde")




# # Binary scatter plot 

# sns.scatterplot(data=df_plot, x="K%", y="Na%", hue="Host")
# ax.set_xlim([0, 12.5])




# Mixed plots

# plot = sns.jointplot(data=df_plot, x="K%", y="B%", hue="Host")

# plot.ax_marg_x.set_xlim(-2.5, 7.5)
# ax.set_ylim(0.001, 30000)



    # Add artistic features to graph
    plt.title(y, fontsize=30, fontname="Arial",fontweight="bold",y=0.92)
    # plt.xlabel('Host', fontsize=26)
    plt.ylabel('concentration (ug/g)', fontsize=26)


    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

    # ax.set_yticks([1e-2, 1e-1, 1e0,10,100,1000,1e4,1e5,1e6,1e7])

    # ax.set_ylim(1e-1, 1e5)

    # plt.savefig(base_dir+'graphs/plot_boxp_'+y+'.pdf')
    plt.show()

   


# %% [markdown]
# # Convert host data from element ppm to oxide mass%

# %%
df_data = pd.read_csv(base_dir + 'Buranga_host_LAICPMS_ppm.csv',
                          encoding = "ANSI")

df_ppm = df_data.drop(['sample', 'piece', 'field', 'analysis', 'Host','NbTa', 'Nb/Ta', 'Li/Na', 'K/Na', 'K/Cs', 'K/Rb',
       'Cs/Na', 'Rb/Na', 'Rb/Cs', 'Ta#', 'Mn#', 'Na%', 'Li%', 'K%', 'Rb%',
       'Cs%', 'B%', 'P%', 'Alkalis', 'Alkalis%', 'Time', 'Info'], axis=1)

df_metadata = df_data.drop(['Li', 'B', 'Na', 'Mg',
       'Al', 'Si', 'P', 'K', 'Ti', 'Mn', 'Rb', 'Sr', 'Nb', 'Sn', 'Cs', 'Ba',
       'Ta', 'W', 'Ca', 'Fe','NbTa', 'Nb/Ta', 'Li/Na', 'K/Na', 'K/Cs', 'K/Rb',
       'Cs/Na', 'Rb/Na', 'Rb/Cs', 'Ta#', 'Mn#', 'Na%', 'Li%', 'K%', 'Rb%',
       'Cs%', 'B%', 'P%', 'Alkalis', 'Alkalis%', 'Time'], axis=1)

print(df_ppm.columns)
df_ppm

# %%
# --- Convert ELEMENT PPM to wt% ELEMENT and multiply by conversion factor for equivalent value as OXIDE.

    # --- change name of columns to match final oxides
df_ppm.columns = ['Li2O', 'B2O3', 'Na2O', 'MgO', 'Al2O3', 'SiO2', 'P2O5', 'K2O', 'TiO2', 'MnO', 'Rb2O', 'SrO',
       'Nb2O5', 'SnO2', 'Cs2O', 'BaO', 'Ta2O5', 'WO3', 'CaO', 'FeO']

    # --- ELEMENT ppm to wt%
    
df_pct = df_ppm/10000    

    # --- create a reference dataframe with data of oxides

df_reference = pd.read_csv(root_dir + "/DATA/_Oxides_mass.csv",index_col=0)

reference_oxides = df_reference[df_pct.columns]

#     # --- create an array with the conversion factor between element and oxide

conv_factor = reference_oxides.iloc[1,:]


#     # --- new dataframe with converted OXIDE mass% from ELEMENT mass%
    
df_ox = df_pct.div(conv_factor, axis=1)
df_ox['total'] = df_ox.sum(axis=1)

df_oxide = pd.concat([df_metadata, df_ox], axis=1)

df_oxide.to_csv(base_dir+'Buranga_host_LAICPMS_oxides.csv',index=False)
df_oxide

# %%
