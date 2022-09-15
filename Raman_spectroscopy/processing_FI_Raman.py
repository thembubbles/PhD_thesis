#!/usr/bin/env python
# coding: utf-8
# %%
# # Raman data processing
# #### Calculating area under peak from Raman files and processing to gas mol%

# %%
# ---------------- processing fluid inclusion Raman data (Raman spectroscopy)

# --- import modules

import os
import glob
import pandas as pd
import numpy as np

import warnings

from module_peak_area import peak_area, cross_section

# --- set working directories
os.chdir("..")

root_dir = os.getcwd()
base_dir = root_dir + "/Data/3-Raman/processed/"


# %%
# --- look into working directory and create a list with selected files - .txt
os.chdir(base_dir)

FI_files = glob.glob('*-v*.txt')

# print(FI_files)
# print(len(FI_files))

df_Raman = pd.DataFrame()


np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings('ignore')


for file in FI_files:
#     print(file)
    current_file = pd.read_csv(base_dir+file,
                    encoding = "ANSI", sep = '\t', names=('x','y'), comment="#")

    # --- add file information
    current_inclusion = file.replace(".txt", "")
    
    
    data = peak_area(current_file,current_inclusion)
    a = pd.DataFrame(data).T
    
    a.insert(0, 'Sample', current_inclusion)
    
    
    df_Raman = pd.concat([df_Raman,a], axis=0, ignore_index=True)
    
df_Raman = df_Raman.fillna(0)

df_Raman


# %%
# #copy data from source dataframe
df_V_mol = df_Raman.copy()

#split source Sample columns into relevant columns
df_V_mol[['sample','piece','field','analysis','rest']
        ] = df_V_mol["Sample"].str.split(pat='-', 
                                         n=4,  
                                         expand=True).replace(('w','bu'), 
                                                              ('','BU18FA'), 
                                                              regex=True)
#adjust values format to fit with other datasets
df_V_mol['piece'] = df_V_mol['piece'].str.upper()
df_V_mol['field'] = df_V_mol['field'].astype(str).replace(('FF'),('ff'),regex=True)
df_V_mol['analysis'] = df_V_mol['analysis'].astype(str).replace(('FI'),('fi'),regex=True)

df_V_mol['analysis'] = df_V_mol['analysis'].astype(str).replace(('10','11','12'),('010','011','012'),regex=True)

#retrive laser wavelength from analysis name
df_V_mol['laser'] = np.where(df_V_mol['rest'].str.contains('633', regex=True),
                            633,
                            np.where(df_V_mol['rest'].str.contains('785', regex=True),
                                     785,
                                     532)
                           )

#Here we calculate the Raman scattering cross-section, the function script is in the module_peak_area.py file
df_V_mol['cross_section_CO2_v1'] = cross_section("CO2_v1",df_V_mol['laser'])
df_V_mol['cross_section_CO2_v2'] = cross_section("CO2_2v2",df_V_mol['laser'])
df_V_mol['cross_section_N2'] = cross_section("N2",df_V_mol['laser'])
df_V_mol['cross_section_CH4'] = cross_section("CH4",df_V_mol['laser'])


#Here the actual molar concentration calculation takes place - peak area divided by peak cross-section 
df_V_mol['CO2_v1_mol'] = df_V_mol['area_CO2_v1'] / cross_section("CO2_v1",df_V_mol['laser'])
df_V_mol['CO2_v2_mol'] = df_V_mol['area_CO2_v2'] / cross_section("CO2_2v2",df_V_mol['laser'])
df_V_mol['N2_mol'] = df_V_mol['area_N2'] / cross_section("N2",df_V_mol['laser'])
df_V_mol['CH4_mol'] = df_V_mol['area_CH4'] / cross_section("CH4",df_V_mol['laser'])

#Sum each of the molar concentrations
df_V_mol['mol_sum'] = df_V_mol['CO2_v1_mol'] + df_V_mol['CO2_v2_mol'] + df_V_mol['N2_mol'] + df_V_mol['CH4_mol'] 

#molar pencertages from normalized molar concentrations
df_V_mol['XCO2(mol%)'] = round(((df_V_mol['CO2_v1_mol']/df_V_mol['mol_sum']) + (df_V_mol['CO2_v2_mol']/df_V_mol['mol_sum'])),3)
df_V_mol['XN2(mol%)'] = round((df_V_mol['N2_mol']/df_V_mol['mol_sum']),3)
df_V_mol['XCH4(mol%)'] = round((df_V_mol['CH4_mol']/df_V_mol['mol_sum']),3)




df_FI_V = df_V_mol[['sample', 'piece', 'field', 'analysis',
                     'XCO2(mol%)', 'XN2(mol%)', 'XCH4(mol%)',
                     'area_CO2_v1', 'area_CO2_v2', 'area_N2', 'area_CH4', 
                      'CO2_v1_mol', 'CO2_v2_mol', 'N2_mol', 'CH4_mol',
                      'laser','rest'
                      ,'cross_section_CO2_v1','cross_section_CO2_v2','cross_section_N2','cross_section_CH4'
                     ]
                    ]


df_FI_V.to_csv(root_dir+'/Data/3-Raman/Buranga_FI_Raman_V_composition.csv',index=False)
df_FI_V

