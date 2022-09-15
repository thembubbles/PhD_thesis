#!/usr/bin/env python
# coding: utf-8
# Author: Fernando Prado Araujo

# # This script converts *Element mass%* from EPMA analysis to *Cation apfu*

# In[1]:

# --- import modules

import os
import pandas as pd
import numpy as np

# --- set working directory
os.chdir("..")

root_dir = os.getcwd()
data_dir = root_dir + "/Data/Mineral_chemistry/"

# ## 1. Reading the raw data

# In[2]:


# --- Create a dataframe from the raw data csv file 

df_raw = pd.read_csv(data_dir + "1-EPMA_raw.csv", header=0, sep=',', engine = 'python')

df_raw


# In[3]:


df_raw.columns


# ## 2. Converting the **ELEMENT mass%** data to **OXIDES mass%**

# In[4]:


# --- Separate data from attributes and fill values below detection limit with 0.00001 to allow calculations

df_metadata = df_raw[['Facies', 'Sample', 'Area',  'Comment', 'Mineral', 'Texture', 'Total']]

df_data = df_raw.drop(['Facies', 'Sample', 'Area',  'Comment', 'Mineral', 'Texture', 'Total'], axis=1).fillna(0.00001)
df_data = df_data.astype(float) #Convert data to float to allow calculations

   
    # -- Rearrange the columns to follow an array of decreasing ionic charge, and change names to oxides 

df_data = df_data[['Ta', 'Nb', 'Sn', 'Ti', 'Al', 'Fe', 'Mn', 'Ca', 'Mg']]
df_data.columns = ['Ta2O5', 'Nb2O5', 'SnO2', 'TiO2', 'Al2O3', 'FeO', 'MnO', 'CaO', 'MgO']

df_data


# In[5]:


# --- For transforming element to oxide mass%, divide the element value by a conversion factor

    # -- First, read a reference dataframe with the info of all oxides
df_reference = pd.read_csv(data_dir + "_Oxides_mass.csv",index_col=0).T
        # - From this reference, only select the oxides used in the currrent dataset
reference_oxides = df_reference[df_data.columns]

    # -- Then divide the data by the reference values
df_oxides = df_data.div(reference_oxides.iloc[1], axis=1)

    # -- Sum the values from all oxides to assess the analyses quality 
df_oxides['Total_ox'] = df_oxides.sum(axis=1)

        # - If total is much lower than 100% (i.e., bad analyses), remove the data row
df_excluded = df_oxides.drop(df_oxides.loc[df_oxides['Total_ox']>95].index)
df_oxides.drop(df_oxides.loc[df_oxides['Total_ox']<90].index, inplace=True)


# --- Insert metadata at the start of dataframe

df_oxides.insert(0,'Facies',df_metadata['Facies'])
df_oxides.insert(1,'Sample',df_metadata['Sample'])
df_oxides.insert(2,'Area',df_metadata['Area'])   
df_oxides.insert(3,'Comment',df_metadata['Comment'])
df_oxides.insert(4,'Mineral',df_metadata['Mineral'])
df_oxides.insert(5,'Texture',df_metadata['Texture'])

    # -- Export the result

# df_oxides.to_csv(data_dir + '2-EPMA_oxides.csv', index=False)  
df_oxides


# In[6]:


# --- Here a side calculation can be done to transform oxide mass% into mol%

df_analysis = df_oxides.drop(['Facies', 'Sample', 'Area', 'Comment', 'Mineral', 'Texture', 'Total_ox'], axis=1)

    # -- First divide the oxide data by the oxide molecular weight in the reference values
df_moles = df_analysis.div(reference_oxides.iloc[0], axis=1)


    # -- Sum the mol amount of all oxides to get the total molar amount 
Sum_moles = pd.Series(df_moles.sum(axis=1))


    # -- Then, divide the mol amount of each oxide by the total mol amount to get the mol% of each oxide
df_mol_pcent = df_moles.div(Sum_moles, axis=0)*100

df_mol_pcent['Total_mol'] = df_mol_pcent.sum(axis=1)


# --- Insert metadata at the start of dataframe

df_mol_pcent.insert(0,'Facies',df_metadata['Facies'])
df_mol_pcent.insert(1,'Sample',df_metadata['Sample'])
df_mol_pcent.insert(2,'Area',df_metadata['Area'])   
df_mol_pcent.insert(3,'Comment',df_metadata['Comment'])
df_mol_pcent.insert(4,'Mineral',df_metadata['Mineral'])
df_mol_pcent.insert(5,'Texture',df_metadata['Texture'])

#         # -- Export the result

# df_mol_pcent.to_csv(data_dir + '3-EPMA_moles.csv', index=False)  
df_mol_pcent


# ## 3. Recalculating the *OXIDES mass%* into *Cation APFU* in the mineral formula

# In[7]:


# --- separate data from metadata - only do calculation in data afterwards

df_analysis = df_oxides.drop(['Facies', 'Sample', 'Area', 'Comment', 'Mineral', 'Texture', 'Total_ox'], axis=1)


    # -- create an array with molecular weights of oxides
molecular_weights = reference_oxides.iloc[0,:]


    # -- create an array with the amount of oxygen in each of the oxides
oxygen_in_ox = reference_oxides.iloc[2,:]


    # -- create an array with the ratio between cations and oxygen in oxides
cation_ratio_ox = reference_oxides.iloc[3].div(reference_oxides.iloc[2])


    # -- new dataframe with measured mass% divided by oxides molecular weights
df_mol = df_analysis.div(molecular_weights, axis=1)


    # -- create a dataframe with the oxygen numbers
df_oxygen_N = df_mol.mul(oxygen_in_ox,axis=1)


    # -- add sum of anionic proportions to the last colum in dataframe
total_oxygen = df_oxygen_N.sum(axis=1)


    # -- amount of oxygen in the formula unit - basis for formula calculation
O_in_formula = 24

    # -- Oxygen Normalization Factor = number of oxygen in the formula divided by sum of anions
ONF = (O_in_formula/total_oxygen)


    # -- Anionic proportion = oxygen number multiplied by ONF
df_anionic = df_oxygen_N.mul(ONF,axis=0)


    # -- Cationic proportion = anionic proportion multiplied by oxide cation ratio
df_cationic = df_anionic.mul(cation_ratio_ox)

    # -- rename dataframe columns for simplification
df_cationic.columns = ['Ta', 'Nb', 'Sn', 'Ti', 'Al', 'Fe', 'Mn', 'Ca', 'Mg']

    # -- combine calculated data with metadata
df_cationic['Sum'] = df_cationic.sum(axis=1,numeric_only=True)

# --- Insert metadata at the start of dataframe
df_cationic.insert(0,'Facies',df_metadata['Facies'])
df_cationic.insert(1,'Sample',df_metadata['Sample'])
df_cationic.insert(2,'Area',df_metadata['Area'])   
df_cationic.insert(3,'Comment',df_metadata['Comment'])
df_cationic.insert(4,'Mineral',df_metadata['Mineral'])
df_cationic.insert(5,'Texture',df_metadata['Texture'])


# df_cationic.to_csv(data_dir + '4-EPMA_apfu.csv', index=False)
df_cationic


# ## 4. Combining all the elements into a single file and exporting it

# In[8]:


# --- Add identifier columns for each of the dataframes to separate them in the output file
df_raw.insert(0,"element","")
df_oxides.insert(0,"oxides","") 
df_cationic.insert(0,"apfu","") 

# --- Concatenate all the dataframes (raw, oxides, and apfu) into one dataframe. Also drop empty lines and duplicate columns
df_output = pd.concat([df_raw, df_oxides, df_cationic], axis = 1)
df_output = df_output.dropna(subset=["Sum"]).drop(['Facies','Sample', 'Area', 'Comment', 'Mineral', 'Texture'],axis=1)

# --- Insert metadata at the start of dataframe
df_output.insert(0,'Facies',df_metadata['Facies'])
df_output.insert(1,'Sample',df_metadata['Sample'])
df_output.insert(2,'Area',df_metadata['Area'])   
df_output.insert(3,'Comment',df_metadata['Comment'])
df_output.insert(4,'Mineral',df_metadata['Mineral'])
df_output.insert(5,'Texture',df_metadata['Texture'])


# --- Export combined file
# df_output.to_csv('Supplementary_material-S2-Mineral_chemistry_data.csv', index=False)
df_output


# ## 5. Statistical analysis of the data

# In[9]:


df_mean = df_output.groupby(['Facies','Mineral'], as_index=False).mean()

df_std = df_output.groupby(['Facies','Mineral'], as_index=False).std()

df_stats = pd.concat([df_mean, df_std], axis = 0)


# df_stats.to_csv(data_dir + '5-EPMA_statistics.csv', index=False)
df_stats


# In[ ]:




