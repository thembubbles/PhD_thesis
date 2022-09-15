#!/usr/bin/env python
# coding: utf-8

# In[5]:


# ---------------- converting Element% to Oxide mass%


# --- import modules

import os
import glob
import pandas as pd
import numpy as np
import re # 're' stands for regular expressions.

# --- set working directory

base_dir = "_DATA/"


# # From EDS data

# In[2]:


#--- look into working directory and create a list with selected files - .txt
os.chdir(base_dir)
txt_files = glob.glob('*.txt')

print(txt_files)
len(txt_files)


# In[3]:


# - EDS data processing function made in collaboration with Bram Paredis

def EDS(i):
    # -- Create a dataframe from the txt file 
    df = pd.read_csv(txt_files[i], header=None, sep='\s+', names=range(30))

    # -- Get filename info to use for further auto df construction
    filename_info = re.split("-|_", txt_files[i])


# ----------------


    # -- Specify the names of the point measurements
    table_names = [f"pt{i}" for i in range(1, 30)]

    # -- Perform a check to see which rows in the dataframe belong to a certain
    # -- point measurement (called group here)
    groups = df[0].isin(table_names).cumsum()
    
    # Let's keep track of with how many 'point measurements we're dealing
    n_groups = groups.unique().size


# ----------------


    # -- Create a dictionary of dataframes in which the names of the point
    # -- measurements will be used as the dict keys and the according 
    # -- dataframes will be used as dict values - dict = {pt1:values}. 
    tables = {g.iloc[0,0]: g.iloc[1:].dropna(axis=1, how='all') 
      for k,g in df.groupby(groups)}

    # -- Only select the values we're interested in
    tables = {k: v.iloc[1:, :2].T for k,v in tables.items()}

    # -- Change column names to first row of df and strip '*'
    for df in tables.values():
df.columns = df.iloc[0, :].str.strip("*")

    # -- Drop the now redundant row with column name info
    tables = {k: v.drop(0) for k,v in tables.items()}


# ----------------


    df_area = pd.DataFrame()

    for k,v in tables.items():
df_area = pd.concat((df_area, v),sort=False)


# ----------------


    # -- Add additional info to df
    df_area["Analysis"] = table_names[:n_groups]
    df_area["Sample"] = filename_info[0]
    df_area["Area"] = filename_info[1]
    
    return df_area


# In[55]:


i = 0
b = pd.DataFrame()
for file in txt_files:
    print(file)
    a = EDS(i)
    b = pd.concat((b, a),sort=False, ignore_index = True)
    i = i + 1
   
        
b = b[["Sample", "Area", "Analysis", "O", "F", "Na", "Mg", "Al", "Si", "P", "Cl", 
       "Ca", "Mn", "Fe", "Ti", "Sr", "Nb", "Ba", "Ta", "C", "Total"]]

#b.to_csv('out_spectra.csv', index=False)
b


# In[66]:


# --- Separate data from attributes and fill NA values with 0.00 to allow calculations
b_data = b.drop(["Sample", "Area", "Analysis", "O", "C", "Total"], axis=1).fillna(0.0000)

#Convert data to float to allow calculations
b_data = b_data.astype(float)

print(b_data)


# In[67]:


# --- Multiply wt% ELEMENT by numerical value below for equivalent expressed as OXIDE.

#idx = pd.Index(["F", "Na", "Mg", "Al", "Si", "P", "Cl", "Ca", "Mn", "Fe", "Ti", "Sr", "Nb", "Ba", "Ta"], name='element')
conversion =  np.array([1.0000, 1.3480, 1.6582, 1.8895, 2.1392, 2.2916, 1.0000, 1.3992, 1.2912, 1.2865, 1.6681, 1.1826, 1.4305, 1.1165, 1.2211])

oxides = b_data.mul(conversion, axis=1)

total_sum = oxides.sum(axis=1)
print(total_sum)

oxides.columns = ['F', 'Na2O', 'MgO', 'Al2O3', 'SiO2', 'P2O5', 'Cl', 'CaO', 'MnO', 'FeO', 'TiO2', 'SrO',
                 'Nb2O5', 'BaO', 'Ta2O5']

oxides
#result.to_csv('out_oxides.csv', index=False)  


# In[59]:


def normalize(data, total=None):
    """Normalize data to 100%"""
    if total is None:
        total = data.sum(axis=1)
    return data.divide(total, axis=0) * 100


# In[71]:


ox_normalized = normalize(oxides)

ox_normalized['total'] = ox_normalized.sum(axis=1)
ox_normalized


# In[73]:


el_to_ox = b.drop(["O", "F", "Na", "Mg", "Al", "Si", "P", "Cl", 
       "Ca", "Mn", "Fe", "Ti", "Sr", "Nb", "Ba", "Ta", "C", "Total"], axis=1)

el_to_ox = pd.concat([el_to_ox,ox_normalized], axis=1)

el_to_ox.to_csv('oxides.csv', index=False)
el_to_ox


# # From LAICPMS data

# In[1]:


def ppmTOpct(ppm):
    """Convert values from ppm to wt% of element"""
    wt_pct = ppm / 10000
    
    return wt_pct


# In[7]:


df_data = pd.read_csv('C:/Users/u0125722/OneDrive - KU Leuven/Buranga/_Article3-MH_transition/_git/Data/4-LAICPMS/Buranga_host_LAICPMS_ppm.csv',
                          encoding = "ANSI")

df_data.columns


# In[33]:


df_ppm = df_data.drop(['sample', 'piece', 'field', 'analysis', 'Host','NbTa', 'Nb/Ta', 'Li/Na', 'K/Na', 'K/Cs', 'K/Rb',
       'Cs/Na', 'Rb/Na', 'Rb/Cs', 'Ta#', 'Mn#', 'Na%', 'Li%', 'K%', 'Rb%',
       'Cs%', 'B%', 'P%', 'Alkalis', 'Alkalis%', 'Time', 'Info'], axis=1)

df_metadata = df_data.drop(['Li', 'B', 'Na', 'Mg',
       'Al', 'Si', 'P', 'K', 'Ti', 'Mn', 'Rb', 'Sr', 'Nb', 'Sn', 'Cs', 'Ba',
       'Ta', 'W', 'Ca', 'Fe','NbTa', 'Nb/Ta', 'Li/Na', 'K/Na', 'K/Cs', 'K/Rb',
       'Cs/Na', 'Rb/Na', 'Rb/Cs', 'Ta#', 'Mn#', 'Na%', 'Li%', 'K%', 'Rb%',
       'Cs%', 'B%', 'P%', 'Alkalis', 'Alkalis%', 'Time'], axis=1)

df_ppm


# In[35]:


df_pct = df_ppm/10000

df_pct.columns = ['Li2O', 'B2O3', 'Na2O', 'MgO', 'Al2O3', 'SiO2', 'P2O5', 'K2O', 'TiO2', 'MnO', 'Rb2O', 'SrO',
       'Nb2O5', 'SnO2', 'Cs2O', 'BaO', 'Ta2O5', 'WO3', 'CaO', 'FeO']

df_pct


# In[36]:


# --- Multiply wt% ELEMENT by numerical value below for equivalent expressed as OXIDE.

    # --- create a reference dataframe with data of oxides

df_reference = pd.read_csv("C:/Users/u0125722/OneDrive - KU Leuven/Python_Scripts/_DATA/_Oxides_mass.csv",index_col=0)

reference_oxides = df_reference[df_pct.columns]

reference_oxides

#     # --- create an array with molecular weights of oxides
    
molecular_weights = reference_oxides.iloc[0,:]

#     # --- create an array with the convertion factor between element and oxide

conv_factor = reference_oxides.iloc[1,:]


#     # --- new dataframe with converted oxide mass% from element mass%
    
df_ox = df_pct.div(conv_factor, axis=1)
df_ox['total'] = df_ox.sum(axis=1)

df_oxide = pd.concat([df_metadata, df_ox], axis=1)
df_oxide


# In[ ]:




