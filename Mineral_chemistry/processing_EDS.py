#!/usr/bin/env python
# coding: utf-8

# In[2]:


# ---------------- importing EDS data files

    # - EDS data processing script developed with the help of Bram Paredis

# --- import modules

import os
import glob
import pandas as pd
import re # 're' stands for regular expressions.

# --- set working directory

base_dir = "EPMA/"


# In[3]:


# --- look into working directory and create a list with selected files - .txt
os.chdir(base_dir + '/EDS')
txt_files = glob.glob('*.txt')

print(txt_files)
len(txt_files)


# # Solution for multiple files
# Applies the same code as before, but in a loop for every file in a folder

# In[5]:


def EDS(i):
    # Create a dataframe from the txt file 
    df = pd.read_csv(txt_files[i], header=None, sep='\s+', names=range(30))

    # Get filename info to use for further auto df construction
    # To better order things, put this at the start of your for loop/function instead 

    filename_info = re.split("-|_", txt_files[i])


        # ----------------


    # Specify the names of the point measurements
    table_names = [f"pt{i}" for i in range(1, 30)]


    # Perform a check to see which rows in the dataframe belong to a certain
    # point measurement (called group here)
    groups = df[0].isin(table_names).cumsum()


    # Let's keep track of with how many 'point measurements' we're dealing
    n_groups = groups.unique().size


        # ----------------


    # Create a dictionary of dataframes in which the names of the point
    # measurements will be used as the dict keys and the according 
    # dataframes will be used as dict values - dict = {pt1:values}. 
    tables = {g.iloc[0,0]: g.iloc[1:].dropna(axis=1, how='all') 
              for k,g in df.groupby(groups)}


    # Only select the values we're interested in
    tables = {k: v.iloc[2:, :2].T for k,v in tables.items()}


    # Change column names to first row of df and strip '*'
    for df in tables.values():
        df.columns = df.iloc[0, :].str.strip("*")

    # Drop the now redundant row with column name info
    tables = {k: v.drop(0) for k,v in tables.items()}


        # ----------------


    df_area = pd.DataFrame()

    for k,v in tables.items():
        df_area = pd.concat((df_area, v),sort=False)


        # ----------------


    # Add additional info to df
    df_area["Analysis"] = table_names[:n_groups]
    df_area["Sample"] = filename_info[0]
    df_area["Area"] = filename_info[1]
    return df_area


# In[11]:


i = 0
b = pd.DataFrame()
for file in txt_files:
    #print(file)
    a = EDS(i)
    b = pd.concat((b, a),sort=False,ignore_index = True)
    i = i + 1

b = b[["Sample", "Area", "Analysis", "F", "Na2O", "MgO", "Al2O3", "SiO2", "P2O5", "Cl", "CaO", 
       "MnO", "Fe2O3", "TiO2", "BaO", "Ta2O5", "SnO2", "Total"]]
    
    
# b = b[["Sample", "Area", "Analysis", "F", "Na2O", "MgO", "Al2O3", "SiO2", "P2O5", "Cl", "K2O", "CaO", 
#        "MnO", "FeO", "TiO2", "SrO", "Nb2O5", "BaO", "Ta2O5", "SnO2", "Yb2O3", "CuO", "ZnO", "PbO", "SO3", "UO2", 
#        "ZrO2", "Au2O3", "HfO2", "MoO3", "Total"]]

b.to_csv('BU18FA19(1)TS_oxides.csv', index=False)        
b


# In[7]:


c = pd.read_csv(base_dir+"/EDS/EDS_ele_to_oxides.csv", sep=',')

c


# In[8]:


EDS = pd.concat((b, c),sort=False,ignore_index = True).fillna(0.0)

EDS = EDS.replace(0.0,"")

EDS.to_csv('EDS.csv', index=False) 
EDS

