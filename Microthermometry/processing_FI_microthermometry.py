#!/usr/bin/env python
# coding: utf-8
# %%
# Author: Fernando Prado Araujo

# %%
# ---------------- processing fluid inclusion data (microthermometry + Raman spectroscopy)


# --- import modules

import os
import pandas as pd
import numpy as np


# --- set working directories
os.chdir("..")

root_dir = os.getcwd()
base_dir = root_dir + "/Data/2-microthermometry/"


# %%
# --- create dataframe from csv file using pandas import command 

df_FI = pd.read_csv(base_dir + "Buranga-FI-microth-data.csv", 
                    encoding = "ANSI", 
                   index_col = 0)



# --- print the colums of the imported dataframe - can be changed if necessary

print(df_FI.columns) 


df_FI


# %%
# --- create standard dataframe from csv 

df_std = pd.read_csv(base_dir + "Buranga-FI-microth-standards.csv", 
                    encoding = "ANSI",index_col=0, header=0).T


a = list(df_std.columns[1:]) #list with each entry date (column names) in the dataset
y = list(df_std["Standard"]) #list with expected values for the standards


    # -- compare measured standard against the expected values from the fluid inclusion standards
calibration_dict = dict()

for item in range(len(df_std.columns[1:])):  # loop for assessing each entry date

    x = list(df_std[a[item]]) #list with measured values for the current entry date
    
        # - here the data are fitted by a linear model to obtain the correction equation
        
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
            
        # - create a dictionary with the correction equation values
    calibration_dict[a[item]] = (z[0],z[1])
    
#     print(calibration_dict[a[item]],'\n',z[0],z[1],'\n')

print(calibration_dict)

df_std


# %%
# --- Use formula from calibration trendline to correct measured unknown data

    # -- remove unwanted data that could block the calculation
    
df_corrected = df_FI.drop(['sample', 'piece', 'field', 'Host', 'analysis', 'Length',
       'Width', 'Phases', 'VL (%)', 'VV (%)', 'VS (%)', 'Hmg Mode',
       'Type'], axis=1).T
    

    # -- save important metadata to a separate dataframe
    
df_corrected_metadata = df_FI[['sample', 'piece', 'field', 'Host', 'analysis', 
                               'Length', 'Width', 'Phases', 'VL (%)', 'VV (%)','Hmg Mode',
                               'Type']]


df_corrected = df_corrected.drop(index=['Tm Xts','Th total'])


    # -- walk through dataframe columns to perfom calculation according to calibration trendline
    
for column, values in df_corrected.iteritems():
#     print(column, '\n', values,'\n')
#     print(calibration_dict[values[0]][0], calibration_dict[values[0]][1],'\n')
#     print(df_corrected[column][1:],'\n\n')
#     print(df_corrected[column])

        # - grab values from calibration dictionary to perform calculations
    df_corrected[column][1:] = (df_corrected[column][1:]*calibration_dict[values[0]][0]) + calibration_dict[values[0]][1]


    # -- transpose back the dataframe to usual position and concatenate with metadata
    
df_corrected = df_corrected.T
df_corrected = pd.concat([df_corrected_metadata,df_corrected], axis=1)

df_corrected = df_corrected.drop(['Date'], axis=1)



df_corrected = df_corrected[['sample','piece', 'field', 'analysis', 'Host',
                            'Length', 'Width', 'Phases', 'VL (%)', 'VV (%)',
                            'Tm CO2','Te', 'Tm Hh', 'Tm H2O', 'Tm CL', 
                            'Th CO2','Hmg Mode']]


# --- Calculate the salinity of fluid inclusions using published equations 

#Equation from Darling (1991) GCA
df_corrected['salinity(NaCleq wt%)'] = 0.00098241*(10-df_corrected['Tm CL'])*((df_corrected['Tm CL']**2) + (45.385*df_corrected['Tm CL']) + 1588.75) 

#Equation from Chen (1972) as presented in Diamond (1994) GCA 56, pp.273-280
df_corrected['salChen(NaCleq wt%)'] = (15.6192 - (1.1406*df_corrected['Tm CL']) - (0.035*(df_corrected['Tm CL']**2)) - (0.0007*(df_corrected['Tm CL']**3)))

#Equation from Bodnar(1993). GCA, 57(3), 683-684.
df_corrected['salBod(NaCleq%)'] = (1.78*(0-df_corrected['Tm H2O']))-(0.0442*((0-df_corrected['Tm H2O'])**2))+(0.000557*(0-(df_corrected['Tm H2O']**3))) 


df_corrected.to_csv(base_dir + 'Buranga_FI_microthermometry_processed.csv')

print(df_corrected.columns)

df_corrected


# %%
df_wyl = df_corrected.query('Host == "wyllieite"')
df_trl = df_corrected.query('Host == "trolleite"')
df_bts = df_corrected.query('Host == "bertossaite"')
df_aug = df_corrected.query('Host == "augelite"')
df_brz = df_corrected.query('Host == "brazilianite"')
df_qtz = df_corrected.query('Host == "quartz"')
df_scz = df_corrected.query('Host == "scorzalite"')

to_print = df_bts.query('Phases == "L - L - V"')

# to_print

print(to_print.min(),'\n\n',
      to_print.max(),'\n\n',
      to_print.median(),'\n\n',
      to_print.std())
