#!/usr/bin/env python
# coding: utf-8
# %%
#Author: Fernando Prado Araujo


# # Formula Recalculation for Mineral Analyses
# 
# This code is the procedure for the recalculation of mineral formulas using as input the oxide content of mineral anlyses, i.e. mass% of oxides from EDS or WDS analyses

# ## 1. Divide oxide mass% input by oxide molecular weights
#    #### mass% / mol wt = mol amount 
#       Ex.: measured SiO2 in a biotite crystal = 52.8 mass%; molar weight of SiO2 = 60.0843
#       52.8 / 60.0843 = 0.878765
# 
# ## 2. Multiply mol amount by amount of oxygen in the oxide
#    #### mol amount * oxygen in oxide = oxygen number
#        Ex.: molar amount in our case = 0.878765; SiO2 has 2 oxygens in the oxide
#        0.878765 * 2 = 1.757531
#        
# ## 3. Sum up the calculated oxygen number for all oxides
# An exception here is the occurrence of F or Cl in the analyses. As these commonly occur as anions, they should be subtracted from the total sum. Both elements have -1 charge, which is half the charge of oxygen anions. Consequently, their calculated oxygen number should be multiplied by 0.5 before subtraction of the total.
#    #### Σ oxygen numbers of all cations - Σ (0.5 * oxygen numbers of F and Cl) = total oxygen
#        Ex.: oxygen number for SiO2 in our case = 1.757531; 
#        summing up with other oxides is = 4.58;
#        oxygen number for F and Cl is = 0.30 and 0.002 respectively 	
#        4.58 - [(0.5 * 0.3)+(0.5 * 0.002)] = 4.429
#        
# ## 4. Divide the number of oxygens in the mineral formula unit by total oxygen of last step
# The oxygen in the formula unit can be retrieved from the ideal mineral formula. For anhydrous minerals, this value is directly the number of oxygens listed (e.g. 4 in the case of olivine, (Fe,Mg)SiO4). For hydrated phases each other anion (OH, F, or Cl) will count as half oxygen and the final value will be the number of oxygens plus half the number of OH,F,Cl in the formula (e.g. 11 for biotite K(Mg,Fe)3AlSi3O10(OH,F)2)
#    #### O in formula unit / total oxygen = oxygen normalization factor 
#        Ex.: total oxygen in our case = 4.429; biotite has an equivalent value of 11 oxygens in its formula (considering also the amount of OH and F in the ideal formula - see paragraph above for further clarification)
#        11 / 4.429 = 2.484
#        
# ## 5. Multiply the oxygen number from item 2 by the oxygen normalization factor 
#    #### oxygen number * ONF = anionic proportion
#        Ex.: oxygen for SiO2 in our case = 1.757531; ONF = 2.484
#        1.757531 * 2.484 = 4.365
#        
# ## 6. Multiply the anionic proportion by the ratio of cations and oxygen in the oxide 
#    #### anionic proportion * (cations in oxide / oxygen in oxide) = cationic proportion
#        Ex.: in our case anionic proportion = 4.365; and the ratio between Si and O in the oxide formula (i.e. SiO2) is 1/2 = 0.5
#        4.365 * 0.5 = 2.183

# %%
# --- import required modules

import os
import numpy as np
import pandas as pd

# --- set working directories
os.chdir("..")

root_dir = os.getcwd()
base_dir = root_dir + "/Data/"


# %%
# --- create pandas dataframe from external file 

df_analysis = pd.read_csv(base_dir+'4-LAICPMS/Buranga_host_LAICPMS_oxides.csv',
                      encoding = "ANSI")

# --- print the columns of the imported dataframe 

df_analysis.columns 


# %%
# --- print analysis data for initial assessment

df_analysis = df_analysis[df_analysis['total'].notna()]

df_analysis


# %%
# --- separate data from metadata - only do calculation in data afterwards

df_analysis = df_analysis.dropna(subset=['Host']) 

df_data = df_analysis.drop(['sample', 'piece', 'field', 'analysis', 'Host', 'Info', 'total'], axis=1)

df_metadata = df_analysis[['sample', 'piece', 'field', 'analysis', 'Host', 'Info']]

print(len(df_data.columns))
df_data


# %%
# --- create a reference dataframe with data of oxides

df_reference = pd.read_csv(base_dir+"_Oxides_mass.csv",index_col=0)

reference_oxides = df_reference[df_data.columns]

reference_oxides


# %%
# --- create an array with molecular weights of oxides

molecular_weights = reference_oxides.iloc[0,:]

# --- create an array with the amount of oxygen in each of the oxides

oxygen_in_ox = reference_oxides.iloc[2,:]

# --- create an array with the ratio between cations and oxygen in oxides

cation_ratio_ox = reference_oxides.iloc[3].div(reference_oxides.iloc[2])

print(len(molecular_weights),
  len(oxygen_in_ox),
  len(cation_ratio_ox))

# %%
# ## 1. Divide oxide mass% input by oxide molecular weights
#    #### mass% / mol wt = mol amount 
#       Ex.: measured SiO2 in a biotite crystal = 52.8 mass%; molar weight of SiO2 = 60.0843
#       52.8 / 60.0843 = 0.878765


# --- new dataframe with measured mass% divided by oxides molecular weights

df_mol = df_data.div(molecular_weights, axis=1)
df_mol

# %%
# ## 2. Multiply mol amount by amount of oxygen in the oxide
#    #### mol amount * oxygen in oxide = oxygen number
#        Ex.: molar amount in our case = 0.878765; SiO2 has 2 oxygens in the oxide
#        0.878765 * 2 = 1.757531

# --- create a dataframe with the oxygen numbers

df_oxygen_N = df_mol.mul(oxygen_in_ox,axis=1)
df_oxygen_N

# %%
# ## 3. Sum up the calculated oxygen number for all oxides
# An exception here is the occurrence of F or Cl in the analyses. As these commonly occur as anions, they should be subtracted from the total sum. Both elements have -1 charge, which is half the charge of oxygen anions. Consequently, their calculated oxygen number should be multiplied by 0.5 before subtraction of the total.
#    #### Σ oxygen numbers of all cations - Σ (0.5 * oxygen numbers of F and Cl) = total oxygen
#        Ex.: oxygen number for SiO2 in our case = 1.757531; 
#        summing up with other oxides is = 4.58;
#        oxygen number for F and Cl is = 0.30 and 0.002 respectively 	
#        4.58 - [(0.5 * 0.3)+(0.5 * 0.002)] = 4.429

# --- add sum of anionic proportions to the last colum in dataframe

total_oxygen = df_oxygen_N.sum(axis=1)

# --- correct total oxygen sum by removing the influence of other anions (F,Cl,etc.)

#df_oxygen_N["Sum2"] = df_oxygen_N["SumO"]-(0.5*df_oxygen_N["Cl%"]+0.5*df_data_anion["F%"])

# print(total_oxygen.min())
print (np.where(total_oxygen == total_oxygen.min()))

total_oxygen

# %%
print(df_analysis['Host'].unique())


# %%
# --- create Series that will receive the values for oxygen in formula

O_in_formula = df_analysis['Host']
#O_in_formula = 2

# --- assign values of oxygen according to each mineral formula

O_in_formula = np.where(O_in_formula == 'bertossaite', 20, O_in_formula)
O_in_formula = np.where(O_in_formula == 'scorzalite', 9, O_in_formula)
O_in_formula = np.where(O_in_formula == 'trolleite', 15, O_in_formula)
O_in_formula = np.where(O_in_formula == 'mica', 11, O_in_formula)  
O_in_formula = np.where(O_in_formula == 'augelite', 5.5, O_in_formula)  
O_in_formula = np.where(O_in_formula == 'brazilianite', 10, O_in_formula)
O_in_formula = np.where(O_in_formula == 'apatite', 12.5, O_in_formula)  
O_in_formula = np.where(O_in_formula == 'rutile', 2, O_in_formula)
O_in_formula = np.where(O_in_formula == 'Coltan', 6, O_in_formula)
O_in_formula = np.where(O_in_formula == 'burangaite', 17, O_in_formula)
O_in_formula = np.where(O_in_formula == 'variscite+strengite', 4, O_in_formula)  
O_in_formula = np.where(O_in_formula == 'eosphorite', 5, O_in_formula)
O_in_formula = np.where(O_in_formula == 'rosemaryite', 12, O_in_formula)  
O_in_formula = np.where(O_in_formula == 'montebrasite', 4.5, O_in_formula)
O_in_formula = np.where(O_in_formula == 'wyllieite', 12, O_in_formula)  
O_in_formula = np.where(O_in_formula == 'samuelsonite', 41, O_in_formula)
O_in_formula = np.where(O_in_formula == 'lacroixite', 4.5, O_in_formula)
O_in_formula = np.where(O_in_formula == 'strengite', 4, O_in_formula)
O_in_formula = np.where(O_in_formula == 'wardite', 10, O_in_formula)  
O_in_formula = np.where(O_in_formula == 'albite', 8, O_in_formula)
O_in_formula = np.where(O_in_formula == 'zircon', 4, O_in_formula)  
O_in_formula = np.where(O_in_formula == 'K-feldspar', 8, O_in_formula)  
O_in_formula = np.where(O_in_formula == 'tourmaline', 11, O_in_formula)

O_in_formula = np.where(O_in_formula == 'quartz', 2, O_in_formula)
O_in_formula = np.where(O_in_formula == 'Fe-ox', 1, O_in_formula)
O_in_formula = np.where(O_in_formula == 'Mn-ox', 1, O_in_formula)
O_in_formula = np.where(O_in_formula == 'uranite', 1, O_in_formula)
O_in_formula = np.where(O_in_formula == 'baryte', 4, O_in_formula)


# O_in_formula[pd.to_numeric(O_in_formula['Mineral'], errors='coerce').notnull()]

# --- amount of oxygen in the formula unit - basis for formula calculation

O_in_formula


# %%
# ## 4. Divide the number of oxygens in the mineral formula unit by total oxygen of last step
# The oxygen in the formula unit can be retrieved from the ideal mineral formula. For anhydrous minerals, this value is directly the number of oxygens listed (e.g. 4 in the case of olivine, (Fe,Mg)SiO4). For hydrated phases each other anion (OH, F, or Cl) will count as half oxygen and the final value will be the number of oxygens plus half the number of OH,F,Cl in the formula (e.g. 11 for biotite K(Mg,Fe)3AlSi3O10(OH,F)2)
#    #### O in formula unit / total oxygen = oxygen normalization factor 
#        Ex.: total oxygen in our case = 4.429; biotite has an equivalent value of 11 oxygens in its formula (considering also the amount of OH and F in the ideal formula - see paragraph above for further clarification)
#        11 / 4.429 = 2.484

# --- Oxygen Normalization Factor = number of oxygen in the formula divided by sum of anions

ONF = (O_in_formula/total_oxygen)
ONF

# %%
# ## 5. Multiply the oxygen number from item 2 by the oxygen normalization factor 
#    #### oxygen number * ONF = anionic proportion
#        Ex.: oxygen for SiO2 in our case = 1.757531; ONF = 2.484
#        1.757531 * 2.484 = 4.365

# --- Anionic proportion = oxygen number multiplied by ONF
df_anionic = df_oxygen_N.mul(ONF,axis=0)

print(df_anionic.columns)
df_anionic


# %%
# ## 6. Multiply the anionic proportion by the ratio of cations and oxygen in the oxide 
#    #### anionic proportion * (cations in oxide / oxygen in oxide) = cationic proportion
#        Ex.: in our case anionic proportion = 4.365; and the ratio between Si and O in the oxide formula (i.e. SiO2) is 1/2 = 0.5
#        4.365 * 0.5 = 2.183

# --- Cationic proportion = anionic proportion multiplied by oxide cation ratio
df_cationic = df_anionic.mul(cation_ratio_ox)

# --- rename dataframe columns for simplification
        
df_cationic.columns = ['Li', 'B', 'Na', 'Mg', 'Al', 'Si', 'P', 'K', 'Ti', 
                       'Mn', 'Rb', 'Sr', 'Nb', 'Sn', 'Cs', 'Ba', 'Ta', 'W',
                       'Ca', 'Fe']

# --- combine calculated data with metadata
#df_cationic['Total'] = df_cationic.sum(axis=0,numeric_only=True)

df_cationic.insert(0,"sample",df_analysis["sample"])
df_cationic.insert(1,"piece",df_analysis["piece"])
df_cationic.insert(2,"field",df_analysis["field"])
df_cationic.insert(3,"analysis",df_analysis["analysis"])
df_cationic.insert(4,"Host",df_analysis["Host"])
df_cationic.insert(5,"Info",df_analysis["Info"])


df_cationic.to_csv(base_dir+'4-LAICPMS/Buranga_host_LAICPMS_apfu.csv', index=False)
df_cationic


# %%


df_data_cationic["Li"] = (number_oxygen/4) - (df_data_cationic["K"]+df_data_cationic["Na"]+df_data_cationic["Ca"])
df_data_cationic["OH"] = (number_oxygen/4) - (df_data_cationic["Cl"]+df_data_cationic["F"])
df_data_cationic


# %%




