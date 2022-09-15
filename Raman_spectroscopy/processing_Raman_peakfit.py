#!/usr/bin/env python
# coding: utf-8
# Author: Fernando Prado Araujo

# In[1]:


# ---------------- processing Raman files exported from Peakfit

    # - This code was developed in close collaboration with Valeria Fonseca Díaz and Marco Dalla Vecchia, and would not exist without their help!
    
# --- import modules

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

from module_Raman_peaks import spectrum_data, gauss_lor_sum

# --- set working directories
os.chdir("..")

root_dir = os.getcwd()
base_dir = root_dir + "/Data/Raman/"


# In[2]:


# --- get list of files

spectra_files = sorted(os.listdir(base_dir+"NTO-spectra"))

peak_files = sorted(os.listdir(base_dir+"NTO-peaks"))

print(spectra_files,
      "\n",len(spectra_files),
      "\n\n",
    peak_files,
      "\n",len(peak_files),)


# In[4]:


# --- select one file from the spectra dataset to work with

#ff = 1
ff = spectra_files.index('CGM_bu04-ore1_150 (500nm)_532nm_edge_10%_x50_vis_lwd_h30um_20sx2.dat')

print(peak_files[ff])
print(spectra_files[ff])

df_spectrum_peaks = spectrum_data(spectra_files[ff])

df_spectrum_peaks


# In[5]:


# --- make plot of single spectrum file

    # - set start and end for the plot x axis 

start = 100 
end = 1000

fig, ax = plt.subplots(figsize=(15,7.5))

    # --- RUTILE - add specific annotations for normal modes in the spectrum

# plt.annotate("$B_{1g}$",xy=(143, 0), xytext=(126, np.percentile(df_spectrum_peaks["Y Value"], 90)),
#               arrowprops=dict(arrowstyle="-",
#                               edgecolor = "black",
#                               linewidth=1,
#                               alpha=0.65), size = 16)

# plt.annotate("$E_{g}$",xy=(447, 0), 
#              xytext=(437, np.percentile(df_spectrum_peaks["Y Value"], 100)+np.percentile(df_spectrum_peaks["Y Value"], 80)),
#               arrowprops=dict(arrowstyle="-",
#                               edgecolor = "black",
#                               linewidth=1,
#                               alpha=0.65), size = 16)

# plt.annotate("$A_{1g}$",xy=(612, 0), 
#              xytext=(597, np.percentile(df_spectrum_peaks["Y Value"], 100)+np.percentile(df_spectrum_peaks["Y Value"], 80)),
#               arrowprops=dict(arrowstyle="-",
#                               edgecolor = "black",
#                               linewidth=1,
#                               alpha=0.65), size = 16)

# plt.annotate("$B_{2g}$",xy=(826, 0), xytext=(810, np.percentile(df_spectrum_peaks["Y Value"], 90)),
#               arrowprops=dict(arrowstyle="-",
#                               edgecolor = "black",
#                               linewidth=1,
#                               alpha=0.65), size = 16)

# plt.text(240, np.percentile(df_spectrum_peaks["Y Value"], 91), 
#             "Second order \nphonons",horizontalalignment='center',fontsize=16)

    # --- COLTAN - add specific annotations for normal modes in the spectrum

plt.annotate("$A_{1g}$",xy=(880, 0), 
             xytext=(864, np.percentile(df_spectrum_peaks["Y Value"], 100)+np.percentile(df_spectrum_peaks["Y Value"], 88)),
              arrowprops=dict(arrowstyle="-",
                              edgecolor = "black",
                              linewidth=1,
                              alpha=0.65), size = 16)

plt.annotate("$B_{2g}$",xy=(630, 0), 
             xytext=(614, np.percentile(df_spectrum_peaks["Y Value"], 98)),
              arrowprops=dict(arrowstyle="-",
                              edgecolor = "black",
                              linewidth=1,
                              alpha=0.65), size = 16)

plt.annotate("$A_{1g}$",xy=(532, 0), 
             xytext=(516, np.percentile(df_spectrum_peaks["Y Value"], 99)),
              arrowprops=dict(arrowstyle="-",
                              edgecolor = "black",
                              linewidth=1,
                              alpha=0.65), size = 16)

plt.annotate("$A_{1g}$",xy=(400, 0), 
             xytext=(384, np.percentile(df_spectrum_peaks["Y Value"], 99)),
              arrowprops=dict(arrowstyle="-",
                              edgecolor = "black",
                              linewidth=1,
                              alpha=0.65), size = 16)

plt.annotate("$A_{1g}$",xy=(275, 0), 
             xytext=(259, np.percentile(df_spectrum_peaks["Y Value"], 99)),
              arrowprops=dict(arrowstyle="-",
                              edgecolor = "black",
                              linewidth=1,
                              alpha=0.65), size = 16)

plt.annotate("$B_{1g}$",xy=(129, 0), 
             xytext=(113, np.percentile(df_spectrum_peaks["Y Value"], 98)),
              arrowprops=dict(arrowstyle="-",
                              edgecolor = "black",
                              linewidth=1,
                              alpha=0.65), size = 16)

plt.annotate("$A_{1g}$",xy=(140, 0), 
             xytext=(124, np.percentile(df_spectrum_peaks["Y Value"], 99)),
              arrowprops=dict(arrowstyle="-",
                              edgecolor = "black",
                              linewidth=1,
                              alpha=0.65), size = 16)

plt.annotate("$A_{1g}$",xy=(210, 0), 
             xytext=(194, np.percentile(df_spectrum_peaks["Y Value"], 99)),
              arrowprops=dict(arrowstyle="-",
                              edgecolor = "black",
                              linewidth=1,
                              alpha=0.65), size = 16)

plt.annotate("$B_{2g}$",xy=(785, 0), 
             xytext=(769, np.percentile(df_spectrum_peaks["Y Value"], 99)),
              arrowprops=dict(arrowstyle="-",
                              edgecolor = "black",
                              linewidth=1,
                              alpha=0.65), size = 16)

plt.annotate("$B_{2g}$",xy=(839, 0), 
             xytext=(823, np.percentile(df_spectrum_peaks["Y Value"], 98)),
              arrowprops=dict(arrowstyle="-",
                              edgecolor = "black",
                              linewidth=1,
                              alpha=0.65), size = 16)

plt.annotate("$B_{1g}$",xy=(170, 0), 
             xytext=(154, np.percentile(df_spectrum_peaks["Y Value"], 98)),
              arrowprops=dict(arrowstyle="-",
                              edgecolor = "black",
                              linewidth=1,
                              alpha=0.65), size = 16)

plt.annotate("$B_{2g}$",xy=(247, 0), 
             xytext=(231, np.percentile(df_spectrum_peaks["Y Value"], 98)),
              arrowprops=dict(arrowstyle="-",
                              edgecolor = "black",
                              linewidth=1,
                              alpha=0.65), size = 16)

plt.annotate("$B_{1g}$",xy=(308, 0), 
             xytext=(292, np.percentile(df_spectrum_peaks["Y Value"], 98)),
              arrowprops=dict(arrowstyle="-",
                              edgecolor = "black",
                              linewidth=1,
                              alpha=0.65), size = 16)

plt.annotate("$B_{3g}$",xy=(498, 0), 
             xytext=(482, np.percentile(df_spectrum_peaks["Y Value"], 98)),
              arrowprops=dict(arrowstyle="-",
                              edgecolor = "black",
                              linewidth=1,
                              alpha=0.65), size = 16)

    # - plot the raw spectrum line (measured data) and the modelled spectrum (after peak fitting)

plt.plot(df_spectrum_peaks["X Value"],df_spectrum_peaks["Y Predict"],'b-', label='model', alpha=0.4)
plt.plot(df_spectrum_peaks["X Value"],df_spectrum_peaks["Y Value"],'k.', label='data', alpha=0.6)


    # - plot the modelled center value (Raman shift) of peaks above the curve.
            # plotting the FWHM is also possible by using the commented section   
    
for ii in range(df_spectrum_peaks.shape[0]):
    if start < df_spectrum_peaks.iloc[ii]["center"] < end:
        text = str(int(df_spectrum_peaks.iloc[ii]["center"]))
        plt.text(df_spectrum_peaks.iloc[ii]["X Value"]-20,
                 df_spectrum_peaks.iloc[ii]["Y Predict"]+np.percentile(df_spectrum_peaks["Y Predict"], 80),
                 text,fontsize=18)
#         if np.percentile(df_spectrum_peaks.iloc[ii]["height"], 50) > df_spectrum_peaks.iloc[ii]["center"]:
#             text = str(int(df_spectrum_peaks.iloc[ii]["FWHM"]))
#             plt.text(df_spectrum_peaks.iloc[ii]["X Value"]-15,df_spectrum_peaks.iloc[ii]["Y Predict"]+10,text,fontsize=18)
   
    # - get positions of peak data inside the overall file
    
counter = 0
peak_lines_loc = []

for value in df_spectrum_peaks['center']:
    if value > 0:
        peak_lines_loc.append(counter)
    counter += 1

    
    # --- add modelled peaks to plot
    
x = np.arange(start,end)

for peak in peak_lines_loc:
    a_0 = df_spectrum_peaks['height'][peak]
    a_1 = df_spectrum_peaks['a1'][peak]
    a_2 = df_spectrum_peaks['a2'][peak]
    a_3 = df_spectrum_peaks['a3'][peak]
    y = gauss_lor_sum(x, a_0, a_1, a_2, a_3)
    plt.plot(x,y,ls='--')
    ax.fill_between(x, y.min(),y, alpha=0.3)
        
        
    # --- adjust plot parameters

plt.legend(loc='best',fontsize=18)    
    
plt.xlabel("Raman shift ($cm^{-1}$)",fontsize=20)
plt.xlim([start,end])
plt.ylabel("Intensity ($counts$)",fontsize=20)
plt.ylim([-5,np.percentile(df_spectrum_peaks["Y Value"], 100)+np.percentile(df_spectrum_peaks["Y Value"], 94)])

ax.xaxis.set_minor_locator(MultipleLocator(100))
ax.yaxis.set_minor_locator(AutoMinorLocator())

ax.tick_params(axis='both', which='major',labelsize=20,direction='in',width=1.5,length=8)
ax.tick_params(axis='both', which='minor',direction='in',width=1,length=4)

plt.title(df_spectrum_peaks.iloc[0]["file"],horizontalalignment='center',fontsize=22)

# plt.text(start+150, np.percentile(df_spectrum_peaks["Y Value"], 100), 
#             "$R^2$="+str(round(float(df_spectrum_peaks.iloc[0]["R2"]),4)),fontsize=20)


        
# plt.savefig('C:/Users/u0125722/Documents/Python_Scripts/output/figures/'+df_spectrum_peaks.iloc[0]["file"]+'_peakfit.pdf')
plt.show()


# In[ ]:


df_spectra = pd.DataFrame()

for ii in range(len(spectra_files)):
    print(spectra_files[ii])
#     print(ii)
    df_a = spectrum_data(spectra_files[ii])
    df_a = df_a.drop(['Residual%','90% Confidence', 'Limits', '90% Prediction', 'Limits2', 
                      'Weights','a0', 'a1', 'a2', 'a3'],axis=1)
#     print(a.columns)
#     df_a
    df_spectra = pd.concat([df_spectra,df_a], axis=0, sort=False)
    
    #--- separate file information into relavant acquisition parameters
    
df_spectra[['mineral','sample','grating','laser','edge',
             'filter','objective','vis','lwd','pinhole',
             'acquisition']]=df_spectra["file"].str.split(pat='_', expand=True)

df_spectra = df_spectra.drop(['edge','vis','lwd'], axis=1)
    
    
    
df_spectra.to_csv('NTO-Raman_data.csv',index = True)
df_spectra


# In[ ]:


# --- make plot of multiple spectra
for ii in range(len(spectra_files)):
    print(spectra_files[ii])
    spectrum_data(spectra_files[ii])

    df_spectrum_peaks = spectrum_data(spectra_files[ii])
    
        # - set start and end for the plot x axis 

    start = 100 
    end = 1000

    fig, ax = plt.subplots(figsize=(15,7.5))

        # --- RUTILE - add specific annotations for normal modes in the spectrum

    # plt.annotate("$B_{1g}$",xy=(143, 0), xytext=(126, np.percentile(df_spectrum_peaks["Y Value"], 90)),
    #               arrowprops=dict(arrowstyle="-",
    #                               edgecolor = "black",
    #                               linewidth=1,
    #                               alpha=0.65), size = 16)

    # plt.annotate("$E_{g}$",xy=(447, 0), 
    #              xytext=(437, np.percentile(df_spectrum_peaks["Y Value"], 100)+np.percentile(df_spectrum_peaks["Y Value"], 80)),
    #               arrowprops=dict(arrowstyle="-",
    #                               edgecolor = "black",
    #                               linewidth=1,
    #                               alpha=0.65), size = 16)

    # plt.annotate("$A_{1g}$",xy=(612, 0), 
    #              xytext=(597, np.percentile(df_spectrum_peaks["Y Value"], 100)+np.percentile(df_spectrum_peaks["Y Value"], 80)),
    #               arrowprops=dict(arrowstyle="-",
    #                               edgecolor = "black",
    #                               linewidth=1,
    #                               alpha=0.65), size = 16)

    # plt.annotate("$B_{2g}$",xy=(826, 0), xytext=(810, np.percentile(df_spectrum_peaks["Y Value"], 90)),
    #               arrowprops=dict(arrowstyle="-",
    #                               edgecolor = "black",
    #                               linewidth=1,
    #                               alpha=0.65), size = 16)

    # plt.text(240, np.percentile(df_spectrum_peaks["Y Value"], 91), 
    #             "Second order \nphonons",horizontalalignment='center',fontsize=16)

        # --- COLTAN - add specific annotations for normal modes in the spectrum

    plt.annotate("$A_{1g}$",xy=(880, 0), 
                 xytext=(864, np.percentile(df_spectrum_peaks["Y Value"], 100)+np.percentile(df_spectrum_peaks["Y Value"], 88)),
                  arrowprops=dict(arrowstyle="-",
                                  edgecolor = "black",
                                  linewidth=1,
                                  alpha=0.65), size = 16)

    plt.annotate("$B_{2g}$",xy=(630, 0), 
                 xytext=(614, np.percentile(df_spectrum_peaks["Y Value"], 98)),
                  arrowprops=dict(arrowstyle="-",
                                  edgecolor = "black",
                                  linewidth=1,
                                  alpha=0.65), size = 16)

    plt.annotate("$A_{1g}$",xy=(532, 0), 
                 xytext=(516, np.percentile(df_spectrum_peaks["Y Value"], 99)),
                  arrowprops=dict(arrowstyle="-",
                                  edgecolor = "black",
                                  linewidth=1,
                                  alpha=0.65), size = 16)

    plt.annotate("$A_{1g}$",xy=(400, 0), 
                 xytext=(384, np.percentile(df_spectrum_peaks["Y Value"], 99)),
                  arrowprops=dict(arrowstyle="-",
                                  edgecolor = "black",
                                  linewidth=1,
                                  alpha=0.65), size = 16)

    plt.annotate("$A_{1g}$",xy=(275, 0), 
                 xytext=(259, np.percentile(df_spectrum_peaks["Y Value"], 99)),
                  arrowprops=dict(arrowstyle="-",
                                  edgecolor = "black",
                                  linewidth=1,
                                  alpha=0.65), size = 16)

    plt.annotate("$B_{1g}$",xy=(129, 0), 
                 xytext=(113, np.percentile(df_spectrum_peaks["Y Value"], 98)),
                  arrowprops=dict(arrowstyle="-",
                                  edgecolor = "black",
                                  linewidth=1,
                                  alpha=0.65), size = 16)

    plt.annotate("$A_{1g}$",xy=(140, 0), 
                 xytext=(124, np.percentile(df_spectrum_peaks["Y Value"], 99)),
                  arrowprops=dict(arrowstyle="-",
                                  edgecolor = "black",
                                  linewidth=1,
                                  alpha=0.65), size = 16)

    plt.annotate("$A_{1g}$",xy=(210, 0), 
                 xytext=(194, np.percentile(df_spectrum_peaks["Y Value"], 99)),
                  arrowprops=dict(arrowstyle="-",
                                  edgecolor = "black",
                                  linewidth=1,
                                  alpha=0.65), size = 16)

    plt.annotate("$B_{2g}$",xy=(785, 0), 
                 xytext=(769, np.percentile(df_spectrum_peaks["Y Value"], 99)),
                  arrowprops=dict(arrowstyle="-",
                                  edgecolor = "black",
                                  linewidth=1,
                                  alpha=0.65), size = 16)

    plt.annotate("$B_{2g}$",xy=(839, 0), 
                 xytext=(823, np.percentile(df_spectrum_peaks["Y Value"], 98)),
                  arrowprops=dict(arrowstyle="-",
                                  edgecolor = "black",
                                  linewidth=1,
                                  alpha=0.65), size = 16)

    plt.annotate("$B_{1g}$",xy=(170, 0), 
                 xytext=(154, np.percentile(df_spectrum_peaks["Y Value"], 98)),
                  arrowprops=dict(arrowstyle="-",
                                  edgecolor = "black",
                                  linewidth=1,
                                  alpha=0.65), size = 16)

    plt.annotate("$B_{2g}$",xy=(247, 0), 
                 xytext=(231, np.percentile(df_spectrum_peaks["Y Value"], 98)),
                  arrowprops=dict(arrowstyle="-",
                                  edgecolor = "black",
                                  linewidth=1,
                                  alpha=0.65), size = 16)

    plt.annotate("$B_{1g}$",xy=(308, 0), 
                 xytext=(292, np.percentile(df_spectrum_peaks["Y Value"], 98)),
                  arrowprops=dict(arrowstyle="-",
                                  edgecolor = "black",
                                  linewidth=1,
                                  alpha=0.65), size = 16)

    plt.annotate("$B_{3g}$",xy=(498, 0), 
                 xytext=(482, np.percentile(df_spectrum_peaks["Y Value"], 98)),
                  arrowprops=dict(arrowstyle="-",
                                  edgecolor = "black",
                                  linewidth=1,
                                  alpha=0.65), size = 16)

        # - plot the raw spectrum line (measured data) and the modelled spectrum (after peak fitting)

    plt.plot(df_spectrum_peaks["X Value"],df_spectrum_peaks["Y Predict"],'b-', label='model', alpha=0.4)
    plt.plot(df_spectrum_peaks["X Value"],df_spectrum_peaks["Y Value"],'k.', label='data', alpha=0.6)


        # - plot the modelled center value (Raman shift) of peaks above the curve.
                # plotting the FWHM is also possible by using the commented section   

    for ii in range(df_spectrum_peaks.shape[0]):
        if start < df_spectrum_peaks.iloc[ii]["center"] < end:
            text = str(int(df_spectrum_peaks.iloc[ii]["center"]))
            plt.text(df_spectrum_peaks.iloc[ii]["X Value"]-20,
                     df_spectrum_peaks.iloc[ii]["Y Predict"]+np.percentile(df_spectrum_peaks["Y Predict"], 80),
                     text,fontsize=18)
    #         if np.percentile(df_spectrum_peaks.iloc[ii]["height"], 50) > df_spectrum_peaks.iloc[ii]["center"]:
    #             text = str(int(df_spectrum_peaks.iloc[ii]["FWHM"]))
    #             plt.text(df_spectrum_peaks.iloc[ii]["X Value"]-15,df_spectrum_peaks.iloc[ii]["Y Predict"]+10,text,fontsize=18)

        # - get positions of peak data inside the overall file

    counter = 0
    peak_lines_loc = []

    for value in df_spectrum_peaks['center']:
        if value > 0:
            peak_lines_loc.append(counter)
        counter += 1


        # --- add modelled peaks to plot

    x = np.arange(start,end)

    for peak in peak_lines_loc:
        a_0 = df_spectrum_peaks['height'][peak]
        a_1 = df_spectrum_peaks['a1'][peak]
        a_2 = df_spectrum_peaks['a2'][peak]
        a_3 = df_spectrum_peaks['a3'][peak]
        y = gauss_lor_sum(x, a_0, a_1, a_2, a_3)
        plt.plot(x,y,ls='--')
        ax.fill_between(x, y.min(),y, alpha=0.3)


        # --- adjust plot parameters

    plt.legend(loc='best',fontsize=18)    

    plt.xlabel("Raman shift ($cm^{-1}$)",fontsize=20)
    plt.xlim([start,end])
    plt.ylabel("Intensity ($counts$)",fontsize=20)
    plt.ylim([-5,np.percentile(df_spectrum_peaks["Y Value"], 100)+np.percentile(df_spectrum_peaks["Y Value"], 94)])

    ax.xaxis.set_minor_locator(MultipleLocator(100))
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax.tick_params(axis='both', which='major',labelsize=20,direction='in',width=1.5,length=8)
    ax.tick_params(axis='both', which='minor',direction='in',width=1,length=4)

    plt.title(df_spectrum_peaks.iloc[0]["file"],horizontalalignment='center',fontsize=22)

    # plt.text(start+150, np.percentile(df_spectrum_peaks["Y Value"], 100), 
    #             "$R^2$="+str(round(float(df_spectrum_peaks.iloc[0]["R2"]),4)),fontsize=20)



    # plt.savefig('C:/Users/u0125722/Documents/Python_Scripts/output/figures/'+df_spectrum_peaks.iloc[0]["file"]+'_peakfit.pdf')
    plt.show()


# In[ ]:


df_spectra = pd.DataFrame()

raw_files = sorted(os.listdir(base_dir+"raw"))

print(raw_files)


for ii in range(len(raw_files)):
#     print(spectra_files[ii])
#     print(ii)
    df_a = pd.read_csv(base_dir+"raw/"+raw_files[ii], comment='#', sep='\t',
                       names=['X Value', 'Y Value'])
    
    df_a['file']=raw_files[ii].replace(".dat", "").replace(".txt", "")

    df_spectra = pd.concat([df_spectra,df_a], axis=0, sort=False)
    
    
# df_spectra.to_csv('Supplementary_material-S3-Raw_Raman_spectra.csv',index = True)
df_spectra


# In[ ]:




