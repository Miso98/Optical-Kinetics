# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import scipy.optimize as opt
from scipy.optimize import curve_fit
matplotlib.rcParams.update({'font.size': 16})
from sklearn.metrics import r2_score 
from scipy.stats import linregress

#This value is pulled from literature and used as a conversion factor
absorbtivity = 20000 * 0.24 

#read lockin_data
def read_lockin_data(file, delimiter=";", channels=1):
    if channels == 1:
        skiprows = 5
    elif channels == 2:
        skiprows = 7
    df = pd.read_csv(file, delimiter=delimiter, skiprows=skiprows).to_numpy()
    if channels > 1:
        channel2_index = np.where(abs(df[1:,0]-df[:-1,0]) > 0.5*np.max(abs(df[:,0])))[0]
        channel2 = df[channel2_index[0]+1:,1]
        if len(channel2) != len(df[:channel2_index[0]+1,0]):
            channel2 = channel2[:min(len(channel2), len(df[:channel2_index[0]+1,0]))]
            df = df[:min(len(channel2), len(df[:channel2_index[0]+1,0])), :]
        df = np.concatenate((df[:channel2_index[0]+1,:], channel2.reshape(len(channel2),1)), axis=1)
    return df

#takes voltage values read from lockin and converts to a transmission value
def vol2transmission(df, zero_val=None):
    
    df2 = df.copy()
    if zero_val is None:
        # df2[:,1] = df2[:,1] / np.average(df2[:,1])
        zero_val = np.average(df2, axis=0)[1:]
    if np.size(df2,axis=1) > 2:
        df2[:,1:] = np.einsum('ij,j->ij', df2[:,1:], zero_val**-1)
    else:
        df2[:,1] = df2[:,1] * zero_val**-1
    return df2

#adjusts the timeset of the lockin
def adjust_time(df,cutoff=None, ms=False, minutes=True):
    if cutoff is not None:
        df = df[df[:,1]<cutoff,:]
    df[:,0] = df[:,0] - df[0,0]
    if ms:
        df[:,0] = df[:,0] / 1e3
    if minutes == True:
        df[:,0] = df[:,0] / 60
    return df
    
def plot(df, label=""):
    plt.plot(df[:,0], df[:,1:], label=label)

#possible reaction rates (returns transmission)
def logistic_rxn(t, Cmax, k, t0):
    C = Cmax * (1 + np.exp(-1*k*(t - t0)))**-1
    T = 10**(-1*absorbtivity*C)
    return T
def exponential_rxn(t, Cmax, k, t0):
    C = Cmax * (1 - np.exp(-1*k*(t-t0)))
    T = 10**(-1*absorbtivity*C)
    return T
def secondorder_rxn(t, Cmax, k, alpha, t0):
    C = Cmax*(1 - (-1*(t-t0)*k*(1 - alpha) + 1)**(1/(1-alpha)))
    T = 10**(-1*absorbtivity*C)
    return T
def double_exp_rxn(t, Cmax, k1, k2, t0):
    C = Cmax * (1 - 1*(k2*np.exp(-1*k1*(t-t0)) -k1*np.exp(-1*k2*(t-t0))) / (k2-k1))
    T = 10**(-1*absorbtivity*C)
    return T
# def triple_exp_rxn(t, Cmax, k1, k2, k3, t0):
#     #not currently fitting the absorbtivity factor
#     agg_factor = 10.0
       
#     C3 = Cmax*k1*k2*( np.exp(-1*k1*(t-t0)) / ( (k1-k2)*(k1-k3) )
#                      + np.exp(-1*k2*(t-t0)) / ( (k2-k1)*(k2-k3) )
#                      + np.exp(-1*k3*(t-t0)) / ( (k3-k1)*(k3-k2) )
#                      )
#     C4 = Cmax + Cmax / ( (k1-k2)*(k2-k3)*(k1-k3) ) * (
#         k2*k3*(k3-k2)*np.exp(-1*k1*(t-t0))
#         + k1*k3*(k1-k3)*np.exp(-1*k2*(t-t0))
#         + k1*k2*(k2-k1)*np.exp(-1*k3*(t-t0)))
    
#     T = 10**( -1*absorbtivity*(C3 + agg_factor*C4) )
#     return T

cutoff = 10

# folder = "C:/Users/Mitch So/OneDrive/Documents/Python files/"
folder = "C:/Users/mso/Leiden Tech Dropbox/Projects - Shared/2008 - Phosphator Phase II/8-Data/Lock-in Optics Testing Data/Kinetics Data/20240424 Ladder/"

#input files here
files = ["20240424-Ladder-0uM.txt",
         "20240424-Ladder-1uM.txt",
         "20240424-Ladder-5uM.txt",
         "20240424-Ladder-20uM-after-3min.txt",
         "20240424-Ladder-150uM.txt",
         "20240424-Ladder-250uM.txt"
         ]
#input baseline file names here
baselines = ["20240424-Ladder-Baseline-1.txt",
             "20240424-Ladder-Baseline-2.txt",
             "20240424-Ladder-Baseline-3.txt",
             "20240424-Ladder-Baseline-4.txt",
             "20240424-Ladder-Baseline-5.txt",
             "20240424-Ladder-Baseline-6.txt"
         ]
#label names array here
labels = ["0uM",
          "1uM",
          "5uM",
          "20uM",
          "150uM",
          "250uM"
          ]

pin0=True

#ladder concentrations
exp_concs = [0, 1, 5, 20, 150, 250]
max_absorbance_values = []

#initializing one figure two subplots
#fig, axs = plt.subplots(nrows = 1, ncols = 6, figsize=(15,6))

for i in range(len(exp_concs)):
    baseline = read_lockin_data(folder + baselines[i], delimiter=";", channels=2)
    experiment = read_lockin_data(folder + files[i], delimiter=";", channels=2)
    
    #this is hard coded for now to represent that the cutoff for transmission measurement is .75 when the graph actually reaches that point
    #this is not the case for sub 20uM experiments 
    
    if i < 4:
        cutoff = 10
    else:
        cutoff = .75
   
    experiment = vol2transmission(experiment, zero_val=np.array([np.average(baseline[:, 1]), np.average(experiment[:, 2])]))
    experiment = adjust_time(experiment[:, :2], cutoff=cutoff)
    transmission_values = experiment[:,1]   
    time_values = experiment[:,0]
    
    #absorbance from transmission A = -log(T) 
    absorbance_values = -np.log10(transmission_values)
    max_absorbance = np.max(absorbance_values)
    max_absorbance_values.append(max_absorbance)
    
    
    #absorbance values are more or less irrelevant against time, looking for asymptotic values for linearity
    # axs[i].plot(experiment[:,0], experiment[:,1], label=f'{exp_concs[i]}+uM')
    half = int(len(experiment)/1)
    
    # half = np.argmax(experiment[:,0] > 5) #3 minutes
    if pin0:
        temp = experiment[0,1] * np.ones((int(half/4),2))
        temp[:,0] = 0
        experiment =np.concatenate((temp, experiment[:,0:2]))
        
    f = double_exp_rxn
    popt, pcov = opt.curve_fit(f, 
                                experiment[:half,0], experiment[:half,1],
                                bounds=([0, 0,1e-3,-1e1],[1e-3, 1e1, 1e1, 1e1]),
                                x_scale=[1000, 1, 1,  1]
                                )
    
    #print constants of kinetics
  
    print(popt)
    y_predicted = (f(experiment[:,0], *popt))
    y_observed = experiment[:,1]
    r2_calc = r2_score(y_observed, y_predicted)
    #  Set labels and title for subplots
    # axs[i].plot(experiment[:,0], experiment[:,1], label=f'{exp_concs[i]}uM')
    # axs[i].plot(experiment[:,0], (f(experiment[:,0], *popt)), "--", label=f'{exp_concs[i]}uM')
    # axs[i].set_xlabel("time (s)")
    # axs[i].set_ylabel("Transmission")
    # axs[i].set_ylim([-0.01,1.01])
    # axs[i].set_title(f'{exp_concs[i]}uM')
    # axs[i].legend()
    
    plt.plot(experiment[:,0], experiment[:,1], label=f'{exp_concs[i]}uM')
    plt.plot(experiment[:,0], (f(experiment[:,0], *popt)), "--", label=f'{exp_concs[i]}uM curve fit')
    plt.xlabel("time (min)")
    plt.ylabel("Transmission")
    plt.ylim([-0.01,1.01])
    plt.title(f'{exp_concs[i]}uM')
    plt.legend(loc = 5)
    
    
    curve_coeff = [0] * 4
    
    #print out equations on each graph truncating/rounding at 3 sig figs, coefficient 1 is displayed in uM
    for i in range(len(popt)):
        curve_coeff[i] = popt[i]
        if i ==0:
            curve_coeff[i] = round(curve_coeff[i]* 1000000, 3)
        else:
            curve_coeff[i] = round(popt[i], 3)
        
    #absorbance from transmission A = -log(T) 
    
    
   
    equation_text = f'{curve_coeff} , r^2 = {r2_calc}'
    # axs[i].text(0.1, 0.1, equation_text, transform=axs[i].transAxes, fontsize=10, verticalalignment='top')
    plt.text(0.1, 0.1, equation_text, fontsize=10, verticalalignment='top')
    plt.show()
    
   
# fig.tight_layout()    
# fig.show()


# Perform linear regression
slope, intercept, r_value, p_value, std_err = linregress(exp_concs, max_absorbance_values)

# Calculate regression line
regression_line = slope * np.array(exp_concs) + intercept
plt.figure(6)
plt.scatter(exp_concs, max_absorbance_values,  label='Data Points')   
plt.plot(exp_concs, regression_line, linestyle='-', color='r', label='Regression Line')

plt.xlabel("Concentration (uM)")
plt.ylabel("Final Absorbance")
plt.title(f'Absorbance vs Concentration(uM)')
plt.grid(True)
plt.legend(loc = 5)

equation_text = f'y = {slope:.3f}x + {intercept:.3f}\n$R^2$ = {r_value**2:.3f}'
plt.text(0.05, 0.95, equation_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

plt.show()
#does not automatically plot graph needs this for powershell and embedded vs terminal

#This plots the absorbance vs time graphs use for ladder experiments to check linearity of resultants   
