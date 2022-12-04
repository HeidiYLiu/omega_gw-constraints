%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from classy import Class
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
import math
import pandas as pd
from sympy import symbols, solve
import scipy.linalg as la

from scipy.interpolate import pchip_interpolate

from matplotlib.pyplot import cm

from scipy.special import erfc
from math import pi
from numpy import log
from numpy import sqrt
from numpy import e

plt.rcParams["figure.figsize"] = [14, 10]
cmap = cm.get_cmap('plasma', 17)


# Common Parameters
A_s = 2.100549e-09
A_l = 0.05
f_sky = 3/100
mi = 11
ma = 1000
l = np.linspace(2, 1000, 999)

k_center = np.geomspace(1e-6, 1e2, 17)
delta = k_center[1]/k_center[0]
sigma = 0.1

deltas = 2/100


# Function defining the derivative "bumps"
def pri_dev(k, k_center, delta, r_p, r_o):
    sigma = 0.1
    A_s = 2.100549e-09
#     delta = k_center[1]/k_center[0]
 
    # Setting Ranges
    k_left = k_center / sqrt(delta)
    k_right = k_center * sqrt(delta)

    left_edge = erfc((np.log(k_left) - np.log(k)) / (sigma * sqrt(2)))
    right_edge = erfc((np.log(k_right) - np.log(k)) / (sigma * sqrt(2)))

    # Subtraction
    dev_bump = left_edge - right_edge

    # Varying peak
    ro = A_s * r_o
    rp = A_s * r_p
    c_s = (rp-ro)/2

    peak = dev_bump*c_s

    step = peak + ro

    return step


# Function Generating B-mode Power Spectrum

def Cl_bb_r(r, l):
    # Importing new Data
    Pk_example[0] = k
    Pk_example[2] = A_s * r
    new_file = '/Users/yinqiuliu/class/external/external_Pk/newdata.dat'
    np.savetxt(new_file, Pk_example.transpose())

    # Cosmological parameters and other CLASS parameters
    common_settings = {# Set primordial pk to external files
                        'Pk_ini_type': 'external_Pk',
                        'command': 'cat external/external_Pk/newdata.dat',
                        'modes':'t',
                       # output and precision parameters
                        'output':'pCl',
                        'l_max_tensors':l}
    #
    M = Class()
    M.set(common_settings)
    M.compute()

    cl = M.raw_cl(l)
    ell = cl['ell']
    # Converting y axis to dimensionless total [l(l+1)/2pi] C_l's
    factor = (ell*(ell+1))/(2*pi)
    bb = (factor * cl['bb'])
#     plt.loglog(ell[2:1001], cl_bb[2:1001])

    M.empty()
    return bb
    
    
# Derivatives Function
  
def derivative(b_plus, b_minus, delta_rp):
df = (b_plus - b_minus) / (2 * delta_rp)
return df
    
def bins_derivative(k_center, r, l):
    delta = k_center[1]/k_center[0]
    dfs_sp = []
    for i in range(17):
        dfs = []
        for sign in [1, -1]:
            pks = pri_dev(k, k_center[i], delta, r*(1+sign*deltas), r)

            # Importing new Data
            Pk_example[2] = pks
            new_file = '/Users/yinqiuliu/class/external/external_Pk/newdata.dat'
            np.savetxt(new_file, Pk_example.transpose())

            # Cosmological parameters and other CLASS parameters
            common_settings = {# Set primordial pk to external files
                                'Pk_ini_type': 'external_Pk',
                                'command': 'cat external/external_Pk/newdata.dat',
                                'modes':'t',
                               # output and precision parameters
                                'output':'pCl',
                                'l_max_tensors':l,
            }
            #
            M = Class()
            M.set(common_settings)
            M.compute()

            cl = M.raw_cl(l)
            ell = cl['ell']
            # Converting y axis to dimensionless total [l(l+1)/2pi] C_l's
            factor = (ell*(ell+1))/(2*pi)
            cl_bb = (factor * cl['bb'])

            dfs.append(cl_bb)
            # Append

            M.empty()

        dfs_sp.append(derivative(dfs[0], dfs[1], r*deltas))
    return(dfs_sp)


# Forecasring the Fisher Matrix

# mi and ma is the range of multipole (l)

def fisher_forecast(Cl_bb, df, f_sky, mi, ma, n_params):

    n_ell = (((mv_comb[mi:ma]/(2.726**2))*10**-12)*(el[mi:ma]*(el[mi:ma]+1)))/(2*np.pi)
    
    F = np.zeros((n_params, n_params))
    for i in range(n_params):
        for j in range(n_params):
            Fij_ell = (((2*l[mi-2:ma-2] + 1)/2) * ((Cl_bb[mi+1:ma+1] + n_ell)**(-2)) * ((df[i])[mi+1:ma+1]*(df[j])[mi+1:ma+1]))*f_sky
            F[i, j] = np.sum(Fij_ell)
    cov = la.inv(F)
    sigmas = np.diagonal(cov)
    
    return sigmas
    
    
# gravitational wave

def gw_trans(k):
    W_r_h2 = 4.2e-5
    k_eq_inv_Mpc = 1e-2
    return (3/128) * W_r_h2 * (1/2 * (k_eq_inv_Mpc / k)**2 + 16/9)
    
    
    
# generating lensing

def Cl_bb_lensed(r, l):
    # Importing new Data
    Pk_example[0] = k
    Pk_example[2] = A_s * r
    new_file = '/Users/yinqiuliu/class/external/external_Pk/newdata.dat'
    np.savetxt(new_file, Pk_example.transpose())

    # Cosmological parameters and other CLASS parameters
    common_settings = {# Set primordial pk to external files
                        'Pk_ini_type': 'external_Pk',
                        'command': 'cat external/external_Pk/newdata.dat',
                        'modes':'s',
                       # output and precision parameters
                        'output':'tCl, pCl, lCl',
                        'lensing':'yes',
                        'l_max_scalars':l}
    #
    M = Class()
    M.set(common_settings)
    M.compute()

    cl_lensed = M.lensed_cl(l)
    ell = cl_lensed['ell']
    # Converting y axis to dimensionless total [l(l+1)/2pi] C_l's
    factor = (ell*(ell+1))/(2*pi)
    bb = (factor * cl_lensed['bb'])
#     plt.loglog(ell[2:1001], cl_bb[2:1001])

    M.empty()
    return bb
    
    
# lensing forecast
def lensed_forecast(Cl_r, Cl_lensed, A_l, dfs, f_sky, mi, ma, n_params):

    Cl_Al = Cl_lensed * A_l
    
    
    n_ell = (((mv_comb[mi:ma]/(2.726**2))*10**-12)*(el[mi:ma]*(el[mi:ma]+1)))/(2*np.pi)
    
    F = np.zeros((n_params, n_params))
    for i in range(n_params):
        for j in range(n_params):
            Cl_total = Cl_r[mi+1:ma+1] + Cl_Al[mi+1:ma+1]
            Fij_ell = ((2*l[mi-2:ma-2] + 1)/2 * ((Cl_total + n_ell)**-2) * ((dfs[i])[mi+1:ma+1]*(dfs[j])[mi+1:ma+1]))*f_sky
            F[i, j] = np.sum(Fij_ell)
            print(F)
    cov = la.inv(F)
    print(cov)
    sigmas = np.diagonal(cov)**0.5
    
    sig_r = sigmas[0]
    sigma_al = sigmas[1]
    return sig_r, sigma_al
    
# lensing r forecast

def r_forecast_lensing(Cl_r, Cl_lensed, A_l, dfs, f_sky, mi, ma, n_params):

    n_ell = (((mv_comb[mi:ma]/(2.726**2))*10**-12)*(el[mi:ma]*(el[mi:ma]+1)))/(2*np.pi)

    Cl_Al = Cl_lensed * A_l
    Cl_total = Cl_r[mi+1:ma+1] + Cl_Al[mi+1:ma+1]

    F = np.zeros((n_params, n_params))
    for i in range(n_params - 1):
        for j in range(n_params - 1):
            Fij_ell = (((2*l[mi-2:ma-2] + 1)/2) * ((Cl_total + n_ell)**(-2)) * ((dfs[i])[mi+1:ma+1]*(dfs[j])[mi+1:ma+1]))*f_sky
            F[i, j] = np.sum(Fij_ell)
    cov = la.inv(F)
    sigmas = np.diagonal(cov)**0.5
    
    return sigmas
