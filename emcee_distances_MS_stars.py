
# Importing the necessary packages:

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
from scipy.optimize import minimize
from scipy import stats as st
import emcee
import corner
import sys

## Here I fix the random seed ##
np.random.seed(20160806)

### Setting the initial values for the parameters that we will sample ###

M_mean = 4.12
Delta = 0.1
Sigma_M = 0.11
Sigma_M2 = 0.16
a1 = -2.9
a1_2 = 4.64
a2 = 0.73
a3 = -0.268

## Number of stars to try the model first. Change to total length in the data file. ##
Nstars = 80

## I read the file with the data that has good parallax information (10 percent uncertainty in the parallax). We need the K-mag, logg, Teff, parallax and [Fe/H] information##
data= np.loadtxt("LAMOST_TGAS_10percent.dat", unpack=False)
parallax = data[:Nstars,0] / 1000
err_par = data[:Nstars,1] / 1000.
K_mag = data[:Nstars,2]
err_mag = data[:Nstars,3]
Teff_1 = data[:Nstars,6]
Teff_2 = (Teff_1 - 4800.)/4800.
logg_1 = data[:Nstars,7]
logg_2 = logg_1 - np.mean(logg_1)
Fe_H_1= data[:Nstars,8]
Fe_H_2 = Fe_H_1 - np.mean(Fe_H_1)
D = data[:Nstars,11]
### initial vector also needs the new parameters (constants)
vini =np.append(D, [Delta,Sigma_M2,a1,a1_2,a2,a3,M_mean,Sigma_M])

## Defining all of the priors (almost all flat), except for the distance ##
def lnprior(d):
    if d > 0:
        return 2*np.log(d)-d/1000
    return -np.inf

def lnprior_delta(delta):
    if delta < 0. or delta > 1.:
        return -np.inf
    return 0.0


def lnprior_sigma(sigma_m):
    if sigma_m < 0. or sigma_m > 1.:
        return -np.inf
    return 0.0

def lnprior_sigma2(sigma_m2):
    if sigma_m2 < 0. or sigma_m2 > 1.:
        return -np.inf
    return 0.0


def lnprior_M(M0):
    if M0 < 3.0 or M0 > 4.5:
        return -np.inf
    return 0.0


def lnprior_a1(a1):
    if a1 < -10. or a1 > 10.:
        return -np.inf
    return 0.0

def lnprior_a2(a2):
    if a2 < -10. or a2 > 10.:
        return -np.inf
    return 0.0
def lnprior_a3(a3):
    if a3 < -10. or a3 > 10.:
        return -np.inf
    return 0.0


def lnprior_a1_2(a1_2):
    if a1_2 < -10. or a1_2 > 10.:
        return -np.inf
    return 0.0

def lnlike_par(d, parallax, err_parallax):
    if d < 0:
        return -np.inf
    return -0.5*((parallax - 1./d)**2/err_parallax**2 + np.log(2*np.pi*(err_parallax**2)))


## Setting the likelihood function of the mean abs.magnitude as described in Eq.(3) in Coronado+2018 ##
def lnlike_mag(d,M,sigma_m,sigma_m2,delta,K_mag, K_mag_err,logg,T,fe_h,a1,a1_2,a2,a3):
    if d < 0:
        return -np.inf
    mn = M + 5. * np.log10(d/10.)+a1*T + a2*logg + a3*fe_h + a1_2*T**2
    mn2 = mn - 0.6
    gaus1 = 1/(np.sqrt(2*np.pi*(K_mag_err**2+sigma_m**2)))*np.exp((-0.5*((K_mag - mn)**2/(K_mag_err**2+sigma_m**2))))
    gaus2 = 1/(np.sqrt(2*np.pi*(K_mag_err**2+sigma_m2**2)))*np.exp((-0.5*((K_mag - mn2)**2/(K_mag_err**2+sigma_m2**2))))
    SUM = (1-delta)*gaus1 + delta*gaus2
    if SUM==0:
        return -np.inf
    return np.log((1-delta)*gaus1 + delta*gaus2)


## Here we fit the model simultaneously for the distances and all of the rest of the parameters ##
def lnprob(theta, parallax, err_parallax , K_mag, K_mag_err, logg_2,Teff_2, Fe_H_2):
    dists = theta[:-8]
    delta,sigma_m2,a1,a1_2,a2,a3,M0,sigma_m  = theta[-8:]
    sum = lnprior_delta(delta)+lnprior_sigma(sigma_m)+lnprior_sigma2(sigma_m2) +lnprior_M(M0) + lnprior_a1(a1) +lnprior_a1_2(a1_2) +lnprior_a2(a2) + lnprior_a3(a3)
    for d,par,epar,Km,Kme,log_g,T_eff,fe_h in zip(dists,parallax,err_parallax,K_mag,K_mag_err,logg_2,Teff_2,Fe_H_2):
        if not np.isfinite(sum):
            break
        sum += lnprior(d)+ lnlike_par(d,par,epar)+lnlike_mag(d,M0,sigma_m,sigma_m2,delta,Km,Kme,log_g,T_eff,fe_h,a1,a1_2,a2,a3)
    return sum

Nparams = len(vini)
ndim, nwalkers = Nparams, 250

p0 = [vini + 1e-4*np.random.randn(ndim) for j in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(parallax, err_par, K_mag, err_mag, logg_2,Teff_2,Fe_H_2),threads=8)

pos,prob,state=sampler.run_mcmc(p0, 2000)
sampler.reset()
sampler.run_mcmc(pos,2000)


samples = sampler.chain[:, -1000:, :].reshape((-1, ndim))
result = np.percentile(samples, [16,50,84], axis=0)

for i in range(Nstars):
    print ("d_mcmc=%.2f +%.2f -%.2f"%(result[1,i],result[2,i]-result[1,i],result[1,i]-result[0,i]))

print ("sigma_mcmc=%.2f +%.2f -%.2f"%(result[1,-1],result[2,-1]-result[1,-1],result[1,-1]-result[0,-1]))
print ("M_mcm=%.2f +%.2f -%.2f"%(result[1,-2],result[2,-2]-result[1,-2],result[1,-2]-result[0,-2]))
print ("a1_mcmc=%.3e +%.3e -%.3e"%(result[1,-3],result[2,-3]-result[1,-3],result[1,-3]-result[0,-3]))
print ("a1_2_mcmc=%.3e +%.3e -%.3e"%(result[1,-4],result[2,-4]-result[1,-4],result[1,-4]-result[0,-4]))
print ("a2_mcmc=%.2f +%.2f -%.2f"%(result[1,-5],result[2,-5]-result[1,-5],result[1,-5]-result[0,-5]))
print ("a3_mcmc=%.3e +%.3e -%.3e"%(result[1,-6],result[2,-6]-result[1,-6],result[1,-6]-result[0,-6]))
print ("sigma_mcmc_2=%.3e +%.3e -%.3e"%(result[1,-7],result[2,-7]-result[1,-7],result[1,-7]-result[0,-7]))
print ("delta=%.3e +%.3e -%.3e"%(result[1,-8],result[2,-8]-result[1,-8],result[1,-8]-result[0,-8]))

## Save the chains to plot later ##
for k in range(nwalkers):
    np.savetxt('chain_%02d.dat'%(k+1),sampler.chain[k,-1000:,:], fmt = '%11.8lf')


## Some plots ##
'''

plt.clf()
fig, axes = plt.subplots(3,1, sharex=True)
axes[0].plot(sampler.chain[:,:,7].T, color='k', alpha=0.4)
axes[1].plot(sampler.chain[:,:,-8].T, color='k', alpha=0.4)
axes[2].plot(sampler.chain[:,:,-1].T, color='k', alpha=0.4)
plt.show()

plt.subplot(2,1,1)
plt.hist(sampler.flatchain[1000:,-7], histtype='step', bins=20)
plt.axvline(result[1,-7], linestyle='--', label="%.3f"%result[1,-7])
plt.xlabel('$\delta$')
plt.legend(loc='best')

plt.subplot(2,1,2)
plt.hist(sampler.flatchain[1000:,-1], histtype='step', bins=20)
plt.axvline(result[1,-1], linestyle='--', label="%.3f"%result[1,-1])
plt.xlabel('$\sigma_{M}$')
plt.legend(loc='best')
plt.show()

'''
