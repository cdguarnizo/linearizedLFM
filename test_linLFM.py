#! /usr/bin/env python

# linearizedLFM -- Implementation of extended and unscented LFMs.

""" Test script for running the EGP, UGP over a LFM covariance function.
    Based on test code from linearizedGP library
    Modifications: Cristian Guarnizo
"""

import time
import numpy as np
from linearizedGP import unscentedGP
from linearizedGP import extendedGP
import lfmkernels
from linearizedGP.gputils import jitchol, cholsolve
import matplotlib.pyplot as plt
import pickle as pk


# Some parameters for the dataset ---------------------------------------------
npoints = 200   # Training points
ppoints = 200  # Prediction points
noise = 0.01     # Likelihood noise for generated data

# LFM settings

#Covariance function
kfunc = lfmkernels.kern_ode #Second order ODE
k_C = 3.   #Damper parameter
k_B = 1.   #Spring parameter
k_S = 5.   #Sensitivity
k_l = .5   #Length-scale rbr at input

#Type of model
gptype = 'UGP'
#gptype = 'EGP'


# Forward models --------------------------------------------------------------

# Non-differentiable (only works with the UGP)
#nlfunc = lambda f: 2 * np.sign(f) + f**3

# Differentiable
#nlfunc = lambda f: f**2 + 1
#dnlfunc = lambda f: 2*f
#nlfunc = lambda f: f
#dnlfunc = lambda f: np.ones(f.shape)
#nlfunc = lambda f: np.tanh(2*f)
#dnlfunc = lambda f: 2 - 2*np.tanh(f)**2
#nlfunc = lambda f: f**3 + f**2 + f
#dnlfunc = lambda f: 3*f**2 + 2*f + 1
#nlfunc = lambda f: np.exp(f)
#dnlfunc = lambda f: np.exp(f)
nlfunc = lambda f: np.sin(f)
dnlfunc = lambda f: np.cos(f)

# Response and latent function, and data generation -----------------------------------------

# Draw from a LFM
x = np.linspace(0.1, 3., ppoints + npoints)
U, S, V = np.linalg.svd(kfunc(x[np.newaxis, :], x[np.newaxis, :], k_C,k_B, k_S, k_l))
L = U.dot(np.diag(np.sqrt(S))).dot(V)
np.random.seed(10)
f = np.random.randn(ppoints + npoints).dot(L)
y = nlfunc(f) + np.random.randn(npoints + ppoints) * noise

# Training data
tind = np.zeros(ppoints + npoints).astype(bool)
tind[np.random.randint(0, ppoints + npoints, npoints)] = True
xt = x[tind]
ft = f[tind]
yt = y[tind]

# Test data
sind = ~ tind
xs = x[sind]
fs = f[sind]
ys = y[sind]

lml = -np.Inf;
for k in range(10):
# linearized LFM learning --------------------------------------
    if gptype is 'UGP':
        gp = unscentedGP.unscentedGP(nlfunc, kfunc)
    elif gptype is 'EGP':
        gp = extendedGP.extendedGP(nlfunc, dnlfunc, kfunc)
    else:
        raise ValueError('invalid GP type')
    
    # Learn GP
    start = time.clock()
    gp.learnLB((1e-1, 1e-1, -20, 1e-1), ynoise=1e-2)
    gp.learnUB((10., 10., 20, 10.), ynoise=2.)
    try:
        lml_t = gp.learn(xt, yt, np.random.rand(4)+1., ynoise=1., verbose=False)
    except:
        lml_t = -np.Inf

    if lml_t > lml:
        lml = lml_t
        gpt = gp
    
    print("Iteration: {0}, LB: {1}".format(k, lml_t))

gp = gpt
elapsed = (time.clock() - start)

print("\nTraining time = {0} sec".format(elapsed))
print("Lower bound = {0}".format(lml))
print("Hyper-parameters = {0}, noise = {1}".format(gp.kparams, gp.ynoise))
print("Non-linear function parameters = {0}".format(gp.nlparams))

# Predict
start = time.clock()
if gptype is 'UGP':
    Eys, Vys, Ems, Vms = gp.predict(xs)
else:
    Eys, _, Ems, Vms = gp.predict(xs)
elapsed = (time.clock() - start)

print("Prediction time = {0} sec".format(elapsed))


# Performance evaluation ------------------------------------------------------

smse_y = np.mean((ys - Eys)**2)/np.mean((np.mean(ys)-ys)**2)
print("Target prediction normalized mean square error: {0}".format(smse_y))

if gptype is ('UGP' or 'GP'):
    nlpd_y = 0.5 * np.mean((ys - Eys)**2 / Vys + np.log(2 * np.pi * Vys))
    print("Target prediction NLPD: {0}".format(-nlpd_y))

smse_f = np.mean((fs - Ems)**2)/np.mean((np.mean(fs)-fs)**2)
print("Latent prediction normalized mean square error: {0}".format(smse_f))
nlpd_f = 0.5 * np.mean((fs - Ems)**2 / Vms + np.log(2 * np.pi * Vms))
print("Latent prediction NLPD: {0}".format(-nlpd_f))

# Plot the results ------------------------------------------------------------

plt.figure(figsize=(11, 6))
ax = plt.subplot(111)
plt.plot(xt, yt, 'r.', label='Training data $\{t_{n}, y_{n}\}$')
plt.plot(xs, Eys, 'k--', label='Predictions, $\langle y^*\!\\rangle_{qf^*}$',
         linewidth=2)

if gptype is ('UGP' or 'GP'):
    ss = 2. * np.sqrt(Vys)
    plt.fill_between(xs, Eys + ss, Eys - ss, facecolor='gray',
                     edgecolor='gray', alpha=0.3, label=None)
plt.xlabel('Time [s]', fontsize=20)
plt.ylabel('targets ($y$)',fontsize=20)
plt.legend(loc=3, fontsize=18)
plt.autoscale(tight=True)
ax.set_yticklabels(ax.get_yticks(), size=16)
ax.set_xticklabels(ax.get_xticks(), size=16)
plt.grid(True)


plt.figure(figsize=(11, 6))
ax = plt.subplot(111)
plt.plot(xs, fs, label='True process, $f_{true}$', linewidth=2)
plt.plot(xs, Ems, 'k--', label='Estimated process, $m^*$', linewidth=2)
Sfs = 2. * np.sqrt(Vms)
plt.fill_between(xs, Ems + Sfs, Ems - Sfs, facecolor='gray', edgecolor='gray',
                 alpha=0.3, label=None)
plt.xlabel('Time [s]', fontsize=20)
plt.legend(loc=3, fontsize=18)
plt.autoscale(tight=True)
ax.set_yticklabels(ax.get_yticks(), size=16)
ax.set_xticklabels(ax.get_xticks(), size=16)
plt.grid(True)

#plt.savefig('test.pdf', bbox_inches='tight')
#plt.show()

#true Latent function evaluation
#k_C,k_B, k_S, k_l
Kfu = lfmkernels.kern_ode_Kfu(xt[:, np.newaxis],xt[:, np.newaxis],k_C,k_B,k_S,k_l)
Kff = lfmkernels.kern_ode(xt[:, np.newaxis],xt[:, np.newaxis],k_C,k_B,k_S,k_l)
Kffinvfs = cholsolve(jitchol(Kff), ft[:, np.newaxis])
#True mean of u(t)
mu = np.dot(Kfu.T,Kffinvfs)

#estimated latent function u(t)
Kuu = lfmkernels.kern_ode_Kuu(xt[:, np.newaxis],gp.kparams[3])
Kfu = lfmkernels.kern_ode_Kfu(xt[:, np.newaxis],xt[:, np.newaxis],*gp.kparams)
KffinvKfu = cholsolve(gp.Kchol, Kfu)

mus = (KffinvKfu.T).dot(gp.m)
Vms = np.diag(Kuu - KffinvKfu.T.dot( Kfu - gp.C.dot(KffinvKfu)))

plt.figure(figsize=(11, 6))
ax = plt.subplot(111)
plt.plot(xt, mu*k_S, label='True process, $u_{true}$')
plt.plot(xt, mus*gp.kparams[2], 'k--', label='Predictions, $\langle u^*\!\\rangle_{qf^*}$', linewidth=2)
Sus = 2. * np.sqrt(Vms)*np.abs(gp.kparams[2])
plt.fill_between(xt, mus*gp.kparams[2] + Sus, mus*gp.kparams[2] - Sus, facecolor='gray', edgecolor='gray',
                 alpha=0.3, label=None)
plt.xlabel('Time [s]', fontsize=20)
plt.legend(loc=3, fontsize=18)
plt.autoscale(tight=True)
plt.ylim(-15,15)
ax.set_yticklabels(ax.get_yticks(), size=16)
ax.set_xticklabels(ax.get_xticks(), size=16)
plt.grid(True)

#plt.savefig('case3_UGP_nu.pdf', bbox_inches='tight')

plt.show()