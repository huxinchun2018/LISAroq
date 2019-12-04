#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import bilby
from sympy import sin,cos,log,sqrt
from astropy import constants as const
import matplotlib.pyplot as plt

import sys
import os
import six

OrbitRadiusInS = 1e8 /const.c.value    # 1e5 km
MearthInS      = const.M_earth.value*const.G.value/const.c.value**3
OrbitPeriodInS = 2*np.pi*np.sqrt(OrbitRadiusInS**3/MearthInS)
YearInS  = 31536000.0   # one year in [s]

c = 299792458.0  # m/s
G = 6.67*1.0e-11 
AU = 1.5e+11
parsec = 3.085677581491367e+16  # m
solar_mass = 1.9885469549614615e+30  # Kg

def InspiralWaveform(t, m1, m2, DL, Tc, iota, phic, psi, thetaS, phiS):
    c = 299792458.0  # m/s
    G = 6.67*1.0e-11 
    AU = 1.5e11
    fm = 1.0/YearInS
    eta = m1*m2/(m1+m2)**2  # symmetric mass ratio 
    m = m1+m2     # total mass
    tau = c**3*eta/(5*G*m)*(Tc-t);
    wt = c**3/(8*G*m)*(tau**(-3/8)+(743/2688+11/32*eta)*tau**(-5/8)-3*np.pi/10*tau**(-3/4)\
        +(1855099/14450688+56975/258048*eta+371/2048*eta**2)*tau**(-7/8))
    PHI = -2/eta*(tau**(5/8)+(3715/8064+55/96*eta)*tau**(3/8)-3*np.pi/4*tau**(1/4)\
        +(9275495/14450688+284875/258048*eta+1855/2048*eta**2)*tau**(1/8))\
        +wt*AU/c*sin(thetaS)*cos(2*np.pi*fm*t-phiS)      # phi_Doppler
    x = (G*m*wt/c**3)**(2/3)
    hplus = 2*G*m*eta/(c**2*DL)*(1+cos(iota)**2)*x*cos(PHI+phic)
    hcros = -4*G*m*eta/(c**2*DL)*cos(iota)*x*sin(PHI+phic)
    return hplus, hcros

def Dplus_TQ(t,thetaS,phiS):
    """For TianQin (one Michelson interferometer): (thetaS,phiS) is location of source,
    (thJ,phJ) is latitude and longitude of J0806 in heliocentric-ecliptic frame"""
    thJ  = 1.65273
    phJ  = 2.10213
    kap = 2*np.pi/OrbitPeriodInS* t
#    kap = 2*np.pi/YearInS* t(f,theta)
    return np.sqrt(3.)/32*(8*cos(2*kap) *((3 + cos(2*thetaS)) *sin(2*(phiS - phJ))*  
            cos(thJ) + 2*sin(thJ) *sin(phiS - phJ)*sin(2*thetaS))- 2*sin(2*kap)* (3 +               
            cos(2*(phiS - phJ))*(9 + cos(2*thetaS)*(3 + cos(2*thJ))) -6 *cos(2*thJ)*(sin(phiS - phJ))**2 -               
            6* cos(2*thetaS)*(sin(thJ))**2 + 4*cos(phiS - phJ)*sin(2*thJ)*sin(2*thetaS))) 

def Dcros_TQ(t,thetaS,phiS):
    """For TianQin (one Michelson interferometer): (thetaS,phiS) is location of source, 
    (thJ,phJ) is latitude and longitude of J0806 in heliocentric-ecliptic frame"""
    thJ  = 1.65273
    phJ  = 2.10213
    kap = 2*np.pi/OrbitPeriodInS* t
#    kap = 2*np.pi/YearInS* t(f,theta)
    return np.sqrt(3.)/4*(-4*cos(2*kap)*(cos(2*(phiS-phJ))*cos(thJ)*cos(thetaS)+                 
            cos(phiS-phJ)*sin(thetaS)*sin(thJ))+sin(2*kap)*(cos(thetaS)*(3+cos(2*thJ))*sin(2*(phJ-phiS))+                
            2*sin(phJ-phiS)*sin(thetaS)*sin(2*thJ)))

def Fplus_TQ(t,thetaS,phiS,psi):
    """antenna pattern function for '+' mode"""
    return (cos(2*psi)*Dplus_TQ(t,thetaS,phiS)-sin(2*psi)*Dcros_TQ(t,thetaS,phiS))/2.

def Fcros_TQ(t,thetaS,phiS,psi):
    """antenna pattern function for '×' mode"""
    return (sin(2*psi)*Dplus_TQ(t,thetaS,phiS)+cos(2*psi)*Dcros_TQ(t,thetaS,phiS))/2.


m1 = 1.0e6*solar_mass
m2 = 2.0e5*solar_mass
Tc = 1.04*YearInS
DL = 11.01*1.0e9*parsec


thetaS = 1.6398
phiS = 4.1226
psi = 1.373
iota = 0.7647
phic = 0.9849


Dplus = []
Dcros = []
Fplus = []
Fcros = []

duration = 1638400
time = np.arange(0, duration, 50)
print(len(time))

for i in time:
    Dp = Dplus_TQ(i,thetaS,phiS)
    Dc = Dcros_TQ(i,thetaS,phiS)
    Fp = Fplus_TQ(i,thetaS,phiS,psi)
    Fc = Fcros_TQ(i,thetaS,phiS,psi)
    Dplus.append(Dp)
    Dcros.append(Dc)
    Fplus.append(Fp)
    Fcros.append(Fc)

Dplus = np.array(Dplus)         #Convert list to array
Dcros = np.array(Dcros)         #Convert list to array
Fplus = np.array(Fplus)         #Convert list to array
Fcros = np.array(Fcros)         #Convert list to array

fig = plt.plot(time, Fcros, label='Fx')
fig = plt.plot(time, Fplus, label='F+')
plt.xlabel('time(s)')
plt.ylabel('Response')
plt.legend()
plt.title('TianQin Antenna pattern function')
plt.savefig('TianQin Antenna pattern function',dpi=300)


hplus = []
hcros = []
for i in time:
    hp, hc = InspiralWaveform(i,m1, m2, DL, Tc, iota, phic, psi, thetaS, phiS)
    hplus.append(hp)
    hcros.append(hc)
    
hplus = np.array(hplus)  # convert list to array
hcros = np.array(hcros)  # convert list to array

# signal
strain = Fplus*hplus + Fcros*hcros

# noise
N = len(time)
mean = 0
sigma = 1.0e-21
noise   = sigma*np.random.randn(N)
data_time = strain + noise

plt.hist(noise,50)
plt.savefig('noise',dpi=300)

def log_likelihood(theta):
    """
    Returns
    ------
    float: The real part of the log-likelihood for this interferometer
    """
    
    m1, m2, DL, Tc, iota, phic, psi, thetaS, phiS = theta
    hplus = []
    hcros = []
    for i in time:
        hplus_temp, hcros_temp = np.array(InspiralWaveform(i,m1, m2, DL, Tc, iota, phic, psi, thetaS, phiS))
        hplus.append(hplus_temp)
        hcros.append(hcros_temp)
    
    ht = np.array(Fplus_TQ(i,thetaS,phiS,psi))*np.array(hplus) + np.array(Fcros_TQ(i,thetaS,phiS,psi))*np.array(hcros)
    
    log_l = np.real(- 2. / duration * np.vdot(data_time - ht,(data_time - ht)))
    return log_l

def prior_transform(theta):

    # unpack the model parameters from the tuple
#     m1,m2,DL,Tc,iota,phic,psi,thetaS,phiS=theta
    m1prime, m2prime, DLprime, Tcprime, iotaprime, phicprime, psiprime, thetaSprime, phiSprime = theta
    
    # uniform prior on c
    m1min = 5.0e5*solar_mass   # lower range of prior
    m1max = 5.0e6*solar_mass   # upper range of prior
    
    m2min = 1.0e5*solar_mass   # lower range of prior
    m2max = 5.0e5*solar_mass   # upper range of prior
     
    DLmin = 10*1.0e9*parsec
    DLmax = 15*1.0e9*parsec
    
    Tcmin = 0.9*YearInS
    Tcmax = 1.2*YearInS
    
    iotamin = 0
    iotamax = np.pi
    
    phicmin = 0
    phicmax = 2*np.pi
     
    psimin = 0
    psimax = 2*np.pi
    
    thetaSmin = 0
    thetaSmax = np.pi
    
    phiSmin = 0
    phiSmax = 2*np.pi
    
    m1 = m1prime*(m1max-m1min)+m1min
    m2 = m2prime*(m2max-m2min)+m2min
    DL = DLprime*(DLmax-DLmin)+DLmin
    Tc = Tcprime*(Tcmax-Tcmin)+Tcmin
    iota = iotaprime*(iotamax - iotamin)+iotamin
    phic = phicprime*(phicmax-phicmin)+phicmin
    psi = psiprime*(psimax-psimin)+psimin
    thetaS = thetaSprime*(thetaSmax - thetaSmin)+thetaSmin
    phiS = phiSprime*(phiSmax-phiSmin)+phiSmin

    return m1, m2, DL, Tc, iota, phic, psi, thetaS, phiS

# import nestle
import nestle
from datetime import datetime
print('Nestle version: {}'.format(nestle.__version__))

nlive = 256     # number of live points
method = 'multi' # use MutliNest algorithm
ndims = 9        # two parameters
tol = 100        # the stopping criterion

t0 = datetime.now()
res = nestle.sample(log_likelihood, prior_transform, ndims, method=method, npoints=nlive, dlogz=tol)
t1 = datetime.now()

timeultranest = (t1-t0)
print("Time taken to run 'UltraNest' is {} seconds".format(timeultranest))



