#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 09:40:18 2019

@author: hxc
"""

import numpy
import numpy as np
import scipy
# matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import lal
import lalsimulation
from lal.lal import PC_SI as LAL_PC_SI
import h5py
import warnings
import random
warnings.filterwarnings('ignore')
import matplotlib.pylab as pylab
plot_params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 9),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(plot_params)
from mpl_toolkits.mplot3d import axes3d
#import PyROQ.pyroq as pyroq
import pyroq as pyroq

mc_low = 1.0e5
mc_high = 1.0e6
q_low = 1
q_high = 2
s1sphere_low = [0.5, 0, 0]
s1sphere_high = [0.9, numpy.pi, 2.0*numpy.pi]
s2sphere_low = [0.5, 0, 0]
s2sphere_high = [0.9, numpy.pi, 2.0*numpy.pi]
ecc_low = 0.0
ecc_high = 0.2
lambda1_low = 0
lambda1_high = 1000
lambda2_low = 0
lambda2_high = 1000
iota_low = 0
iota_high = numpy.pi
phiref_low = 0
phiref_high = 2*numpy.pi
f_min = 1.0e-4
f_max = 1.0e-1
deltaF = 1.0e-7
distance = 10 * LAL_PC_SI * 1.0e6  # 10 Mpc is default 
waveFlags = lal.CreateDict()


# approximant = lalsimulation.IMRPhenomPv2_NRTidal
# approximant = lalsimulation.TaylorF2Ecc
approximant = lalsimulation.IMRPhenomPv2
print("mass-min, mass-max: ", pyroq.massrange(mc_low, mc_high, q_low, q_high))

npts = 300 # Specify the number of points for each search for a new basis element
nts = 1000 # Number of random test waveforms

nbases = 600 # Specify the number of linear basis elements
ndimlow = 60 # Starting number of basis elements used to check if tolerance is satisfied
ndimhigh = nbases+1
ndimstepsize = 10 # Number of linear basis elements increament to get the basis satisfying the tolerance
tolerance = 1e-8 # Surrogage error threshold for linear basis elements

nbases_quad = 90 # Specify the number of quadratic basis elements
ndimlow_quad = 40 
ndimhigh_quad = nbases_quad + 1
ndimstepsize_quad = 10
tolerance_quad = 1e-6 # Surrogage error threshold for quadratic basis elements

# m1, m2 = pyroq.get_m1m2_from_mcq(mc_high, q_high)
# m3, m4 = pyroq.get_m1m2_from_mcq(mc_high, q_low)
# m5, m6 = pyroq.get_m1m2_from_mcq(mc_low, q_high)
# m7, m8 = pyroq.get_m1m2_from_mcq(mc_low, q_low)
# print(m1, m2, m3, m4, m5, m6, m7, m8)


freq = numpy.arange(f_min,f_max,deltaF)
nparams, params_low, params_high, params_start, hp1 = pyroq.initial_basis(mc_low, mc_high, q_low, q_high, s1sphere_low, s1sphere_high, \
                  s2sphere_low, s2sphere_high, ecc_low, ecc_high, lambda1_low, lambda1_high,\
                 lambda2_low, lambda2_high, iota_low, iota_high, phiref_low, phiref_high, distance, deltaF, f_min, f_max, waveFlags, approximant)

from datetime import datetime
start_time = datetime.now()

known_bases_start = numpy.array([hp1/numpy.sqrt(numpy.vdot(hp1,hp1))])
basis_waveforms_start = numpy.array([hp1])
residual_modula_start = numpy.array([0.0])
known_bases, params, residual_modula = pyroq.bases_searching_results_unnormalized(npts, nparams, nbases, known_bases_start, basis_waveforms_start, params_start, residual_modula_start, params_low, params_high, distance, deltaF, f_min, f_max, waveFlags, approximant)
print(known_bases.shape, residual_modula)
known_bases_copy = known_bases
known_bases = known_bases_copy

end_time = datetime.now()
time_dt = end_time - start_time
print('running_time =',time_dt)

pyroq.roqs(tolerance, freq, ndimlow, ndimhigh, ndimstepsize, known_bases, nts, nparams, params_low, params_high, distance, deltaF, f_min, f_max, waveFlags, approximant)
b1 = numpy.load('./B_linear.npy')
f1 = numpy.load('./fnodes_linear.npy')
#print(b1.shape, f1.shape)
b_linear = numpy.transpose(numpy.load('./B_linear.npy'))
ndim = b_linear.shape[1]
print("Number of basis elements: ", ndim)

test_mc = 5.0e5
test_q = 1
test_s1 = [0.85,0.15,-0.0]
test_s2 = [0.85,0.15,-0.1]
test_ecc = 0
test_lambda1 = 0
test_lambda2 = 0
test_iota = 1.9
test_phiref = 0.6

ndim, inverse_V, emp_nodes = pyroq.empnodes(ndim, known_bases_copy)
pyroq.testrep(b_linear, emp_nodes, test_mc, test_q, test_s1, test_s2, test_ecc, test_lambda1, test_lambda2, test_iota, test_phiref, distance, deltaF, f_min, f_max, waveFlags, approximant)
 

nsamples = 100000 # testing nsamples random samples in parameter space to see their representation surrogate errors
surros = pyroq.surros_of_test_samples(nsamples, nparams, params_low, params_high, tolerance, ndim, known_bases_copy, distance, deltaF, f_min, f_max, waveFlags, approximant)
# If a surrogate error is larger than tolerance, it will be reported on the screen.

plt.semilogy(surros,'o',color='black')
plt.xlabel("Number of Random Test Points")
plt.ylabel("Surrogate Error")
plt.title("IMRPhenomPv2")
plt.savefig("SurrogateErrorsRandomTestPoints.png")
plt.show()

hp1_quad = (numpy.absolute(hp1))**2
known_quad_bases_start = numpy.array([hp1_quad/numpy.sqrt(numpy.vdot(hp1_quad,hp1_quad))])
basis_waveforms_quad_start = numpy.array([hp1_quad])
residual_modula_start = numpy.array([0.0])
known_quad_bases,params_quad,residual_modula_quad = pyroq.bases_searching_quadratic_results_unnormalized(npts, nparams, nbases_quad, known_quad_bases_start, basis_waveforms_quad_start, params_start, residual_modula_start, params_low, params_high, distance, deltaF, f_min, f_max, waveFlags, approximant)
known_quad_bases_copy = known_quad_bases

points = numpy.random.uniform(params_low, params_high, size=(npts,nparams))

known_quad_bases = known_quad_bases_copy
print(distance, deltaF, f_min, f_max)
pyroq.roqs_quad(tolerance_quad, freq, ndimlow_quad, ndimhigh_quad, ndimstepsize_quad, known_quad_bases, nts, nparams, params_low, params_high, distance, deltaF, f_min, f_max, waveFlags, approximant)
b2 = numpy.load('./B_quadratic.npy')
f2 = numpy.load('./fnodes_quadratic.npy')
print(b2.shape, f2.shape)

test_mc_quad =5.0e5
test_q_quad = 1
test_s1_quad = [0.85, 0.0, 0.0]
test_s2_quad = [0.85, 0.0, 0.0]
test_ecc_quad = 0
test_lambda1_quad = 0
test_lambda2_quad = 0
test_iota_quad = 1.0
test_phiref_quad = 0.9

b_quad = numpy.transpose(numpy.load('./B_quadratic.npy'))
ndim_quad = b_quad.shape[1]
print(ndim_quad)
ndim_quad, inverse_V_quad, emp_nodes_quad = pyroq.empnodes_quad(ndim_quad, known_quad_bases_copy)
pyroq.testrep_quad(b_quad, emp_nodes_quad, test_mc_quad, test_q_quad, test_s1_quad, test_s2_quad, test_ecc_quad, test_lambda1_quad, test_lambda2_quad, test_iota_quad, test_phiref_quad, distance, deltaF, f_min, f_max, waveFlags, approximant)



