#!/usr/bin/env python
# coding: utf-8

# In[15]:


# get_ipython().system('pip install --user PyROQ==0.1.25')


# In[16]:


import numpy
import numpy as np
import scipy
# get_ipython().run_line_magic('matplotlib', 'inline')
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


# # Setting up boundary conditions and tolerance requirements.

# In[17]:


mc_low = 1.0e5
mc_high = 4.0e5
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
deltaF = 1.0/(0.5*86400*365) # 1/(0.5*yr)
distance = 10 * LAL_PC_SI * 1.0e6  # 10 Mpc is default 
waveFlags = lal.CreateDict()
approximant = lalsimulation.IMRPhenomPv2
print("mass-min, mass-max: ", pyroq.massrange(mc_low, mc_high, q_low, q_high))

parallel = 0 # The parallel=1 will turn on multiprocesses to search for a new basis. To turn it off, set it to be 0.
             # Do not turn it on if the waveform generation is not slow compared to data reading and writing to files.
             # This is more useful when each waveform takes larger than 0.01 sec to generate.
nprocesses = 4 # Set the number of parallel processes when searching for a new basis.  nprocesses=mp.cpu_count()


nts = 1000 # Number of random test waveforms
          # For diagnostics, 1000 is fine.
          # For real ROQs calculation, set it to be 1000000.

npts = 1000 # Specify the number of points for each search for a new basis element
          # For diagnostic testing, 30 -100 is fine. 
          # For real ROQs computation, this can be 300 to 2000, roughly comparable to the number of basis elments.
          # What value to choose depends on the nature of the waveform, such as how many features it has. 
          # It also depends on the parameter space and the signal length. 
        
nbases = 1000 # Specify the number of linear basis elements. Put your estimation here for the chunk of parameter space.
ndimlow = 600 # Your estimation of fewest basis elements needed for this chunk of parameter space.
ndimhigh = nbases+1 
ndimstepsize = 10 # Number of linear basis elements increament to check if the basis satisfies the tolerance.
tolerance = 1e-8 # Surrogage error threshold for linear basis elements

nbases_quad = 800 # Specify the number of quadratic basis elements, depending on the tolerance_quad, usually two thirds of that for linear basis
ndimlow_quad = 400
ndimhigh_quad = nbases_quad+1
ndimstepsize_quad = 10
tolerance_quad = 1e-10 # Surrogage error threshold for quadratic basis elements


# In[18]:


freq = numpy.arange(f_min,f_max,deltaF)
nparams, params_low, params_high, params_start, hp1 = pyroq.initial_basis(mc_low, mc_high, q_low, q_high, s1sphere_low, s1sphere_high,                   s2sphere_low, s2sphere_high, ecc_low, ecc_high, lambda1_low, lambda1_high,                 lambda2_low, lambda2_high, iota_low, iota_high, phiref_low, phiref_high, distance, deltaF, f_min, f_max, waveFlags, approximant)


# In[5]:


print(nparams)
print(len(freq))


# # Search for linear basis elements to build and save linear ROQ data in the local directory.

# In[5]:


known_bases_start = numpy.array([hp1/numpy.sqrt(numpy.vdot(hp1,hp1))])
basis_waveforms_start = numpy.array([hp1])
residual_modula_start = numpy.array([0.0])
known_bases, params, residual_modula = pyroq.bases_searching_results_unnormalized(parallel, nprocesses, npts, nparams, nbases, known_bases_start, basis_waveforms_start, params_start, residual_modula_start, params_low, params_high, distance, deltaF, f_min, f_max, waveFlags, approximant)
print(known_bases.shape, residual_modula)


# In[6]:


known_bases = numpy.load('./linearbases.npy')
pyroq.roqs(tolerance, freq, ndimlow, ndimhigh, ndimstepsize, known_bases, nts, nparams, params_low, params_high, distance, deltaF, f_min, f_max, waveFlags, approximant)


# In[7]:


fnodes_linear = numpy.load('./fnodes_linear.npy')
b_linear = numpy.transpose(numpy.load('./B_linear.npy'))
ndim = b_linear.shape[1]
freq = numpy.arange(f_min, f_max, deltaF)
emp_nodes = numpy.searchsorted(freq, fnodes_linear)
print(b_linear)
print("emp_nodes", emp_nodes)


# In[8]:


test_mc = 3.0e5
test_q = 2
test_s1 = [0.85,0.2,-0.]
test_s2 = [0.85,0.15,-0.1]
test_ecc = 0
test_lambda1 = 0
test_lambda2 = 0
test_iota = 1.9
test_phiref = 0.6

pyroq.testrep(b_linear, emp_nodes, test_mc, test_q, test_s1, test_s2, test_ecc, test_lambda1, test_lambda2, test_iota, test_phiref, distance, deltaF, f_min, f_max, waveFlags, approximant)


# In[9]:


nsamples = 1000 # testing nsamples random samples in parameter space to see their representation surrogate errors
surros = pyroq.surros_of_test_samples(nsamples, nparams, params_low, params_high, tolerance, b_linear, emp_nodes, distance, deltaF, f_min, f_max, waveFlags, approximant)
# If a surrogate error is larger than tolerance, it will be reported on the screen.


# In[10]:


plt.figure(figsize=(15,9))
plt.semilogy(surros,'o',color='black')
plt.xlabel("Number of Random Test Points")
plt.ylabel("Surrogate Error")
plt.title("IMRPhenomPv2")
plt.savefig("SurrogateErrorsRandomTestPoints.png")
plt.show()


# # Search for quadratic basis elements to build & save quadratic ROQ data.

# In[11]:


hp1_quad = (numpy.absolute(hp1))**2
known_quad_bases_start = numpy.array([hp1_quad/numpy.sqrt(numpy.vdot(hp1_quad,hp1_quad))])
basis_waveforms_quad_start = numpy.array([hp1_quad])
residual_modula_start = numpy.array([0.0])
known_quad_bases,params_quad,residual_modula_quad = pyroq.bases_searching_quadratic_results_unnormalized(parallel, nprocesses, npts, nparams, nbases_quad, known_quad_bases_start, basis_waveforms_quad_start, params_start, residual_modula_start, params_low, params_high, distance, deltaF, f_min, f_max, waveFlags, approximant)
known_quad_bases_copy = known_quad_bases


# In[12]:


known_quad_bases = numpy.load('./quadraticbases.npy')
pyroq.roqs_quad(tolerance_quad, freq, ndimlow_quad, ndimhigh_quad, ndimstepsize_quad, known_quad_bases, nts, nparams, params_low, params_high, distance, deltaF, f_min, f_max, waveFlags, approximant)


# In[13]:


fnodes_quad = numpy.load('./fnodes_quadratic.npy')
b_quad = numpy.transpose(numpy.load('./B_quadratic.npy'))
ndim_quad = b_quad.shape[1]
freq = numpy.arange(f_min, f_max, deltaF)
emp_nodes_quad = numpy.searchsorted(freq, fnodes_quad)


# In[14]:


test_mc_quad =3.0e5
test_q_quad = 1.2
test_s1_quad = [0.8, 0.1, 0.0]
test_s2_quad = [0.8, 0.0, 0.0]
test_ecc_quad = 0
test_lambda1_quad = 0
test_lambda2_quad = 0
test_iota_quad = 1.9
test_phiref_quad = 0.6

pyroq.testrep_quad(b_quad, emp_nodes_quad, test_mc_quad, test_q_quad, test_s1_quad, test_s2_quad, test_ecc_quad, test_lambda1_quad, test_lambda2_quad, test_iota_quad, test_phiref_quad, distance, deltaF, f_min, f_max, waveFlags, approximant)






