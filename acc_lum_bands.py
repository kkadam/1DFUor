#****************************************************************************
# Description:
# The code estimates the accretion luminosity of the inner disk,
# and also the V magnitude of the object, given-
# Mstar, Mdot, Rstar, Rout, d, inc
# "infile" contains 3 columns: 
# Time in arbitrary units, Mstar in units of Msun and Mdot in units Msun/yr
# IDL to python notes- http://mathesaurus.sourceforge.net/idl-numpy.html
#****************************************************************************

import numpy as np
import matplotlib.pyplot as plt
import sys

#----------------------------------------------------------------------------
# Initialization
#----------------------------------------------------------------------------

# Physical and astronomical constants

Rsun  = 6.957E+10     # solar radius in cm
Msun  = 1.9891E+33    # solar mass in gramm
g     = 6.674E-8      # gravitational constant in CGS
sb    = 5.6704E-5     # Stefan-Boltzmann constant in CGS
h     = 6.62620E-27   # Planck constant in CGS
kb     = 1.380662E-16 # Boltzmann constant in CGS
c     = 2.99792E+10   # light speed in cm/s
Lsun  = 3.828E+33     # solar luminosity in CGS

# Parameters of the object

Rstar = 5.0 * Rsun                          # stellar radius in cm
Rout  = 2.0                                 # Outer radius of the accretion disk in AU
d     = 700.                                # distance in pc
inc   = 40.                                 # inclination of the disk in degree, 90 - edge-on
nrings = 100                                # number of annular rings in the inner disk
nfreq = 301                                 # number of points in frequency for SED calculations

# Don't change these

Rout  = Rout * 1.49597871E+13               # Outer radius in cm
d     = d * 3.08567758E+18                  # distance in cm 
Lwave = 88/1000.                            # lower end of V band in micrometer
Hwave = 551/1000.                           # higher end of V band in micrometer


#----------------------------------------------------------------------------
# Import accretion parameters
#----------------------------------------------------------------------------

datarray=np.loadtxt("infile")

time = datarray[:,0]

Mstar = datarray[:,1]
Mstar = Mstar * Msun                          # stellar mass in gramm

Mdot  = datarray[:,2]
Mdot  = Mdot * Msun / (365.24 * 24 * 3600.)   # Accretion rate in g/s

MMdot = Mstar * Mdot


#----------------------------------------------------------------------------
# Initialize arrays
#----------------------------------------------------------------------------

# Make logarithmically separated rings from Rstar to Rout

r1 = np.logspace(np.log10(Rstar),np.log10(Rout),nrings,endpoint=False)
r2 = np.logspace(np.log10(r1[1]),np.log10(Rout),nrings,endpoint=True)
radius = (r1+r2)/2.0

# Find area of rings 

area = np.pi * (r2**2 - r1**2)

# Define wavelengths for SED calculation 
# Logarithmically distributed points in the V band and nu array in Hz
 
lam = np.logspace(np.log10(Lwave),np.log10(Hwave),nfreq,endpoint=True)
nu = c / (lam * 1.0e-4) 

# Solid angle subtended by each annulus

solang = area / (d**2)* np.cos(inc * np.pi / 180.0)

# More initialization

temp = np.zeros(nrings)
sed = np.zeros(nfreq)
SR = np.zeros((nrings,nfreq), dtype=np.float128)
dnu = nu[0:nfreq-1] -nu[1:nfreq]  
Ltot = np.zeros(len(time))
mV = np.zeros(len(time))


#----------------------------------------------------------------------------
# Loop over the timesteps
#----------------------------------------------------------------------------
# j -> time, i -> nrings, k -> nfreq

for j in range (0,len(time)):

# Loop over annuli
    for i in range (0,nrings):

# Compute temperature profile of the rings and compensate for the 
# drop in temperature near the inner edge

        t4   = ((3.0 * g * MMdot[j]) / (8.0 * np.pi * radius[i]**3. * sb)) * (1. - np.sqrt(Rstar/radius[i]))
        temp[i] = (t4**0.25)

        tempmax = np.amax(temp)

    for i in range (0, np.argmax(temp)):
        temp[i] = tempmax

# Calculate luminosity of each ring and the sum to find total accretion luminosity in Lsun

    lum = (area*sb*np.power(temp,4))
    Ltot[j] = np.sum(lum)/Lsun
#    print ("Ltot =", Ltot[j])


# Compute planck spectrum of each ring, SR is spectral radiance array (erg/s/cm2/Hz)
# /sr is taken care of by solang multiplication
    
    for i in range (0,nrings):
        for k in range (0,nfreq):
            SR[i,k] = solang[i] * (2.*h*nu[k]**3 / c**2) * (1.0/(np.exp(h*nu[k]/(kb*temp[i])) - 1.0))

# Compute the SED of the entire disk by summing over rings 
# Units same as SR (erg/s/cm2/Hz)
    sed = SR.sum(axis=0)

# Compute following two if flux is needed in the given wavelength band
#    dSR = (sed[0:nfreq-1] + sed[1:nfreq]) / 2.0
#    dnudSR = np.multiply(dnu, dSR) 

# Compute flux density for entire band (erg/s/cm2/Hz)
    bandfluxden = sed.sum() 

# Convert flux density to Johnson V magnitude, mV
# More info on: https://archive.is/F0WR
    mAB = -2.5 * np.log10(bandfluxden) - 48.6
    mV[j] = mAB + 0.044
#    print ("mV", mV)

# Save lightcurve into output file

np.savetxt("lightcurve.out",np.column_stack((time,Ltot,mV)))

print ("Bolometric luminosity  and V- magnitude stored in file- lightcurve.out")


#----------------------------------------------------------------------------
# Tests
#----------------------------------------------------------------------------
#print ("time", time)
#print ("Ltot",Ltot)

#testplot=plt.semilogx(r1, lum, '--')
#testplot=plt.semilogx(r1, lum, '--')
#plt.plot(time,Ltot)
#plt.show()

#y = np.zeros(nfreq)

#plt.plot(lam,y,'x')
#plt.show()

# Peter's value was acclu = 3.9241295211996935978

#****************************************************************************
