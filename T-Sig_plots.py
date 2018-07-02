#****************************************************************************
# T-Sig_plots.py 
# Description:
# Calculates T vs Sigma plot at specified radius, R1
# Needs extracted data in COLORED directory from script R-t_plots.py
#****************************************************************************
from numpy import *
from pylab import *
#import sys 
import os
import glob 
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext

#----------------------------------------------------------------------------
# Initialization
#----------------------------------------------------------------------------

R1 = 1.0         # Radius for the T-Sigma plot

nx = 121         # Resolution of the simulation
inputdir = "COLORED/"

# Define functions
# Find the index corresponding to the nearest value
def find_nearest(input_array,value):
    temp_array=np.abs(input_array - value)
    idx = np.argmin(temp_array)
    return idx 

# ----------------------------------------------------
# Start program
# ----------------------------------------------------

# Read the Sigma, T, radius and time arrays
sigData = np.loadtxt(inputdir+'sigma')
TcData = np.loadtxt(inputdir+'Tc')
radialData = np.loadtxt(inputdir+'R')
timeData = np.loadtxt(inputdir+'time')

print (sigData.size, TcData.size, radialData.size, timeData.size, 121*6000)
    
# Find the index where the radius is closest to specified R
R1_index=find_nearest(radialData,R1)
print ("R1_index",R1_index,"At",radialData[R1_index])

sigR1 = sigData[R1_index,:]
TcR1 = TcData[R1_index,:]

# Save data and make plot
np.savetxt('TSig_'+str(R1)+'.txt', np.c_[TcR1,sigR1])

plt.plot(np.log10(sigR1),np.log10(TcR1))
plt.xlabel("log10(Sigma/gcm-2)")
plt.ylabel("log10(Tc/K)")
plt.show()

exit()

