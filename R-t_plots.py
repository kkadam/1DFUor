#****************************************************************************
# Description:
# Calculates R vs time plots for fields- Sigma, Tc, nu, Q -as the color info 
#****************************************************************************
from numpy import *
from pylab import *
#import sys 
#import os
import glob 
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext

#----------------------------------------------------------------------------
# Initialization
#----------------------------------------------------------------------------

tmin = 0
tmax = 0.9
Rmin = 0.02
Rmax = 40
autorange = 0
logscaleR = 1

# Get input file names
fileslist = glob.glob('a*.txt')
fileslist.sort()
infiles = np.hstack(fileslist)

nt = infiles.size

# Get the radial array
fileData = np.loadtxt(fileslist[0])
radius = fileData[:,1]

nR = radius.size

# Make Sigma array
sigma = np.empty([nR,nt])
Tc = np.empty([nR,nt])
nu = np.empty([nR,nt])
Q = np.empty([nR,nt])

i=0
for filename in infiles:
    with open(filename) as f:
        firstline = f.readline()
#    print (firstline) Need to obtain the exact time from this line
    fileData = np.loadtxt(filename)
    sigma[:,i] = fileData[:,2]
    Tc[:,i] = fileData[:,3]
    nu[:,i] = fileData[:,7]
    Q[:,i] = fileData[:,8]
    i=i+1

#print (nR, nt)
#print (sigma)
R=radius
t=arange(0,nt*500/1.0e6,500/1e6)
#print  ( x.min(), x.max(), y.min(), y.max())

print (R.size, t.size, sigma.size)




#----------------------------------------------------------------------------
# Make plots
#----------------------------------------------------------------------------

# 1.Sigma
plt.subplot(4,1,1)
ima1 = plt.pcolor(t,R,sigma, cmap='magma', norm=LogNorm())
if (autorange==0):
    axis([tmin,tmax,Rmin,Rmax])
if (logscaleR==1):
    plt.yscale('log')
cbar1=plt.colorbar(ima1, orientation='vertical',format=LogFormatterMathtext())
cbar1.set_label('Sigma (g/cm2)')
plt.clim(100,5e4)


# 2.Tc
plt.subplot(4,1,2)
ima2 = plt.pcolor(t,R, Tc, cmap='magma')
if (autorange==0):
    axis([tmin,tmax,Rmin,Rmax])
if (logscaleR==1):
    plt.yscale('log')
cbar2=plt.colorbar(ima2, orientation='vertical')
cbar2.set_label('Tc (K)')
plt.clim(150,2000)


# 3.nu
plt.subplot(4,1,3)
ima3 = plt.pcolor(t,R,nu, cmap='magma', norm=LogNorm())
if (autorange==0):
    axis([tmin,tmax,Rmin,Rmax])
if (logscaleR==1):
    plt.yscale('log')
cbar3=plt.colorbar(ima3, orientation='vertical',format=LogFormatterMathtext())
cbar3.set_label('nu')


# 3.Q
plt.subplot(4,1,4)
ima4 = plt.pcolor(t,R,Q, cmap='magma')
if (autorange==0):
    axis([tmin,tmax,Rmin,Rmax])
if (logscaleR==1):
    plt.yscale('log')
cbar4=plt.colorbar(ima4, orientation='vertical')
cbar4.set_label('Q')
plt.clim(0,40)

plt.show()










