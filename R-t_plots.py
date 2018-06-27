#****************************************************************************
# Description:
# Calculates R vs time plots for fields- Sigma, Tc, nu, Q -as the color info 
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

tmin = 0.275
tmax = 0.325
Rmin = 0.02
Rmax = 5
autorange = 1    # 1 => whole dataset range, 0 => ranges spacified above
logscaleR = 0    # 1 => logscale for R direction
reread = 0       # 1 => loading from the saved data in COLORED directory
                 # 0 => has to reorganize the simulation files before plotting
outdir = "COLORED"

if (reread == 0):
# Get input file names
    fileslist = glob.glob('a*.txt')
    fileslist.sort()
    infiles = np.hstack(fileslist)

    nt = infiles.size

# Make output directory
    if not os.path.exists(outdir):
        os.makedirs(outdir)

# Get the radial array
    fileData = np.loadtxt(fileslist[0])
    radius = fileData[:,1]

    nR = radius.size

# Initialize arrays
    sigma = np.empty([nR,nt])
    Tc = np.empty([nR,nt])
    nu = np.empty([nR,nt])
    Q = np.empty([nR,nt])
    time = np.empty([nt])

# Make Sigma, Tc, nu, Q arrays
    i=0
    for filename in infiles:
        with open(filename) as f:
            firstline = f.readline()
            time[i] = float(firstline[15:24]) 
        fileData = np.loadtxt(filename)
        sigma[:,i] = fileData[:,2]
        Tc[:,i] = fileData[:,3]
        nu[:,i] = fileData[:,7]
        Q[:,i] = fileData[:,8]
        i=i+1

    R=radius


# Save arrays in files for future use 
    np.savetxt(outdir+'/sigma', sigma, fmt='%d')
    np.savetxt(outdir+'/Tc', Tc, fmt='%d')
    np.savetxt(outdir+'/nu', nu, fmt='%d')
    np.savetxt(outdir+'/Q', Q, fmt='%d')
    np.savetxt(outdir+'/R', R, fmt='%d')
    np.savetxt(outdir+'/time', time, fmt='%d')

elif (reread == 1):

    sigma = np.loadtxt(outdir+'/sigma')
    Tc = np.loadtxt(outdir+'/Tc')
    nu = np.loadtxt(outdir+'/nu')
    Q = np.loadtxt(outdir+'/Q')
    R = np.loadtxt(outdir+'/R')
    time = np.loadtxt(outdir+'/time')

#----------------------------------------------------------------------------
# Make plots
#----------------------------------------------------------------------------
#print  ( x.min(), x.max(), y.min(), y.max())
#print (R.size, t.size, sigma.size)


t=time

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

# Save and show the plot
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(10, 14)
savefig(outdir+'/R-t.png', bbox_inches='tight')

plt.show()










