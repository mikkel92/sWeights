import numpy as np
import matplotlib.pyplot as plt
from GenerateData import genData
from sWeightsGeneral import calc_sWeights
from scipy.optimize import minimize
import math
import warnings

warnings.filterwarnings("ignore")

# Generate your data: 
Data = genData(Dim='2D')

# bins for plotting
bins1 = np.linspace(0.0,2.0,101)
bins2 = np.linspace(-4.0,2.0,101)

# steps in the histogram (Value at the center of each bin)
steps1 = bins1[0:100]+(bins1[1]-bins1[0])/2.0
steps2 = bins2[0:100]+(bins2[1]-bins2[0])/2.0

# binwidth, needed for normalization
binwidth0 = (bins1[-1] - bins1[0]) / (len(bins1)-1)
binwidth1 = (bins2[-1] - bins2[0]) / (len(bins2)-1)
binarea = binwidth0 * binwidth1

# Fit a PDF to the data using a likelihood fit (Binned likelihood here):
# -------------------------------------------------------------------------- #

# poisson used for binned likelihood
def pois(n,x):
    return x**n*np.exp(-x)/math.factorial(n)

# Define the binned likelihood function, which includes a exponential background and gaussian 
# signal in one dimension and a gaussian background + signal in the other dimension in this case:
def BinLike_GEGG(p,x):
    m = 0
    for i in range(0,len(x)):
        for j in range(0,len(x)):
            z0sig = (steps1[j] - p[1]) / p[2]
            z1sig = (steps2[i] - p[3]) / p[4]
            sig = p[0]*binarea / np.sqrt(2.0*np.pi)/p[2] * np.exp(-0.5*z0sig*z0sig) / np.sqrt(2.0*np.pi)/p[4] * np.exp(-0.5*z1sig*z1sig)
            z1bkg = (steps2[i] - p[7]) / p[8]
            bkg = p[5]*binarea * p[6] * np.exp(-p[6]*steps1[j]) / np.sqrt(2.0*np.pi)/p[8] * np.exp(-0.5*z1bkg*z1bkg)
            m += -2.0*np.log(pois(x[i][j],(sig+bkg)))
    return m

# Define the same function in a general expression (just for plotting the PDF using best fit values)
def func_GEGG(p,x) :
    z0sig = (x[0] - p[1]) / p[2]
    z1sig = (x[1] - p[3]) / p[4]
    sig = p[0]*binarea / np.sqrt(2.0*np.pi)/p[2] * np.exp(-0.5*z0sig*z0sig) / np.sqrt(2.0*np.pi)/p[4] * np.exp(-0.5*z1sig*z1sig)
    z1bkg = (x[1] - p[7]) / p[8]
    bkg = p[5]*binarea * p[6] * np.exp(-p[6]*x[0]) / np.sqrt(2.0*np.pi)/p[8] * np.exp(-0.5*z1bkg*z1bkg)
    return sig+bkg

# Define the steps for a 2D plot (to see if your fit values are decent)
x, y = np.meshgrid(steps1, steps2)
hist1 = np.histogram2d(Data[:,0], Data[:,1], bins=(bins1, bins2))
hist1 = hist1[0].transpose() # Numpy lists the data from a wierd starting point


# initial guess on best values
init_pars = (5000.0,0.8,0.2,0.0,0.4,10000.0,2.0,-1.0,0.6)
# Minimize the binned likelihood function to obtain the best fit values
fit_2d = minimize(BinLike_GEGG, init_pars, args=(hist1,), method='SLSQP')
fit_2d = fit_2d.x
print "2d fit parameters:"
print(fit_2d)

# Plot the data
plt.imshow(hist1, interpolation='none', origin='left', extent=[bins1[0], bins1[-1], bins2[0], bins2[-1]], aspect='auto',cmap='terrain')
plt.colorbar()
plt.title('Data + best fit (shown as contours)')
plt.xlabel('Mass')
plt.ylabel('Shape')

# Plot the PDF using best fit values
Cont = func_GEGG( fit_2d, (x,y)).reshape(100,100)
plt.contour(x,y, Cont, extent=[bins1[0], bins1[-1], bins2[0], bins2[-1]], aspect='auto', colors='k')
plt.legend()
plt.show()

# Define functions describing signal and background in data:
# -------------------------------------------------------------------------- #

# Signal PDF
def func_GEGG_signal(x,p) :
    z0sig = (x[0] - p[1]) / p[2]
    z1sig = (x[1] - p[3]) / p[4]
    return p[0] / np.sqrt(2.0*np.pi)/p[2] * np.exp(-0.5*z0sig*z0sig) / np.sqrt(2.0*np.pi)/p[4] * np.exp(-0.5*z1sig*z1sig)

# Background PDF
def func_GEGG_background(x,p) :
    z1bkg = (x[1] - p[2]) / p[3]
    return p[0] * p[1] * np.exp(-p[1]*x[0]) / np.sqrt(2.0*np.pi)/p[3] * np.exp(-0.5*z1bkg*z1bkg)

# Run the sWeights script using the best fit values as your {pars} input:
# Notice how the values are seperated in signal and background in {pars}
pars = ([fit_2d[0],fit_2d[1],fit_2d[2],fit_2d[3],fit_2d[4]]
        ,[fit_2d[5],fit_2d[6],fit_2d[7],fit_2d[8]])

calc_sWeights((func_GEGG_signal,func_GEGG_background),pars,Data[:,0:2],Data[:,-1],plot='True',bins=50)

# Look in the General sWeights script to find information about input of the sWeights function

