import numpy as np
import matplotlib.pyplot as plt
from sWeightsGeneral import calc_sWeights
from GenerateData import genData
from scipy.optimize import minimize
from scipy.stats import poisson

# Import your data:

Data = genData(Dim='1D')

# bins for plotting
bins = np.linspace(0.0,2.0,101)
# steps in the histogram (Value at the center of each bin)
steps = bins[0:100]+(bins[1]-bins[0])/2.0
# binwidth, needed for normalization
binwidth = (bins[-1] - bins[0]) / (len(bins)-1)

# Fit a PDF to the data using a likelihood fit (Binned likelihood here):
# -------------------------------------------------------------------------- #

# Define the binned likelihood function, which includes a exponential background and gaussian signal in this case
def Binlike_GaussExp(p,x):
    m = 0
    for i in range(0,len(x)):
        m += -2.0*np.log(poisson.pmf(x[i],((p[0]*binwidth*np.exp(-0.5*(steps[i] - p[1])**2/(p[2]**2))/
            np.sqrt(2.0*np.pi*p[2]**2))+p[3]*binwidth*p[4]*np.exp(steps[i]*-p[4]))))
    return m

# Define the same function in a general expression (just for plotting the PDF using best fit values)
def func_GaussExp(p,x) :
    zsig = (x - p[1]) / p[2]
    sig = p[0]*binwidth / np.sqrt(2.0*np.pi)/p[2] * np.exp(-0.5*zsig*zsig)
    bkg = p[3]*binwidth*p[4] * np.exp(-p[4] * x)
    return sig+bkg

# Plot data in a histogram
plt.figure()
hist1 = plt.hist(Data[:,0],bins=bins,stacked=False,histtype='step',linewidth=2.0,color='orange',label='Signal + Background')
plt.title('Data + best fit')
plt.xlabel('Mass')
plt.ylabel('Frequency')

# initial guess on best values
init_pars1 = [5000,0.8,0.2,10000,2.0]
# Minimize the binned likelihood function to obtain the best fit values
fit_mass = minimize(Binlike_GaussExp, init_pars1, args=(hist1[0],))
print('1d fit parameters:')
print(fit_mass.x)

# Plot the PDF using best fit values
plt.plot(bins,func_GaussExp(fit_mass.x,bins),color='black',label='Best fit')
plt.legend()
plt.show()

# Define functions describing signal and background in data:
# -------------------------------------------------------------------------- #

# Signal PDF
def fsig(x,p):
    z0sig = (x - p[1]) / p[2]
    return binwidth*p[0] / np.sqrt(2.0*np.pi)/p[2] * np.exp(-0.5*z0sig*z0sig)
    
# Background PDF
def fbkg(x,p):
    return binwidth*p[0] * p[1] * np.exp(-p[1]*x)

# Run the sWeights script using the best fit values as your {pars} input:
# Notice how the values are seperated in signal and background in {pars}
pars = ([fit_mass.x[0],fit_mass.x[1],fit_mass.x[2]]
        ,[fit_mass.x[3],fit_mass.x[4]])
calc_sWeights((fsig,fbkg),pars,Data[:,0],Data[:,-1],plot='True')

# Look in the General sWeights script to find information about input of the sWeights function














