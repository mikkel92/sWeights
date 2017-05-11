from array import array
import sys
import numpy as np
from scipy import stats as stats
from scipy.optimize import minimize
from scipy.stats import poisson
from numpy import matrix
import matplotlib.pyplot as plt
import math
import warnings

warnings.filterwarnings("ignore")

np.random.seed(1)

# Number of events in total:
Nsig =  5000
Nbkg = 10000
mean_sig1 = 0.8
width_sig1 = 0.2
mean_sig2 = 0.0
width_sig2 = 0.4
exp_bkg1 = 0.5
mean_bkg2 = -1.0
width_bkg2 = 0.6


SavePlots = False    
verbose = True
Nverbose = 10
RunFast = False   # Fix PDF shape parameters to truth values!


# Beta distribution: f(x) = (1 + a*x + b*x^2) / (2 + 2b/3) for x in [-1,1]
# ----------------------------------------------------------------------------------- #
def betadist(x,a,b) :
    if (abs(x) <= 1.0) : return (1.0 + a*x + b*x*x) / (2.0 + 2.0*b/3.0)
    else               : return 0.0

# Oscillation distribution: f(x) = 1 + A*cos(omega*x + phi) for x in [-1,1]
# ----------------------------------------------------------------------------------- #
def oscdist(x, A, omega, phi) :
    if (abs(A) <= 1.0) : return 1.0 + A*np.cos(omega*x + phi)
    else               : return 0.0


# Function for generating numbers according to beta distributions:
# ----------------------------------------------------------------------------------- #
def GetAngle( alpha, beta ) :
    if (alpha+beta > 1.0) :
        print("ERROR: alpha and/or beta not in defined range: alpha = %6.2f   beta = %6.2f" % (alpha,beta))
        return -999
    x = -1.0 + 2.0 * np.random.rand()
    y = (1.0+abs(alpha)+abs(beta)) * np.random.rand()
    while (y > betadist(x,alpha,beta)) :
        x = -1.0 + 2.0 * np.random.rand()
        y = (1.0+abs(alpha)+abs(beta)) * np.random.rand()
    return x

# Function for generating numbers according to oscillation distributions:
# ----------------------------------------------------------------------------------- #
def GetAngle2( A, omega, phi ) :
    if (abs(A) > 1.0) :
        print("ERROR: A not in defined range: A = %6.2f" % (A))
        return -999
    x = -1.0 + 2.0 * np.random.rand()
    y = (1.0+abs(A)) * np.random.rand()
    while (y > oscdist(x, A, omega, phi)) :
        x = -1.0 + 2.0 * np.random.rand()
        y = (1.0+abs(A)) * np.random.rand()
    return x

# ------------------------------------------------------------------------ #
def main() : 
# ------------------------------------------------------------------------ #

	# Generate data
    x_sig = []
    x_bkg = []

    for isig in range(0, Nsig ) :
        x_sig.append( [ np.random.randn()*width_sig1+mean_sig1, np.random.randn()*width_sig2+mean_sig2, GetAngle2(0.9, 12.0, 1.0) ] )
    x_sig = np.array(x_sig)


    for ibkg in xrange ( Nbkg ) :
        x_bkg.append( [ np.random.exponential(exp_bkg1, ), np.random.randn()*width_bkg2+mean_bkg2, GetAngle2( 0.8, 17.0, 0.5 ) ] )
    x_bkg = np.array(x_bkg)
    x_all = np.concatenate((x_sig,x_bkg), axis=0)

    # Use data from a file? (Outcomment if you want to generate your own data)
    Data = np.loadtxt('output.txt', dtype='float')

    x_sig = Data[0:5000]
    x_bkg = Data[5000:15000]
    x_all = Data

    #np.savetxt('sWout.txt', x_all, fmt='%8.4f')

    # Functions for fitting
    # ----------------------------------------------------------------------------------- #

    # Defining bins for various histograms
    bins1 = np.linspace(0.0,2.0,101)
    bins2 = np.linspace(-4.0,2.0,101)
    bins3 = np.linspace(-1.0,1.0,101)
    bins4 = np.linspace(-0.5,1.5,201)

    steps1 = bins1[0:100]+(bins1[1]-bins1[0])/2.0
    steps2 = bins2[0:100]+(bins2[1]-bins2[0])/2.0

    binwidth0 = (bins1[-1] - bins1[0]) / (len(bins1)-1)
    binwidth1 = (bins2[-1] - bins2[0]) / (len(bins2)-1)
    binarea = binwidth0 * binwidth1

    def pois(n,x):
        return x**n*np.exp(-x)/math.factorial(n)

    def func_GaussExp(p,x) :
        zsig = (x - p[1]) / p[2]
        sig = p[0]*binwidth0 / np.sqrt(2.0*np.pi)/p[2] * np.exp(-0.5*zsig*zsig)
        bkg = p[3]*binwidth0*p[4] * np.exp(-p[4] * x)
        return sig+bkg

    def Binlike_GaussExp(p,x):
        m = 0
        for i in range(0,len(x)):
            m += -2.0*np.log(poisson.pmf(x[i],((p[0]*binwidth0*np.exp(-0.5*(steps1[i] - p[1])**2/(p[2]**2))/
                np.sqrt(2.0*np.pi*p[2]**2))+p[3]*binwidth0*p[4]*np.exp(steps1[i]*-p[4]))))
        return m

    def func_GaussGauss(p,x) :
        zsig = (x - p[1]) / p[2]
        sig = p[0]*binwidth1 / np.sqrt(2.0*np.pi)/p[2] * np.exp(-0.5*zsig*zsig)
        zbkg = (x - p[4]) / p[5]
        bkg = p[3]*binwidth1 / np.sqrt(2.0*np.pi)/p[5] * np.exp(-0.5*zbkg*zbkg)
        return sig+bkg

    def BinLike_GaussGauss(p,x):
        m = 0
        for i in range(0,len(x)):
            m += -2.0*np.log(poisson.pmf(x[i],((p[0]*binwidth1*np.exp(-0.5*(steps2[i] - p[1])**2/(p[2]**2))/np.sqrt(2.0*np.pi*p[2]**2))+
                                  (p[3]*binwidth1*np.exp(-0.5*(steps2[i] - p[4])**2/(p[5]**2))/np.sqrt(2.0*np.pi*p[5]**2)))))
        return m

    def func_GEGG(p,x) :
        z0sig = (x[0] - p[1]) / p[2]
        z1sig = (x[1] - p[3]) / p[4]
        sig = p[0]*binarea / np.sqrt(2.0*np.pi)/p[2] * np.exp(-0.5*z0sig*z0sig) / np.sqrt(2.0*np.pi)/p[4] * np.exp(-0.5*z1sig*z1sig)
        z1bkg = (x[1] - p[7]) / p[8]
        bkg = p[5]*binarea * p[6] * np.exp(-p[6]*x[0]) / np.sqrt(2.0*np.pi)/p[8] * np.exp(-0.5*z1bkg*z1bkg)
        return (sig+bkg).ravel()

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


    # Plotting and fitting
    # ----------------------------------------------------------------------------------- #

    plt.figure(1,figsize=(16,7))
    plt.subplot(221)
    hist1 = plt.hist(x_sig[:,0],bins=bins1,stacked=False,histtype='step',linewidth=2.0,color='r',label='signal')
    hist2 = plt.hist(x_bkg[:,0],bins=bins1,stacked=False,histtype='step',linewidth=2.0,color='b',label='background')
    hist3 = plt.hist(x_all[:,0],bins=bins1,stacked=False,histtype='step',linewidth=2.0,color='orange',label='combined')
    plt.xlabel('Mass')
    plt.ylabel('Frequency')
    plt.legend()

    # Fit mass dimension:
    init_pars1 = [5000,0.8,0.2,10000,2.0]
    fit_mass = minimize(Binlike_GaussExp, init_pars1, args=(hist3[0],))
    plt.plot(bins1,func_GaussExp(fit_mass.x,bins1),color='black')

    print "Gauss + Exp fit parameters: "
    print(fit_mass.x)

    plt.subplot(222)
    hist4 = plt.hist(x_sig[:,1],bins=bins2,stacked=False,histtype='step',linewidth=2.0,color='r',label='signal')
    hist5 = plt.hist(x_bkg[:,1],bins=bins2,stacked=False,histtype='step',linewidth=2.0,color='b',label='background')
    hist6 = plt.hist(x_all[:,1],bins=bins2,stacked=False,histtype='step',linewidth=2.0,color='orange',label='combined')
    plt.xlabel('Shape')
    plt.ylabel('Frequency')
    plt.legend()

    # Fit shape dimension:
    
    init_pars2 = [5000,0.0,0.4,10000,-1.0,0.6]
    fit_shape = minimize(BinLike_GaussGauss, init_pars2, args=(hist6[0],))
    plt.plot(bins2,func_GaussGauss(fit_shape.x,bins2),color='black')

    print "Gauss + Gauss fit parameters: "
    print(fit_shape.x)

    plt.subplot(223)
    hist7 = plt.hist(x_sig[:,2],bins=bins3,stacked=False,histtype='step',linewidth=2.0,color='r',label='signal')
    hist8 = plt.hist(x_bkg[:,2],bins=bins3,stacked=False,histtype='step',linewidth=2.0,color='b',label='background')
    hist9 = plt.hist(x_all[:,2],bins=bins3,stacked=False,histtype='step',linewidth=2.0,color='orange',label='combined')
    plt.xlabel('Angle')
    plt.ylabel('Frequency')
    plt.legend()

    # Fit the signal and background samples in 2D (mass and shape):
    x, y = np.meshgrid(steps1, steps2)
    hist10 = np.histogram2d(x_all[:,0], x_all[:,1], bins=(bins1, bins2))
    dat = (hist10[0].transpose()).ravel() # curve_fit takes a M*N length vector instead of a (M,N) matrix
    dat2 = hist10[0].transpose() # minimize doesn't

    init_pars = (Nsig,mean_sig1,width_sig1,mean_sig2,width_sig2,Nbkg,1.0/exp_bkg1,mean_bkg2,width_bkg2)

    if (RunFast) :     # For debugging it is worthwhile skipping a 9 parameter 
                       # binned likelihood fit in 2D!!! Uses curve_fit instead(a non-linear least squares)
        fit_massshape = curve_fit(func_GEGG, (x, y), dat, p0=init_pars)
        covmatfit = matrix( [ [fit_massshape[1][0][0], fit_massshape[1][0][5]], [fit_massshape[1][5][0], fit_massshape[1][5][5] ] ] )
        
        fit_massshape = fit_massshape[0]
        print "2d fit parameters:"
        print(fit_massshape)
    else: # The fit with all parameters
        fit_massshape = minimize(BinLike_GEGG, init_pars, args=(dat2,), method='SLSQP', options={'disp':True})
        fit_massshape = fit_massshape.x
        print "2d fit parameters:"
        print(fit_massshape)


    plt.subplot(224)
    plt.imshow(hist10[0].transpose(), interpolation='none', origin='left', extent=[bins1[0], bins1[-1], bins2[0], bins2[-1]], aspect='auto',cmap='terrain')
    plt.colorbar()
    plt.xlabel('Mass')
    plt.ylabel('Shape')

    if (RunFast) :
        Cont = func_GEGG((x,y), fit_massshape[0],fit_mass.x[1],fit_mass.x[2],fit_shape.x[1],fit_shape.x[2],fit_massshape[5],fit_mass.x[4],fit_shape.x[4],fit_shape.x[5]).reshape(100,100)
    else:
        Cont = func_GEGG( fit_massshape, (x,y)).reshape(100,100)
    plt.contour(x,y, Cont, extent=[bins1[0], bins1[-1], bins2[0], bins2[-1]], aspect='auto', colors='k')

    # -----------------------------------------------------------------------------------
    # Given succesful fit, calculate sWeights and produce signal distribution of "angle":
    # -----------------------------------------------------------------------------------


    def func_GEGG_signal(x) :
            z0sig = (x[0] - fit_massshape[1]) / fit_massshape[2]
            z1sig = (x[1] - fit_massshape[3]) / fit_massshape[4]
            return fit_massshape[0] / np.sqrt(2.0*np.pi)/fit_massshape[2] * np.exp(-0.5*z0sig*z0sig) / np.sqrt(2.0*np.pi)/fit_massshape[4] * np.exp(-0.5*z1sig*z1sig)

    def func_GEGG_background(x) :
        z1bkg = (x[1] - fit_massshape[7]) / fit_massshape[8]
        return fit_massshape[5] * fit_massshape[6] * np.exp(-fit_massshape[6]*x[0]) / np.sqrt(2.0*np.pi)/fit_massshape[8] * np.exp(-0.5*z1bkg*z1bkg)

    # Covariance matrix (inverse to be inverted), calculated from sum over all events:
    iclist = [ [0.0, 0.0], [0.0, 0.0] ]
    for i in xrange ( len(x_all) ) :
        sigPDF = func_GEGG_signal( [x_all[i][0], x_all[i][1] ] ) / fit_massshape[0]
        bkgPDF = func_GEGG_background( [x_all[i][0], x_all[i][1] ] ) / fit_massshape[5]
        denominator = ( fit_massshape[0]*sigPDF + fit_massshape[5]*bkgPDF )**2
        iclist[0][0] += sigPDF*sigPDF / denominator
        iclist[0][1] += sigPDF*bkgPDF / denominator
        iclist[1][0] += bkgPDF*sigPDF / denominator
        iclist[1][1] += bkgPDF*bkgPDF / denominator
    invcovmat = matrix( [ [iclist[0][0], iclist[0][1]], [iclist[1][0], iclist[1][1]] ] )
    # print "Inverse covarance matrix from direct calculation:  "
    # print invcovmat

    covmat = invcovmat.I
    print "Covarance matrix as obtained from direct calculation: "
    print covmat

    print "Number of signal events:      %8.1f  (NsigTrue = %5d)"%(fit_massshape[0], Nsig)
    print "Number of background events:  %8.1f  (NbkgTrue = %5d)"%(fit_massshape[5], Nbkg)

    sWeight = []
    bWeight = []

    for i in range(0, len(x_all)) :
        sigPDF = func_GEGG_signal( [x_all[i][0], x_all[i][1] ] ) / fit_massshape[0]
        bkgPDF = func_GEGG_background( [x_all[i][0], x_all[i][1] ] ) / fit_massshape[5]
        sWeight.append((covmat.item(0,0) * sigPDF + covmat.item(0,1) * bkgPDF)  / (fit_massshape[0]*sigPDF + fit_massshape[5]*bkgPDF))
        bWeight.append((covmat.item(1,0) * sigPDF + covmat.item(1,1) * bkgPDF) / (fit_massshape[0]*sigPDF + fit_massshape[5]*bkgPDF))
    
    # Save sWeights and bWeights to a file
    sb = np.column_stack((sWeight,bWeight))
    np.savetxt('sbW1.txt', sb, fmt='%8.4f')

    # Calculate the errors
    err_sW = np.zeros(len(steps1))
    err_bW = np.zeros(len(steps1))
    for j in range(0,len(steps1)):    
        for i in range(0,len(x_all)):
            if bins3[j] <= x_all[i][2] < bins3[j+1]:
                err_sW[j] += sWeight[i]**2
                err_bW[j] += bWeight[i]**2

    err_sW = np.sqrt(err_sW)
    err_bW = np.sqrt(err_bW)

    # Plot weights for signal and background events:
    # ----------------------------------------------------------------------------------- #

    plt.figure(2,figsize=(16,7))
    plt.subplot(221)
    plt.hist(sWeight[0:5000],bins=bins4,stacked=False,histtype='step',linewidth=2.0,color='r')
    plt.xlabel('Sweight (for signal events)')
    plt.ylabel('Frequency')

    plt.subplot(222)
    plt.hist(bWeight[5000:15000],bins=bins4,stacked=False,histtype='step',linewidth=2.0,color='b')
    plt.xlabel('bWeight (for background event)')
    plt.ylabel('Frequency')

    plt.subplot(223)
    hist11 = np.histogram(x_all[:,2],weights=sWeight,bins=bins3)
    hist7 = plt.hist(x_sig[:,2],bins=bins3,stacked=False,histtype='step',linewidth=2.0,color='orange',label='signal')
    plt.errorbar(bins3[0:100]+0.01, hist11[0],yerr=np.sqrt(hist11[0]),fmt='.',capsize=3,capthick=2,color='r')
    plt.xlabel('Signal Angle')
    plt.ylabel('Frequency')

    plt.subplot(224)
    hist12 = np.histogram(x_all[:,2],weights=bWeight,bins=bins3)
    hist8 = plt.hist(x_bkg[:,2],bins=bins3,stacked=False,histtype='step',linewidth=2.0,color='orange',label='background')
    plt.errorbar(bins3[0:100]+0.01, hist12[0],yerr=np.sqrt(hist12[0]),fmt='.',capsize=3,capthick=2,color='b')
    plt.xlabel('Background Angle')
    plt.ylabel('Frequency')
    plt.show(block=False)

    chi1 = stats.chisquare(hist11[0],hist7[0])
    chi2 = stats.chisquare(hist12[0],hist8[0])
    ks1 = stats.ks_2samp(hist11[0],hist7[0])
    ks2 = stats.ks_2samp(hist12[0],hist8[0])
    p1 = 1 - stats.chi2.cdf(chi1[0],df=99)
    p2 = 1 - stats.chi2.cdf(chi2[0],df=99)

    print "Prob-Signal:    Prob-Background:"
    print(chi1[1],chi2[1])
    print(ks1[1],ks2[1])
    print(p1,p2)

    
    raw_input( ' ... ' )
    return 


# ------------------------------------------------------------------------ #
if __name__ == '__main__':
    sys.exit( main() )






    
    
    
    
    