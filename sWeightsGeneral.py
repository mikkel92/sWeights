# F_sig: The function defining the signal pdf. Assumues the first parameter (p[0]) 
# to be the normalization. Needs to be of the form: f(x,p). 
# Exponential example: f(x,p) = p[0]*p[1]*exp(-p[1]*x[0])

# F_bkg: The function defining the background pdf. Assumues the first parameter (p[0]) 
# to be the normalization. Needs to be of the form: f(x,p).

# x: Data. x has to be an N*M+1 ndarray, N being the number of data points and M
# being the number of dimensions. The last column is the data which needs to be seperated.

# pars: Parameters found by fitting the input functions to data. Has to be an N*2 ndarray,
# N being the number of parameters for either function.

# bins: The number of bins wanted on the plot.

def sWeights(F_sig,F_bkg,data,pars,bins):
    
    import numpy as np
    from numpy import matrix
    import parser
    import matplotlib.pyplot as plt
    
    Bins = []
    Steps = []
    Less_Bins = False
    New_Bins = []
    for i in range(0,np.shape(data)[1]):
        if len(np.unique(data[:,i])) <= bins:
            Less_Bins = True
            New_Bins.append(len(np.unique(data[:,i])))
    
    if Less_Bins == True:
        print "Number of bins reduced: More bins than number of different values in data. New number of bins: %i" % min(New_Bins)
       
        
    for i in range(0,np.shape(data)[1]):
        if Less_Bins == True:
            Bins.append(np.linspace(min(data[:,i]),max(data[:,i]),(min(New_Bins)+1)))
            Steps.append(Bins[i][0:min(New_Bins)]+(Bins[i][1]-Bins[i][0])/2.0)
        else:
            Bins.append(np.linspace(min(data[:,i]),max(data[:,i]),(bins+1)))
            Steps.append(Bins[i][0:bins]+(Bins[i][1]-Bins[i][0])/2.0)
    
    
    iclist = [ [0.0, 0.0], [0.0, 0.0] ]
    
    N_data = ''
    for i in range(0,np.shape(data)[1]-1):
        N_data += 'data[i][%d], ' %(i)
        
    sigPDF = 'F_sig( [%s] , pars[0] ) / pars[0][0]' % N_data
    sigPDF = parser.expr(sigPDF).compile()
    bkgPDF = 'F_bkg( [%s] , pars[1] ) / pars[1][0]' % N_data
    bkgPDF = parser.expr(bkgPDF).compile()
    
        
    for i in range(0,len(data)) :
        sig_PDF = eval(sigPDF)
        bkg_PDF = eval(bkgPDF)
        denominator = ( pars[0][0]*sig_PDF + pars[1][0]*bkg_PDF )**2
        iclist[0][0] += sig_PDF*sig_PDF / denominator
        iclist[0][1] += sig_PDF*bkg_PDF / denominator
        iclist[1][0] += bkg_PDF*sig_PDF / denominator
        iclist[1][1] += bkg_PDF*bkg_PDF / denominator
    invcovmat = matrix( [ [iclist[0][0], iclist[0][1]], [iclist[1][0], iclist[1][1]] ] )

    covmat = invcovmat.I
    print "Covarance matrix as obtained from direct calculation: "
    print covmat
    
    sWeight = []
    bWeight = []
    
    for i in range(0, np.shape(data)[0]) :
        
        sig_PDF = eval(sigPDF)
        bkg_PDF = eval(bkgPDF)
        sWeight.append((covmat.item(0,0) * sig_PDF + covmat.item(0,1) * bkg_PDF)  / (pars[0][0]*sig_PDF + pars[1][0]*bkg_PDF))
        bWeight.append((covmat.item(1,0) * sig_PDF + covmat.item(1,1) * bkg_PDF) / (pars[0][0]*sig_PDF + pars[1][0]*bkg_PDF))

    # Calculate the errors
    err_sW = np.zeros(len(Steps[-1]))
    err_bW = np.zeros(len(Steps[-1]))
    for j in range(0,len(Steps[-1])):    
        for i in range(0,len(data)):
            if Bins[-1][j] <= data[i][-1] < Bins[-1][j+1]:
                err_sW[j] += sWeight[i]**2
                err_bW[j] += bWeight[i]**2

    err_sW = np.sqrt(err_sW)
    err_bW = np.sqrt(err_bW)

    plt.figure(1,figsize=(16,7))
    plt.subplot(221)
    hist1 = np.histogram(data[:,-1],weights=sWeight,bins=Bins[-1])
    hist2 = plt.hist(data[:,-1],bins=Bins[-1],stacked=False,histtype='step',linewidth=2.0,color='orange',label='Data')
    plt.errorbar(Steps[-1], hist1[0],yerr=err_sW,fmt='.',capsize=3,capthick=2,color='r',label='Signal')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()

    plt.subplot(222)
    hist3 = np.histogram(data[:,-1],weights=bWeight,bins=Bins[-1])
    hist4 = plt.hist(data[:,-1],bins=Bins[-1],stacked=False,histtype='step',linewidth=2.0,color='orange',label='Data')
    plt.errorbar(Steps[-1], hist3[0],yerr=err_bW,fmt='.',capsize=3,capthick=2,color='b',label='Background')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()

    plt.subplot(223)
    hist5 = plt.hist(data[:,-1],bins=Bins[-1],stacked=False,histtype='step',linewidth=2.0,color='orange',label='Data')
    plt.errorbar(Steps[-1], hist1[0],yerr=err_sW,fmt='.',capsize=3,capthick=2,color='r',label='Signal')
    plt.errorbar(Steps[-1], hist3[0],yerr=err_bW,fmt='.',capsize=3,capthick=2,color='b',label='Background')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show(block=False)
    
    raw_input( ' ... ' )
    plt.close('all')
    return #(sWeight,bWeight)


# ------------------------------------------------------------------------ #
if __name__ == '__sWeights__':
    sys.exit( main() )

import numpy as np

def func_GEGG_sig(x,p) :
    z0sig = (x[0] - p[1]) / p[2]
    z1sig = (x[1] - p[3]) / p[4]
    return 0.06*0.02*p[0] / np.sqrt(2.0*np.pi)/p[2] * np.exp(-0.5*z0sig*z0sig) / np.sqrt(2.0*np.pi)/p[4] * np.exp(-0.5*z1sig*z1sig)

def func_GEGG_bkg(x,p) : 
    z1bkg = (x[1] - p[2]) / p[3]
    return 0.06*0.02*p[0] * p[1] * np.exp(-p[1]*x[0]) / np.sqrt(2.0*np.pi)/p[3] * np.exp(-0.5*z1bkg*z1bkg)

def fsig(x,p):
    z0sig = (x[0] - p[1]) / p[2]
    return 0.02*p[0] / np.sqrt(2.0*np.pi)/p[2] * np.exp(-0.5*z0sig*z0sig)

def fbkg(x,p):
    return 0.02*p[0] * p[1] * np.exp(-p[1]*x[0])

def sig2(x,p):
    return p[0] * p[1] * np.exp(-p[1]*x[0])


Data = np.loadtxt('output.txt', dtype='float')
pars = ([5000.0,0.8,0.2,0.0,0.4],[10000.0,2.0,-1.0,0.6])
sWeights(func_GEGG_sig,func_GEGG_bkg,Data,pars,50)

pars = ([5000.0,0.8,0.2],[10000.0,2.0])
sWeights(fsig,fbkg,Data[:,0:2],pars,50)

Data = np.loadtxt('16var.txt', dtype='float')
pars = ([10000.0,2.0],[10000.0,2.5])
sWeights(sig2,sig2,Data[:,3:17],pars,50)
