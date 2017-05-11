# Functions: A list with functions. The PDFs defining the different signal/backgrounds in data. 
# Assumues the first parameter (p[0]) to be the normalization. Needs to be of the form: f(x,p). 
# Exponential example: f(x,p) = p[0]*p[1]*exp(-p[1]*x[0]), p[1] = 2.0; 
# or f(x,p) = p[0]*2.0*exp(-2.0*x[0]). Only p[0] has to be a variable and inserted via {pars}.

# data: Has to be an N*M+1 ndarray, N being the number of data points and M
# being the number of dimensions. The last column is the data which needs to be seperated.

# pars: Parameters found by fitting the input functions to data. Has to be an N*M ndarray,
# N being the number of parameters for either function and M being the number of dimensions.

# plot: If plot='True' plots the last column in data in a histogram and shows the individual contributions
# from functions on individual plots + individual contributions on a combined plot. 
# If plot='Combined' only the combined plot is showed.

# bins: The number of bins wanted on the plot and in the error calculation.

# write: If write='True' the sWeights are saved in a text file in the format N*M. N is the number of data points
# and M is the number of signal types

# save_err: If save_err='True' the errors calculated are saved to a text file in the format N*M
# N is the number if bins chosen and M is the number of signal types

def sWeights(Functions,data,pars,plot=False,bins=None,write=False,save_err=False):
    
    import numpy as np
    from numpy import matrix
    import parser
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import cm
    
    # Preparing the data input in the given function to match with data dimensions
    N_data = ''
    for i in range(0,np.shape(data)[1]-1):
        N_data += 'data[i][%d], ' %(i)
    
    # Making a PDF for each dimension in data
    PDF = []
    for i in range(0,len(Functions)):
        PDF.append('Functions[%d]( ([%s]) , pars[%d] ) / pars[%d][0]' %(i,N_data,i,i))
        PDF[i] = parser.expr(PDF[i]).compile()

    # Calculation the denominator needed in covariance matrix calculation
    denominator = '0'
    for i in range(0,len(Functions)):
        denominator += ' + pars[%d][0]*eval(PDF[%d])' %(i,i)
    denominator = parser.expr(denominator).compile()

    iclist = np.zeros([len(Functions),len(Functions)])

    # Caculation of elements in the inverse covariance matrix
    for j in range(0,len(Functions)):
        for k in range(0,len(Functions)):
            for i in range(0,len(data)) :
                PDF1=eval(PDF[j])
                PDF2=eval(PDF[k])
                den = eval(denominator)**2
                iclist[j][k] += PDF1*PDF2 / den
        
    invcovmat = matrix( iclist )
    
    covmat = invcovmat.I
    print "Covarance matrix as obtained from direct calculation: "
    print covmat

    # Calculating sWeights for each signal present in data
    sWeight = np.zeros([len(data),len(Functions)])
    for j in range(0,len(Functions)):
        Weight = ''
        for k in range(0,len(Functions)):
            Weight += '+ covmat.item(%d,%d) * eval(PDF[%d]) ' %(j,k,k)
        for i in range(0,len(data)):
            sWeight[i,j] = eval(Weight)/eval(denominator)
    
    # ------------------------------------------------------------------- #

    # Making bins and steps for the plotting option
    if bins == None:
        bins = 100

    Bins = []
    Steps = []
    Less_Bins = False
    New_Bins = []

    # Downscaling number of bins if the number of bins given is bigger than the number of unique
    # values in one of the datasets used
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

    # Calculation errors per bin using equation 22 in the sPlot paper. https://arxiv.org/pdf/physics/0402083.pdf
    err = np.zeros([len(Steps[-1]),len(Functions)])
    for k in range(0,len(Functions)):
        for j in range(0,len(Steps[-1])):  
            for i in range(0,len(data)):
                if Bins[-1][j] <= data[i][-1] < Bins[-1][j+1]:
                    err[j][k] += sWeight[i][k]**2 
    err = np.sqrt(err)

    # Saving the errors if 'True'
    if save_err == 'True':
        np.savetxt('errBins.txt', err, fmt='%8.4f')
    
    color = cm.rainbow(np.linspace(0,1,len(Functions)))

    # Plotting the entire dataset and the individual contributions from the different signals
    # calculated with the corrosponding sWeights.
    if plot == 'True':

        for i in range(0,len(Functions)):
            plt.figure(i)
            label = 'Signal %d' %(i+1)
            hist = np.histogram(data[:,-1],weights=sWeight[:,i],bins=Bins[-1])
            plt.hist(data[:,-1],bins=Bins[-1],stacked=False,histtype='step',linewidth=2.0,color='orange',label='Data')
            plt.errorbar(Steps[-1], hist[0],yerr=err[:,i],fmt='.',capsize=3,capthick=2,color=color[i],label=label)
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.legend()
            plt.show(block=False)

        plt.figure(len(Functions))
        for i in range(0,len(Functions)):
            label = 'Signal %d' %(i+1)
            hist = np.histogram(data[:,-1],weights=sWeight[:,i],bins=Bins[-1])
            plt.errorbar(Steps[-1], hist[0],yerr=err[:,i],fmt='.',capsize=3,capthick=2,color=color[i],label=label) 
            
        plt.hist(data[:,-1],bins=Bins[-1],stacked=False,histtype='step',linewidth=2.0,color='orange',label='Data')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show(block=False)

    # Plotting only the combined image
    if plot == 'Combined':

        for i in range(0,len(Functions)):
            label = 'Signal %d' %(i+1)
            hist = np.histogram(data[:,-1],weights=sWeight[:,i],bins=Bins[-1])
            plt.errorbar(Steps[-1], hist[0],yerr=err[:,i],fmt='.',capsize=3,capthick=2,color=color[i],label=label) 
            
        plt.hist(data[:,-1],bins=Bins[-1],stacked=False,histtype='step',linewidth=2.0,color='orange',label='Data')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show(block=False)
    
    # Saves the sWeights if 'True'
    if write == 'True':
        np.savetxt('allsWeights.txt', sWeight, fmt='%8.4f')
    
    raw_input( ' ... ' )
    plt.close('all')
    return sWeight


# ------------------------------------------------------------------------ #
if __name__ == '__sWeights__':
    sys.exit( main() )


