

def calc_sWeights(Functions,pars,data,testdata,plot=False,bins=100,write=False,save_err=False):
    
    """

    Calculation of sWeights given data that is fitable by some function.

    Parameters
    ----------

    Functions : list of callable functions
        The "PDFs" defining the different signals in data. 
        Assumes the first parameter (p[0]) to be the scaling for the function to fit data. Needs to be of the form: f(x,p). 
        
            - example: f(x,p) = p[0] * p[1] * exp(-p[1] * x[0]), p[1] = 2.0 (p[1] inserted via **pars**); 
              or f(x,p) = p[0] * 2.0 * exp(-2.0 * x[0]) (p[1] inserted directly in function). 
              Only p[0] has to be a variable and inserted via **pars**.

    pars : Best fit values found by fitting **Functions** to data
        Has to be an N*M ndarray, N being the number of parameters for either function and M being the number of functions.
        A M dimensional fit is required if the data used has M dimensions.

    data : An N*M ndarray 
        N is the number of data points and M the number of dimensions. 
        
    testdata : N length vector
        Vector containing the data which needs to be seperated into different signals.
        N is the number of data points.

    plot : False, 'True' or 'Combined', optional
        If plot='True' plots the last column in the data given in a histogram and shows the individual contributions
        from functions on individual plots + individual contributions on a combined plot. 
        If plot='Combined' only the combined plot is shown.

    bins : Real positive integer, optional
        The number of bins wanted on the plot and in the error calculation (default is 100).

    write : False or 'True', optional
        If write='True' the sWeights are saved in a text file in the format N*M. N is the number of data points
        and M is the number of signal types

    save_err : False or 'True', optional
        If save_err='True' the errors calculated are saved to a text file in the format N*M
        N is the number of bins chosen and M is the number of signal types

    Returns
    -------

    sWeights : N*M array containing the sWeights
        N is the number of data points and M the number of Functions. 
    """

    import numpy as np
    from numpy import matrix

    # References:
    # [1] arXiv:physics/0402083

    # Defining dimensions of inputs
    nSignals = len(Functions)
    nEvents = data.shape[0]

    if len(data.shape) == 1:
        nDim = 1
    else:
        nDim = data.shape[1]

    allData = np.column_stack((data, testdata))

    # Check that the input has a minimum number of functions
    if nSignals < 2:
        raise Exception( "sWeights requires at least two input functions")

    # Check that each function has a set of parameters
    if nSignals != len(pars):
        raise Exception( "Must have parameters for each function")

    # Check that each dimension has the same number of events
    for i_d in range(0,nDim+1):
        if len(allData[:,i_d]) != nEvents:
            raise Exception("Number of events in dimension %i does match expected number of events (%i)" % (i,nEvents) )
    
    # Creating a PDF for each signal in data (to be called later)
    pdfFuncs = []
    expectedNEvents = []
    for i_s in range(nSignals):
        pdfFuncs.append(lambda x, f=Functions[i_s], p=pars[i_s] : f(x,p) )
        expectedNEvents.append(pars[i_s][0])

    #
    # Calculate covariance matrix
    #

    # Use eqn 10 in [1] to calculate the inverse covaraince matrix, then invert to get covaraince matrix.
    # Subsequntly we will need many of the same values to calculate the sWeight according to eqn 14 in [1], so store the values to avoid calculating twice
    #  e.g. the value of each signals PDF for each event.
    # The common deniominator normalisation term per event for eqn 10 and 14 in [1] : sum_k[ N_k f_k(dataN)] = denom (need then to square it for eqn 10 in [1])

    # Arrays to fill (assign memory now)
    pdfValue = np.zeros([nSignals,nEvents],dtype=float)
    denom = np.zeros([nEvents],dtype=float)
    iclist = np.zeros([nSignals,nSignals],dtype=float)

    # Loop over events
    for i_e in range(0,nEvents):

        # Get list of discriminating variable values for this event (will pass to PDF)
        if nDim == 1:
            dataN = data[i_e]
        else: 
            dataN = [ data[i_e,i_d] for i_d in range(0,nDim) ]

        # Loop over signals (e.g. PDFs)
        for i_s in range(0,nSignals):

            # Calculate PDF value for this signals for this event
            pdfValue[i_s,i_e] = pdfFuncs[i_s](dataN) / expectedNEvents[i_s] 

            # Calculate the contribution to the normalisation denominator term for this event for this signal and add to running sum
            denom[i_e] += expectedNEvents[i_s] * pdfValue[i_s,i_e]

        # Add contribution from this event to each covairance matrix element
        denom2 = np.square(denom[i_e]) # Only square once
        for i_s_n in range(0,nSignals): # TODO Use symmetry to speed up
            for i_s_j in range(0,nSignals):
                iclist[i_s_n,i_s_j] += ( pdfValue[i_s_n,i_e] * pdfValue[i_s_j,i_e] ) / denom2

    # Invert to get covariance matrix
    iclist = matrix( iclist )
    covMatrix = iclist.I

    print "\nCovariance matrix :\n%s" % covMatrix

    # Calculating sWeights for each event, for each signals (using eqn 14 in [1])
    # Use values stored above from the covariance matrix calculation to speed up

    # Create the array to fill
    sWeight = np.zeros([nEvents,nSignals],dtype=float)

    # Loop over events
    for i_e in range(0,nEvents): #e in eqn 14 in [1]

        # Loop over signals (n in eqn 14 in [1])
        for i_s_n in range(0,nSignals):

            # Calculate this sWeight (there is a loop over signals, j from eqn 14 in [1], happening here)
            sWeight[i_e,i_s_n] = np.sum( [ covMatrix[i_s_n,i_s_j]*pdfValue[i_s_j,i_e] for i_s_j in range(0,nSignals) ] ) / denom[i_e]
    
    # ------------------------------------------------------------------- #
    
    # Making bins and steps for the plotting option
    less_bins = False
    new_bins = []

    # Downscaling number of bins if the number of bins given is bigger than the number of unique
    # values in one of the datasets used
    for i_d in range(0,nDim+1):
        if len(np.unique(allData[:,i_d])) <= bins:
            less_bins = True
            new_bins.append(len(np.unique(allData[:,i_d])))
           
    
    if less_bins == True:
        bins = min(new_bins)
        print "Number of bins reduced: More bins than number of different values in data. New number of bins: %i" % min(new_bins)
    
    used_bins = (np.linspace(min(testdata),max(testdata),(bins+1)))
    steps = (used_bins[0:bins]+(used_bins[1]-used_bins[0])/2.0)    
            

    # Calculation errors per bin using equation 22 in [1]. 
    err = np.zeros([len(steps),nSignals])
    for i_s in range(0,nSignals):
        for i_st in range(0,len(steps)):  
            for i_e in range(0,nEvents):
                if used_bins[i_st] <= testdata[i_e] < used_bins[i_st+1]:
                    err[i_st][i_s] += sWeight[i_e][i_s]**2 
    err = np.sqrt(err)

    # Saving the errors if 'True'
    if save_err == 'True':
        np.savetxt('errBins.txt', err, fmt='%8.4f')

    # Plotting the combined image
    if plot == 'True' or plot == 'Combined':
        
        import matplotlib.pyplot as plt
        from matplotlib.pyplot import cm

        color = cm.rainbow(np.linspace(0,1,nSignals))
        
        plt.figure(0)
        for i_s in range(0,nSignals):
            label = 'Signal %d' %(i_s+1)
            hist = np.histogram(testdata,weights=sWeight[:,i_s],bins=used_bins)
            plt.errorbar(steps, hist[0],yerr=err[:,i_s],fmt='.',capsize=3,capthick=2,color=color[i_s],label=label) 
            
        plt.hist(testdata,bins=used_bins,stacked=False,histtype='step',linewidth=2.0,color='orange',label='Data')
        plt.title('Test data')
        plt.xlabel('Value')
        plt.ylabel('Counts')
        plt.legend()
        plt.show(block=False)

    # Plotting the entire dataset and the individual contributions from the different signals
    # calculated with the corrosponding sWeights.
    if plot == 'True':

        for i_s in range(0,nSignals):
            plt.figure(i_s+1)
            label = 'Signal %d' %(i_s+1)
            hist = np.histogram(testdata,weights=sWeight[:,i_s],bins=used_bins)
            plt.hist(testdata,bins=used_bins,stacked=False,histtype='step',linewidth=2.0,color='orange',label='Data')
            plt.errorbar(steps, hist[0],yerr=err[:,i_s],fmt='.',capsize=3,capthick=2,color=color[i_s],label=label)
            plt.title('Test data')
            plt.xlabel('Value')
            plt.ylabel('Counts')
            plt.legend()
            plt.show(block=False)

    
    
    # Saves the sWeights if 'True'
    if write == 'True':
        np.savetxt('allsWeights.txt', sWeight, fmt='%8.4f')
    
    raw_input( ' ... ' )
    plt.close('all')
    return sWeight


    # ------------------------------------------------------------------------ #
if __name__ == '__main__':
    
    raise Exception("Should not run sWeightsGeneral.py, import it instead")


