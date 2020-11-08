from scipy.stats import norm
#
class  Bayesian_opt():
    def __init__(self):
        self.acqui_name = 'PI' #'PI', 'EI', 'UCB'
        self.xi = 0.01

#### PI
    def acqui_PI(self, mean, std, maxval):
        Z = (mean -  maxval - self.xi)/std
        return norm.cdf(Z)
#### EI
    def acqui_EI(self, mean, std, maxval):
        Z = (mean -  maxval - self.xi)/std
        return (mean - maxval - self.xi)*norm.cdf(Z) + std*norm.pdf(Z)
#### UCB
    def acqui_UCB(self, mean, std, maxval):
        return mean + 1.0*std

    def get_acqui(self, mean, std, maxval):
        if (self.acqui_name == 'PI'):
            acqui = self.acqui_PI(mean, std, maxval)
        elif (self.acqui_name == 'EI'):
            acqui = self.acqui_EI(mean, std, maxval)
        elif (self.acqui_name == 'UCB'):
            acqui = self.acqui_UCB(mean, std, maxval)
        else:
            print('# ERROR: undefined acquisition function called.')
            sys.exit()

        return acqui
#
import numpy as np
import matplotlib.pyplot as plt
#
def DO_BO(GPR, BO, x2y, x, x_sample_init, y_sample_init, Nepoch, nplotevery, answer_is_there = False):
    if (answer_is_there):
        y = list(map(x2y,x)) #for python3
        y = np.array(y)
    xmin = np.amin(x)
    xmax = np.amax(x)
    x_sample = 1.0*x_sample_init
    y_sample = 1.0*y_sample_init
    maxval_list = []
    maxval = max(y_sample)
    maxval_list.append(maxval)
    plt.figure(figsize=(16, 20))

    for i in range(Nepoch):
        mean, std = GPR.xsample2meanstd(x_sample, y_sample, x) #Get mean/std
        maxval = max(y_sample)                                                     #Update maximum value up to now
        acqui = BO.get_acqui(mean, std, maxval)                                      #Define acquisition function
        if(i%nplotevery==0):
            plt.subplot(Nepoch//nplotevery+1,2,(i//nplotevery)*2+1)
            plt.title('epoch = '+str(i))
            plt.plot(x,np.array(mean))
            if (answer_is_there):
                plt.plot(x,y,'k--')
            high_bound = mean+ 1.0*std
            lower_bound = mean- 1.0*std
            plt.fill_between(x,high_bound,lower_bound, alpha=0.5)
            plt.xlim(xmin,xmax)
            plt.ylim(1.2*min(np.amin(mean),min(y_sample)),1.2*max(np.amax(mean),max(y_sample)))
            if (answer_is_there):
                plt.ylim(1.2*np.amin(y),1.2*np.amax(y))
            plt.scatter(x_sample,y_sample)
            plt.subplot(Nepoch//nplotevery+1,2,(i//nplotevery)*2+2)
            plt.xlim(xmin,xmax)
            plt.plot(x,acqui)

        x_point = x[np.argmax(acqui)]                                               #Determin the next candidate of the search
        if x_point not in x_sample:
            x_sample = np.append(x_sample,x_point)                          #Add x_point to x_sample
            y_point = x2y(x_point)                              #Get y_point value from x_sample
            y_sample = np.append(y_sample,y_point)                          #Add y_point to x_ample
            print ("epoch = ",str(i),", x_point, maxval = "+str(x_point)+',  '+str(max(y_sample)))
        else: 
            print('No new sampling point in this sequence.')
        maxval_list.append(maxval)
    plt.savefig("bayes_"+str(BO.acqui_name)+".png")
    plt.show()


    if (answer_is_there):
        print("# Actual value of maximum is "+str(max(y)))
    print("# Got value in the Bayesian_opt is "+str(maxval))
    print("# Optimization is finished.")
    plt.title('epoch = '+str(i))
    plt.xlabel('x',fontsize = 18)
    plt.ylabel('y',fontsize = 18)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.plot(x,np.array(mean), label='mean',linewidth = 3.0)
    if (answer_is_there):
        plt.plot(x,y,'k--', label='True data')
    high_bound = mean+ 1.0*std
    lower_bound = mean- 1.0*std
    plt.fill_between(x,high_bound,lower_bound, alpha=0.5,label='2$\sigma$ confidence')
    plt.xlim(xmin,xmax)
    plt.ylim(1.2*np.amin(mean),1.2*np.amax(mean))
    if (answer_is_there):
        plt.ylim(1.2*np.amin(y),1.2*np.amax(y))
    plt.scatter(x_sample,y_sample, label='Sampled data')
    plt.grid()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',fontsize = 18)
    plt.show()
    return mean, std, x_point, y_point, maxval_list
