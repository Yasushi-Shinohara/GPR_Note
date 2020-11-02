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