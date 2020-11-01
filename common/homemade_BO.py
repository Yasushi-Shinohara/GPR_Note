from scipy.stats import norm
#
class  Bayesian_opt():
    def __init__(self):
        self.aqui_name = 'PI' #'PI', 'EI', 'UCB'
        self.xi = 0.01

#### PI
    def aqui_PI(self, mean, std, maxval):
        Z = (mean -  maxval - self.xi)/std
        return norm.cdf(Z)
#### EI
    def aqui_EI(self, mean, std, maxval):
        Z = (mean -  maxval - self.xi)/std
        return (mean - maxval - self.xi)*norm.cdf(Z) + std*norm.pdf(Z)
#### UCB
    def aqui_UCB(self, mean, std, maxval):
        return mean + 1.0*std

    def get_aqui(self, mean, std, maxval):
        if (self.aqui_name == 'PI'):
            aqui = self.aqui_PI(mean, std, maxval)
        elif (self.aqui_name == 'EI'):
            aqui = self.aqui_EI(mean, std, maxval)
        elif (self.aqui_name == 'UCB'):
            aqui = self.aqui_UCB(mean, std, maxval)
        else:
            print('# ERROR: undefined acquisition function called.')
            sys.exit()

        return aqui