import numpy as np
from scipy.stats import norm

class Option:

    def __init__(self, r_f, S, X, T, sigma=None, u=None, d=None, eu_or_am='eu', call_or_put='call', time_step=1):

        """
        sigma : volatility
        r_f : risk-free rate
        S : stock price
        X : exercise price
        T : Expiration Time
        """
        self.sigma = sigma
        self.r_f = r_f
        self.S = S
        self.X = X
        self.T = T

        """
        u : multplicative upward movement in the stock price 
        """
        if u != None:
            self.u = u
        elif u == None and sigma != None:
            self.u = np.exp(self.sigma*np.sqrt(time_step))
        else:
            self.u = None
           
        """
        d : multiplicative downward movement in the stock price
        """
        if d != None:
            self.d = d
        elif d == None and sigma != None:
            self.d = np.exp(-self.sigma*np.sqrt(time_step))
        else:
            self.d = None
        
        """
        American or European Option
        """

        self.eu_or_am = eu_or_am

        """
        Call or Put option
        """
        self.call_or_put = call_or_put

        """
        p : risk-neutral probability
        """
        if self.u != None and self.d != None:
            self.p = (1 + self.r_f - self.d)/( self.u - self.d )

        """
        d1, d2 : some quantities for calculating the price in the Black-Scholes Formula
        """
        if self.sigma != None:
            self.d1 = (np.log(self.S / self.X) + self.r_f * self.T)/(self.sigma*np.sqrt(self.T)) + 0.5 * self.sigma * np.sqrt(self.T)
            self.d2 = self.d1 - self.sigma * self.T

        """
        Option Value
        """
        self.value = None

        

    def binomial_model(self):
        print("Using binomial model ... \n")
        
        if not isinstance(self.T, int):
            raise TypeError("Expiration Time (T) must be an integer.")

        self.binomial_tree = np.zeros((self.T + 1, self.T + 1))

        for j in range(self.T + 1):
            if self.call_or_put == 'call':
                self.binomial_tree[j, self.T] = np.max([0,
                                np.power(self.u, self.T - j) * np.power(self.d, j) * self.S - self.X])
            else:
                self.binomial_tree[j, self.T] = np.max([0,
                                self.X - np.power(self.u, self.T - j) * np.power(self.d, j) * self.S])
                
        for j in [ self.T - 1 - x for x in range(self.T)]:
            for i in range(j + 1):
                european_value = self.p * self.binomial_tree[i, j + 1] + (1 - self.p) * self.binomial_tree[i + 1, j + 1]
                european_value /= 1 + self.r_f

                if self.eu_or_am == 'am':
                    if self.call_or_put == 'call':
                        self.binomial_tree[i, j] = np.max([european_value, 
                                np.power(self.u, j - i) * np.power(self.d, i) * self.S - self.X])
                    else:
                        self.binomial_tree[i, j] = np.max([european_value, 
                                self.X - np.power(self.u, j - i) * np.power(self.d, i) * self.S])
                else:    
                    self.binomial_tree[i, j] = european_value

        self.value = self.binomial_tree[0,0]

        return self.binomial_tree
        
    def black_scholes_formula(self):
        print("Using the Black-Scholes Formula to calculate the call option value ... \n")
        self.value = self.S * norm.cdf(self.d1) - self.X * np.exp(- self.r_f * self.T ) * norm.cdf(self.d2)

        return self.value
    

if __name__ == '__main__':
    call_option = Option(r_f=0.05, S=400, X=100, T=4, sigma=0.6)
    print(call_option.black_scholes_formula())
    print(norm.cdf(call_option.d1))

    # call_option = Option(r_f=0.1, S=20, X=21, T=2, u=1.2, d=0.67, eu_or_am='eu', call_or_put='call')
    # print(call_option.binomial_model())

    # eu_put_option = Option(r_f=0.1, S=50, X=50, T=2, u=1.2, d=0.6, eu_or_am='eu', call_or_put='put')
    # print(eu_put_option.binomial_model())

    # am_put_option = Option(r_f=0.1, S=50, X=50, T=2, u=1.2, d=0.6, eu_or_am='am', call_or_put='put')
    # print(am_put_option.binomial_model())





