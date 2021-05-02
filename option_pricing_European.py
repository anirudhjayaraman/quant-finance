import numpy as np                           # For its math capabilities
import matplotlib as mpl                     # Plotting using matplotlib
import matplotlib.pyplot as plt              # Plotting using matplotlib.pyplot                          
from scipy.integrate import quad             # For quadratic integration (CDF)
import mpl_toolkits.mplot3d.axes3d as p3     # For 3D visualizations

class EuropeanOptions:
    def __init__(self, St: float, K: float, t: float, T: float, r: float, sigma: float):
        '''
        Parameters
        ==========
        St: float -> stock/index level at time t
        K:  float -> strike price
        t:  float -> valuation date
        T:  float -> date of maturity; T > t
        r:  float -> constant, risk-less short rate
        sigma:  float -> volatility
        ''' 
        self.St = St
        self.K = K
        self.t = t
        self.T = T
        self.r = r
        self.sigma = sigma
        
    def dN(self, x) -> float:        
        ''' Probability density function of standard normal random variable x.'''
        return np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)

    def N(self, d) -> float:
        ''' Cumulative density function of standard normal random variable x. 
            Integral of the Gaussian density function between -Infinity and d.
        '''    
        return quad(lambda x: self.dN(x), -np.Inf, d)[0]

    def d1_func(self) -> float:
        ''' Black-Scholes-Merton d1 function.'''
        d1 = (np.log(self.St / self.K) + (self.r + 0.5 * self.sigma ** 2) * \
              (self.T - self.t)) / (self.sigma * np.sqrt(self.T - self.t))

        return d1

    def d2_func(self) -> float:
        '''Black-Scholes-Merton d2 function.'''
        d1 = self.d1_func()
        d2 = d1 - self.sigma * np.sqrt(self.T - self.t)

        return d2

    def BSM_call(self) -> float:
        ''' Calculates Black-Scholes-Merton European call option value.

        Returns
        =======
        call_value:  float -> European call present value at t
        '''
        d1 = self.d1_func()
        d2 = self.d2_func()
        call_value = self.St * self.N(d1) - np.exp(-self.r * (self.T - self.t)) * self.K * self.N(d2)

        return call_value

    def BSM_put(self) -> float:
        ''' Calculates Black-Scholes-Merton European put option value (relying on put-call parity).
        
        Returns
        =======
        put_value: float -> European put present value at t
        '''
        put_value = self.BSM_call() - self.St + np.exp(-self.r * (self.T - self.t)) * self.K

        return put_value


if __name__ == "__main__":
    St = 100.0
    K = 100.0
    t = 0.0
    T = 1.0
    
    r = 0.05
    sigma = 0.30
    
    model = EuropeanOptions(St, K, t, T, r, sigma)
    call_value = model.BSM_call()
    put_value = model.BSM_put()
    
    print("Theoretical call option price is: ", call_value)
    print("Theoretical put option price is: ", put_value)