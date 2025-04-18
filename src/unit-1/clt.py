import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

class Distribution:
    def __init__(self, n_variables, m_vectors):
        self.n_variables = n_variables
        self.m_vectors = m_vectors

    def normal(self, tru_mean, std):
        mean = []
        for _ in range(self.m_vectors):
            sample = np.random.normal(loc= tru_mean, scale= std, size = self.n_variables)
            sample_mean = np.mean(sample)
            
            mean.append(sample_mean)

        return mean
    
    def binomial(self,r = 10, p = 0.5):
        mean = []
        for _ in range(self.m_vectors):
            sample = np.random.binomial(n = r, p = p, size= self.n_variables)
            sample_mean = np.mean(sample)

            mean.append(sample_mean)

        return mean
    
    def poisson(self, lam = 5):
        mean = []
        for _ in range(self.m_vectors):
            sample = np.random.poisson(lam = lam, size= self.n_variables)
            sample_mean = np.mean(sample)

            mean.append(sample_mean)

        return mean        
    def cauchy(self):
        mean = []
        for _ in range(self.m_vectors):
            sample = np.random.standard_cauchy(size = self.n_variables)
            sample_mean = np.mean(sample)

            mean.append(sample_mean)

        return mean
    
def plot_dist(distribution, plot_name : str):
    plt.figure(figsize= (10,6))
    sns.distplot(distribution, bins = 50, kde= True)
    plt.xlabel("Sample mean")
    plt.ylabel("Density")
    plt.grid(True)

    plt.savefig(plot_name)

dist = Distribution(n_variables= 30, m_vectors= 10000)

def verify_clt(distribution, tru_mean, tru_std,N, plot_name):
    sample_means  = np.array(distribution)

    z_values = (sample_means - tru_mean)/(tru_std/np.sqrt(N))

    plt.figure(figsize = (10,6))
    sns.distplot(z_values, bins = 60, kde = True)
    
    x = np.linspace(-4, 4, 500)
    plt.plot(x, norm.pdf(x), 'r--', label="Standard Normal N(0,1)")
    plt.title(f'CLT Verification for Distribution')
    plt.xlabel('Standardized Mean')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_name)
    plt.show()

# verify_clt(dist.binomial(), tru_mean= 2, tru_std= 3, N = dist.n_variables,plot_name= "Nice" )

verify_clt(dist.cauchy(), tru_mean= 2, tru_std=3, N = dist.n_variables, plot_name= "cauchy")

