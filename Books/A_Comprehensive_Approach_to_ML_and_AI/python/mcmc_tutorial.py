import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def target_density(x):
    # Unnormalized standard normal density
    return np.exp(-0.5 * x**2)

def metropolis_hastings(target_func, num_samples, proposal_std):
    samples = np.zeros(num_samples)
    samples[0] = 0  # initial state
    for i in range(1, num_samples):
        current = samples[i-1]
        proposal = current + proposal_std * np.random.randn()
        p_current = target_func(current)
        p_proposal = target_func(proposal)
        acceptance = min(1, p_proposal / p_current)
        if np.random.rand() < acceptance:
            samples[i] = proposal
        else:
            samples[i] = current
    return samples

if __name__ == '__main__':
    num_samples = 10000
    burn_in = 1000
    proposal_std = 1.0
    
    samples = metropolis_hastings(target_density, num_samples, proposal_std)
    samples = samples[burn_in:]
    
    plt.figure()
    plt.hist(samples, bins=50, density=True, edgecolor='none', alpha=0.7, label='MCMC Samples')
    x = np.linspace(np.min(samples)-1, np.max(samples)+1, 100)
    plt.plot(x, norm.pdf(x, 0, 1), 'r-', linewidth=2, label='True Normal PDF')
    plt.title("MCMC Sampling using Metropolis-Hastings")
    plt.xlabel("x")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid(True)
    plt.show()
