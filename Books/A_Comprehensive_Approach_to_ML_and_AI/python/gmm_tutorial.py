import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def myGMM(X, maxIter, tol):
    """
    EM algorithm for a 2-component Gaussian Mixture Model (GMM).

    Parameters:
        X       - 1D data array.
        maxIter - Maximum number of iterations.
        tol     - Tolerance for convergence based on change in log-likelihood.

    Returns:
        pi_est        - Estimated mixing coefficient for component 1.
        mu1, mu2      - Estimated means of the two components.
        sigma1, sigma2- Estimated standard deviations of the two components.
        loglikelihood - Array containing the log-likelihood history.
    """
    n = len(X)
    
    # Initialize parameters
    pi_est = 0.5
    mu1 = np.mean(X) - np.std(X) / 2
    mu2 = np.mean(X) + np.std(X) / 2
    sigma1 = np.std(X)
    sigma2 = np.std(X)
    
    loglikelihood = []
    
    for iter in range(maxIter):
        # E-step: compute responsibilities using vectorized operations
        p1 = pi_est * norm.pdf(X, loc=mu1, scale=sigma1)
        p2 = (1 - pi_est) * norm.pdf(X, loc=mu2, scale=sigma2)
        gamma = p1 / (p1 + p2)
        
        # M-step: update parameters
        pi_new = np.mean(gamma)
        mu1_new = np.sum(gamma * X) / np.sum(gamma)
        mu2_new = np.sum((1 - gamma) * X) / np.sum(1 - gamma)
        sigma1_new = np.sqrt(np.sum(gamma * (X - mu1_new)**2) / np.sum(gamma))
        sigma2_new = np.sqrt(np.sum((1 - gamma) * (X - mu2_new)**2) / np.sum(1 - gamma))
        
        # Compute log-likelihood
        ll = np.sum(np.log(pi_new * norm.pdf(X, loc=mu1_new, scale=sigma1_new) +
                           (1 - pi_new) * norm.pdf(X, loc=mu2_new, scale=sigma2_new)))
        loglikelihood.append(ll)
        
        # Check for convergence
        if iter > 0 and abs(loglikelihood[iter] - loglikelihood[iter - 1]) < tol:
            break
        
        # Update parameters for next iteration
        pi_est, mu1, mu2 = pi_new, mu1_new, mu2_new
        sigma1, sigma2 = sigma1_new, sigma2_new
        
    return pi_est, mu1, mu2, sigma1, sigma2, np.array(loglikelihood)

# Generate sample data: 100 points from N(0,1) and 100 points from N(3,1)
np.random.seed(0)  # For reproducibility
X = np.concatenate((np.random.randn(100), np.random.randn(100) + 3))

# Run the EM algorithm for GMM
pi_est, mu1, mu2, sigma1, sigma2, ll = myGMM(X, maxIter=100, tol=1e-4)

# Plot the log-likelihood convergence
plt.figure()
plt.plot(ll, linewidth=2)
plt.title('Log-Likelihood Convergence')
plt.xlabel('Iteration')
plt.ylabel('Log-Likelihood')
plt.show()
