import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    true_mu = 5
    true_sigma = 2
    n = 1000
    data = true_mu + true_sigma * np.random.randn(n)
    
    # MLE estimates
    mu_hat = np.mean(data)
    sigma2_hat = np.mean((data - mu_hat)**2)
    sigma_hat = np.sqrt(sigma2_hat)
    
    print(f"Estimated Mean: {mu_hat:.4f}")
    print(f"Estimated Std Dev: {sigma_hat:.4f}")
    
    plt.figure()
    plt.hist(data, bins=30, density=True, alpha=0.7, label='Data Histogram')
    x_values = np.linspace(np.min(data), np.max(data), 100)
    y_values = (1/(sigma_hat * np.sqrt(2*np.pi))) * np.exp(-0.5*((x_values - mu_hat)/sigma_hat)**2)
    plt.plot(x_values, y_values, 'r-', linewidth=2, label='Fitted Normal PDF')
    plt.title("MLE for Normal Distribution")
    plt.xlabel("Data Value")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid(True)
    plt.show()
