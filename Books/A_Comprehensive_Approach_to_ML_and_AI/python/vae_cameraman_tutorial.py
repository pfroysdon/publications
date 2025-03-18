import numpy as np
import matplotlib.pyplot as plt
from skimage import io

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Load and preprocess the image.
I = io.imread('data/cameraman.tif')
I = I.astype(np.float64) / 255.0  # normalize to [0,1]
H, W = I.shape
img_size = (H, W)
x = I.flatten().reshape(-1, 1)  # shape (H*W, 1)

# VAE hyperparameters.
input_dim = x.shape[0]
latent_dim = 20
hidden_dim = 100
alpha = 0.001
epochs = 2000

def vae_train(x, input_dim, latent_dim, hidden_dim, alpha, epochs):
    # Initialize encoder parameters.
    W_enc = 0.01 * np.random.randn(hidden_dim, input_dim)
    b_enc = np.zeros((hidden_dim, 1))
    W_mu = 0.01 * np.random.randn(latent_dim, hidden_dim)
    b_mu = np.zeros((latent_dim, 1))
    W_logvar = 0.01 * np.random.randn(latent_dim, hidden_dim)
    b_logvar = np.zeros((latent_dim, 1))
    
    # Initialize decoder parameters.
    W_dec = 0.01 * np.random.randn(hidden_dim, latent_dim)
    b_dec = np.zeros((hidden_dim, 1))
    W_out = 0.01 * np.random.randn(input_dim, hidden_dim)
    b_out = np.zeros((input_dim, 1))
    
    for epoch in range(1, epochs+1):
        # Encoder forward.
        h_enc = np.tanh(W_enc @ x + b_enc)
        mu = W_mu @ h_enc + b_mu
        logvar = W_logvar @ h_enc + b_logvar
        sigma = np.exp(0.5 * logvar)
        
        # Reparameterization trick.
        epsilon = np.random.randn(latent_dim, 1)
        z = mu + sigma * epsilon
        
        # Decoder forward.
        h_dec = np.tanh(W_dec @ z + b_dec)
        x_hat = sigmoid(W_out @ h_dec + b_out)
        
        # Loss: reconstruction loss (binary cross-entropy) + KL divergence.
        recon_loss = -np.sum(x * np.log(x_hat + 1e-8) + (1 - x) * np.log(1 - x_hat + 1e-8))
        kl_loss = -0.5 * np.sum(1 + logvar - mu**2 - np.exp(logvar))
        loss = recon_loss + kl_loss
        
        # Simplified gradient update (only updating W_out, b_out for demonstration).
        dL_dxhat = (x_hat - x)  # approximate gradient
        grad_W_out = dL_dxhat @ h_dec.T
        grad_b_out = dL_dxhat
        W_out = W_out - alpha * grad_W_out
        b_out = b_out - alpha * grad_b_out
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f} (Recon: {recon_loss:.4f}, KL: {kl_loss:.4f})")
    
    model = {
        'W_enc': W_enc, 'b_enc': b_enc,
        'W_mu': W_mu, 'b_mu': b_mu,
        'W_logvar': W_logvar, 'b_logvar': b_logvar,
        'W_dec': W_dec, 'b_dec': b_dec,
        'W_out': W_out, 'b_out': b_out
    }
    return model

def vae_predict(model, x):
    h_enc = np.tanh(model['W_enc'] @ x + model['b_enc'])
    mu = model['W_mu'] @ h_enc + model['b_mu']
    # Use mu as the latent representation.
    z = mu
    h_dec = np.tanh(model['W_dec'] @ z + model['b_dec'])
    x_hat = sigmoid(model['W_out'] @ h_dec + model['b_out'])
    return x_hat

model = vae_train(x, input_dim, latent_dim, hidden_dim, alpha, epochs)
x_recon = vae_predict(model, x)
I_recon = x_recon.reshape(img_size)

plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.imshow(I, cmap='gray')
plt.title("Original Image")
plt.subplot(1,3,2)
plt.imshow(I_recon, cmap='gray')
plt.title("Reconstructed Image")
plt.subplot(1,3,3)
plt.imshow(np.abs(I - I_recon), cmap='gray')
plt.title("Absolute Difference")
plt.axis('off')
plt.show()
