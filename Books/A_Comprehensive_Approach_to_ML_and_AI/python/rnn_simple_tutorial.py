import numpy as np
import matplotlib.pyplot as plt

def rnn_train(X, Y, hidden_size, alpha, epochs):
    """
    Trains a simple RNN for time-series prediction.

    Args:
        X (np.ndarray): Input sequence of shape (d, T) where each column is a time step.
        Y (np.ndarray): Target sequence of shape (1, T).
        hidden_size (int): Number of neurons in the hidden layer.
        alpha (float): Learning rate.
        epochs (int): Number of training epochs.

    Returns:
        model (dict): Dictionary containing trained parameters.
                      Keys: 'Wxh', 'Whh', 'bh', 'Why', 'by', 'hiddenSize'
    """
    d, T = X.shape
    output_size = 1  # Regression output

    # Initialize weights and biases with small random values
    Wxh = np.random.randn(hidden_size, d) * 0.01       # Input-to-hidden weights
    Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # Hidden-to-hidden weights
    bh = np.zeros((hidden_size, 1))                      # Hidden bias
    Why = np.random.randn(output_size, hidden_size) * 0.01  # Hidden-to-output weights
    by = np.zeros((output_size, 1))                      # Output bias

    for epoch in range(epochs):
        # Initialize hidden state for the current sequence
        h = np.zeros((hidden_size, T))
        h_prev = np.zeros((hidden_size, 1))

        # Forward pass
        y_pred = np.zeros((1, T))
        for t in range(T):
            x_t = X[:, t].reshape(d, 1)  # Shape (d, 1)
            h_current = np.tanh(np.dot(Wxh, x_t) + np.dot(Whh, h_prev) + bh)  # (hidden_size, 1)
            h[:, t] = h_current.reshape(-1)  # Store as vector
            y_pred[0, t] = np.dot(Why, h_current) + by  # Scalar output
            h_prev = h_current

        # Compute loss (Mean Squared Error)
        loss = 0.5 * np.sum((Y - y_pred) ** 2)

        # Initialize gradients for all parameters
        dWxh = np.zeros_like(Wxh)
        dWhh = np.zeros_like(Whh)
        dbh = np.zeros_like(bh)
        dWhy = np.zeros_like(Why)
        dby = np.zeros_like(by)
        dh_next = np.zeros((hidden_size, 1))

        # Backpropagation Through Time (BPTT)
        for t in reversed(range(T)):
            # Compute output error at time t (scalar)
            dy = y_pred[0, t] - Y[0, t]
            # Gradients for output weights and bias
            h_t = h[:, t].reshape(hidden_size, 1)
            dWhy += dy * h_t.T
            dby += np.array([[dy]])
            # Backpropagate into hidden state
            dh = np.dot(Why.T, np.array([[dy]])) + dh_next  # (hidden_size, 1)
            # Backprop through tanh nonlinearity
            dtanh = (1 - h_t ** 2) * dh  # (hidden_size, 1)
            # Gradients for input weights and bias
            x_t = X[:, t].reshape(d, 1)
            dWxh += np.dot(dtanh, x_t.T)
            dbh += dtanh
            # Gradients for recurrent weights
            if t > 0:
                h_prev_t = h[:, t - 1].reshape(hidden_size, 1)
                dWhh += np.dot(dtanh, h_prev_t.T)
            # Update dh_next for previous time step
            dh_next = np.dot(Whh.T, dtanh)

        # Average gradients over time steps
        dWxh /= T
        dWhh /= T
        dbh  /= T
        dWhy /= T
        dby  /= T

        # Update parameters using gradient descent
        Wxh -= alpha * dWxh
        Whh -= alpha * dWhh
        bh  -= alpha * dbh
        Why -= alpha * dWhy
        by  -= alpha * dby

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")

    # Store trained parameters in the model dictionary
    model = {
        'Wxh': Wxh,
        'Whh': Whh,
        'bh': bh,
        'Why': Why,
        'by': by,
        'hiddenSize': hidden_size
    }
    return model

def rnn_predict(model, X):
    """
    Predicts outputs for an input sequence X using the trained RNN model.

    Args:
        model (dict): Trained model parameters.
        X (np.ndarray): Input sequence of shape (d, T).

    Returns:
        y_out (np.ndarray): Predicted output sequence of shape (1, T).
    """
    d, T = X.shape
    hidden_size = model['hiddenSize']
    h_prev = np.zeros((hidden_size, 1))
    y_out = np.zeros((1, T))
    for t in range(T):
        x_t = X[:, t].reshape(d, 1)
        h = np.tanh(np.dot(model['Wxh'], x_t) + np.dot(model['Whh'], h_prev) + model['bh'])
        y_out[0, t] = np.dot(model['Why'], h) + model['by']
        h_prev = h
    return y_out

# Example usage
if __name__ == '__main__':
    np.random.seed(1)  # For reproducibility

    # Generate synthetic time-series data (a sine wave with noise)
    T = 100
    t_axis = np.linspace(0, 2 * np.pi, T)
    X = np.sin(t_axis).reshape(1, T)  # Shape (1, T): 1-dimensional input
    Y = np.sin(t_axis + 0.5) + 0.1 * np.random.randn(1, T)  # Target: phase-shifted sine with noise

    hidden_size = 10
    alpha = 0.01   # Learning rate
    epochs = 5000
    model = rnn_train(X, Y, hidden_size, alpha, epochs)
    Y_pred = rnn_predict(model, X)

    # Plot true vs. predicted time series
    plt.figure()
    plt.plot(t_axis, Y.flatten(), 'b-', linewidth=2, label='True Values')
    plt.plot(t_axis, Y_pred.flatten(), 'r--', linewidth=2, label='Predicted Values')
    plt.xlabel('Time')
    plt.ylabel('Output')
    plt.title('RNN Time-Series Prediction')
    plt.legend()
    plt.grid(True)
    plt.show()
