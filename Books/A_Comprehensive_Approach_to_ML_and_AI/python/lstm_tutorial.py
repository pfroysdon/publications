import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def lstm_train(X, Y, hidden_size, alpha, epochs):
    # X: input sequence of shape (d, T) where d=1 and T is number of time steps.
    # Y: target sequence of shape (1, T)
    d, T = X.shape
    output_size = 1  # Regression output
    
    # Initialize LSTM parameters (small random values)
    Wxi = np.random.randn(hidden_size, d) * 0.01
    Whi = np.random.randn(hidden_size, hidden_size) * 0.01
    bi  = np.zeros((hidden_size, 1))
    
    Wxf = np.random.randn(hidden_size, d) * 0.01
    Whf = np.random.randn(hidden_size, hidden_size) * 0.01
    bf  = np.zeros((hidden_size, 1))
    
    Wxo = np.random.randn(hidden_size, d) * 0.01
    Who = np.random.randn(hidden_size, hidden_size) * 0.01
    bo  = np.zeros((hidden_size, 1))
    
    Wxc = np.random.randn(hidden_size, d) * 0.01
    Whc = np.random.randn(hidden_size, hidden_size) * 0.01
    bc  = np.zeros((hidden_size, 1))
    
    # Output layer parameters
    Why = np.random.randn(output_size, hidden_size) * 0.01
    by  = np.zeros((output_size, 1))
    
    for epoch in range(1, epochs+1):
        # Forward pass: initialize states
        h = np.zeros((hidden_size, T))
        c = np.zeros((hidden_size, T))
        h_prev = np.zeros((hidden_size, 1))
        c_prev = np.zeros((hidden_size, 1))
        y_pred = np.zeros((1, T))
        
        # For storing gate activations (for BPTT)
        i_store = np.zeros((hidden_size, T))
        f_store = np.zeros((hidden_size, T))
        o_store = np.zeros((hidden_size, T))
        g_store = np.zeros((hidden_size, T))
        
        # Forward propagation through time
        for t in range(T):
            x_t = X[:, t:t+1]  # shape (d, 1)
            i_t = sigmoid(Wxi @ x_t + Whi @ h_prev + bi)
            f_t = sigmoid(Wxf @ x_t + Whf @ h_prev + bf)
            o_t = sigmoid(Wxo @ x_t + Who @ h_prev + bo)
            g_t = np.tanh(Wxc @ x_t + Whc @ h_prev + bc)
            
            c_t = f_t * c_prev + i_t * g_t
            h_t = o_t * np.tanh(c_t)
            
            # Save activations
            i_store[:, t:t+1] = i_t
            f_store[:, t:t+1] = f_t
            o_store[:, t:t+1] = o_t
            g_store[:, t:t+1] = g_t
            c[:, t:t+1] = c_t
            h[:, t:t+1] = h_t
            
            y_pred[0, t] = (Why @ h_t + by)[0,0]
            
            h_prev = h_t
            c_prev = c_t
        
        # Compute loss (Mean Squared Error)
        loss = 0.5 * np.sum((Y - y_pred)**2)
        
        # Initialize gradients for parameters
        dWxi = np.zeros_like(Wxi); dWhi = np.zeros_like(Whi); dbi = np.zeros_like(bi)
        dWxf = np.zeros_like(Wxf); dWhf = np.zeros_like(Whf); dbf = np.zeros_like(bf)
        dWxo = np.zeros_like(Wxo); dWho = np.zeros_like(Who); dbo = np.zeros_like(bo)
        dWxc = np.zeros_like(Wxc); dWhc = np.zeros_like(Whc); dbc = np.zeros_like(bc)
        dWhy = np.zeros_like(Why); dby = np.zeros_like(by)
        
        dh_next = np.zeros((hidden_size, 1))
        dc_next = np.zeros((hidden_size, 1))
        
        # Backpropagation Through Time (BPTT)
        for t in reversed(range(T)):
            # Output error (scalar)
            dy = y_pred[0, t] - Y[0, t]
            dWhy += dy * h[:, t:t+1].T
            dby += dy
            
            dh = (Why.T * dy) + dh_next  # shape (hidden_size, 1)
            # h_t = o_t * tanh(c_t)
            do = dh * np.tanh(c[:, t:t+1])
            do = do * o_store[:, t:t+1] * (1 - o_store[:, t:t+1])
            
            dct = dh * o_store[:, t:t+1] * (1 - np.tanh(c[:, t:t+1])**2) + dc_next
            
            # For c_t = f_t * c_prev + i_t * g_t
            di = dct * g_store[:, t:t+1]
            df = dct * (c[:, t-1:t] if t > 0 else np.zeros((hidden_size, 1)))
            dg = dct * i_store[:, t:t+1]
            dc_prev = dct * f_store[:, t:t+1]
            
            di = di * i_store[:, t:t+1] * (1 - i_store[:, t:t+1])
            df = df * f_store[:, t:t+1] * (1 - f_store[:, t:t+1])
            dg = dg * (1 - g_store[:, t:t+1]**2)
            
            x_t = X[:, t:t+1]
            h_prev_t = h[:, t-1:t] if t > 0 else np.zeros((hidden_size, 1))
            
            dWxi += di @ x_t.T
            dWhi += di @ h_prev_t.T
            dbi += di
            
            dWxf += df @ x_t.T
            dWhf += df @ h_prev_t.T
            dbf += df
            
            dWxo += do @ x_t.T
            dWho += do @ h_prev_t.T
            dbo += do
            
            dWxc += dg @ x_t.T
            dWhc += dg @ h_prev_t.T
            dbc += dg
            
            dh_next = Whi.T @ di + Whf.T @ df + Who.T @ do + Whc.T @ dg
            dc_next = dc_prev
        
        # Average gradients over time steps
        dWxi /= T; dWhi /= T; dbi /= T
        dWxf /= T; dWhf /= T; dbf /= T
        dWxo /= T; dWho /= T; dbo /= T
        dWxc /= T; dWhc /= T; dbc /= T
        dWhy /= T; dby /= T
        
        # Update parameters
        Wxi -= alpha * dWxi
        Whi -= alpha * dWhi
        bi  -= alpha * dbi
        
        Wxf -= alpha * dWxf
        Whf -= alpha * dWhf
        bf  -= alpha * dbf
        
        Wxo -= alpha * dWxo
        Who -= alpha * dWho
        bo  -= alpha * dbo
        
        Wxc -= alpha * dWxc
        Whc -= alpha * dWhc
        bc  -= alpha * dbc
        
        Why -= alpha * dWhy
        by  -= alpha * dby
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
            
    # Store parameters in model dictionary
    model = {
        'Wxi': Wxi, 'Whi': Whi, 'bi': bi,
        'Wxf': Wxf, 'Whf': Whf, 'bf': bf,
        'Wxo': Wxo, 'Who': Who, 'bo': bo,
        'Wxc': Wxc, 'Whc': Whc, 'bc': bc,
        'Why': Why, 'by': by,
        'hidden_size': hidden_size
    }
    return model

def lstm_predict(model, X):
    d, T = X.shape
    hidden_size = model['hidden_size']
    h_prev = np.zeros((hidden_size, 1))
    c_prev = np.zeros((hidden_size, 1))
    y_out = np.zeros((1, T))
    
    for t in range(T):
        x_t = X[:, t:t+1]
        i_t = sigmoid(model['Wxi'] @ x_t + model['Whi'] @ h_prev + model['bi'])
        f_t = sigmoid(model['Wxf'] @ x_t + model['Whf'] @ h_prev + model['bf'])
        o_t = sigmoid(model['Wxo'] @ x_t + model['Who'] @ h_prev + model['bo'])
        g_t = np.tanh(model['Wxc'] @ x_t + model['Whc'] @ h_prev + model['bc'])
        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * np.tanh(c_t)
        y_out[0, t] = (model['Why'] @ h_t + model['by'])[0, 0]
        h_prev = h_t
        c_prev = c_t
    return y_out

# Main script for LSTM tutorial
if __name__ == '__main__':
    T = 100  # number of time steps
    t_axis = np.linspace(0, 2*np.pi, T)
    X = np.sin(t_axis).reshape(1, T)
    # Target: phase-shifted sine with noise
    Y = (np.sin(t_axis + 0.5) + 0.1*np.random.randn(T)).reshape(1, T)
    
    hidden_size = 10
    alpha = 0.01
    epochs = 5000
    
    model = lstm_train(X, Y, hidden_size, alpha, epochs)
    Y_pred = lstm_predict(model, X)
    
    plt.figure()
    plt.plot(t_axis, Y.flatten(), 'b-', linewidth=2, label='True Values')
    plt.plot(t_axis, Y_pred.flatten(), 'r--', linewidth=2, label='Predicted Values')
    plt.xlabel("Time")
    plt.ylabel("Output")
    plt.title("LSTM Time–Series Prediction")
    plt.legend()
    plt.grid(True)
    plt.show()
