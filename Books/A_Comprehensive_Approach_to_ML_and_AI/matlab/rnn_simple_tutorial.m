close all; clear all; clc; 

% Example usage:
rng(1); % For reproducibility
% Generate synthetic time-series data: e.g., a sine wave with noise
T = 100;
t_axis = linspace(0, 2*pi, T);
X = sin(t_axis);  % 1 x T (1-dimensional input)
X = reshape(X, 1, T);  % Ensure proper shape: 1 x T
Y = sin(t_axis + 0.5) + 0.1*randn(1, T);  % Target: phase-shifted sine wave with noise

hiddenSize = 10;
alpha = 0.01;  % Adjusted learning rate for stability
epochs = 5000;
model = rnnTrain(X, Y, hiddenSize, alpha, epochs);
Y_pred = rnnPredict(model, X);

% Plot true vs. predicted time series
figure;
plot(t_axis, Y, 'b-', 'LineWidth', 2);
hold on;
plot(t_axis, Y_pred, 'r--', 'LineWidth', 2);
xlabel('Time');
ylabel('Output');
title('RNN Time-Series Prediction');
legend('True Values', 'Predicted Values');
grid on;

% save_all_figs_OPTION('results/rnnSimple','png',1)


function model = rnnTrain(X, Y, hiddenSize, alpha, epochs)
    % RNNTRAIN trains a simple RNN for time-series prediction.
    %   X: d x T input sequence (each column is a time step; d=dimensionality)
    %   Y: 1 x T target vector
    %   hiddenSize: number of neurons in the hidden layer
    %   alpha: learning rate
    %   epochs: number of training epochs
    %   model: structure containing trained parameters
    
    [d, T] = size(X);
    outputSize = 1; % For regression
    
    % Initialize weights and biases with small random values
    Wxh = randn(hiddenSize, d) * 0.01;      % Input to hidden
    Whh = randn(hiddenSize, hiddenSize) * 0.01;  % Hidden to hidden
    bh = zeros(hiddenSize, 1);
    Why = randn(outputSize, hiddenSize) * 0.01;  % Hidden to output
    by = zeros(outputSize, 1);
    
    for epoch = 1:epochs
        % Initialize hidden state for the current sequence
        h = zeros(hiddenSize, T);
        h_prev = zeros(hiddenSize, 1);
        
        % Forward pass
        y_pred = zeros(1, T);
        for t = 1:T
            h(:,t) = tanh(Wxh * X(:,t) + Whh * h_prev + bh);
            y_pred(t) = Why * h(:,t) + by;
            h_prev = h(:,t);
        end
        
        % Compute loss (Mean Squared Error)
        loss = 0.5 * sum((Y - y_pred).^2);
        
        % Initialize gradients
        dWxh = zeros(size(Wxh));
        dWhh = zeros(size(Whh));
        dbh = zeros(size(bh));
        dWhy = zeros(size(Why));
        dby = zeros(size(by));
        dh_next = zeros(hiddenSize,1);
        
        % Backpropagation Through Time (BPTT)
        for t = T:-1:1
            dy = y_pred(t) - Y(t);
            dWhy = dWhy + dy * h(:,t)';
            dby = dby + dy;
            
            dh = (Why' * dy) + dh_next;
            dtanh = (1 - h(:,t).^2) .* dh;
            
            dWxh = dWxh + dtanh * X(:,t)';
            dbh = dbh + dtanh;
            if t > 1
                dWhh = dWhh + dtanh * h(:,t-1)';
                dh_next = Whh' * dtanh;
            end
        end
        
        % Average gradients over time steps to prevent exploding gradients
        dWxh = dWxh / T;
        dWhh = dWhh / T;
        dbh  = dbh  / T;
        dWhy = dWhy / T;
        dby  = dby  / T;
        
        % Update parameters
        Wxh = Wxh - alpha * dWxh;
        Whh = Whh - alpha * dWhh;
        bh  = bh  - alpha * dbh;
        Why = Why - alpha * dWhy;
        by  = by  - alpha * dby;
        
        if mod(epoch, 100) == 0
            fprintf('Epoch %d, Loss: %.4f\n', epoch, loss);
        end
    end
    
    % Store trained parameters in the model structure
    model.Wxh = Wxh;
    model.Whh = Whh;
    model.bh = bh;
    model.Why = Why;
    model.by = by;
    model.hiddenSize = hiddenSize;
end


function y_out = rnnPredict(model, X)
    % RNNPREDICT predicts outputs for input sequence X using the trained RNN model.
    [d, T] = size(X);
    h_prev = zeros(model.hiddenSize, 1);
    y_out = zeros(1, T);
    for t = 1:T
        h = tanh(model.Wxh * X(:,t) + model.Whh * h_prev + model.bh);
        y_out(t) = model.Why * h + model.by;
        h_prev = h;
    end
end