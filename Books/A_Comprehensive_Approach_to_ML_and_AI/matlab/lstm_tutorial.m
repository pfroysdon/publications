% LSTM Time–Series Prediction Tutorial (from scratch)
%
% This tutorial demonstrates how to implement a simple LSTM network for
% time–series prediction without using MATLAB’s deep learning libraries.
% We generate a synthetic sine–wave signal with noise, train the LSTM to
% predict the signal, and then plot the true vs. predicted values.

clear; close all; clc;

% Generate Synthetic Data
T = 100;                           % Number of time steps
t_axis = linspace(0, 2*pi, T);       % Time axis
X = sin(t_axis);                   % Input: sine wave
X = reshape(X, 1, T);              % Ensure X is 1 x T
Y = sin(t_axis + 0.5) + 0.1*randn(1, T);  % Target: phase-shifted sine with noise

% Set LSTM Hyperparameters
hiddenSize = 10;
alpha = 0.01;      % Learning rate
epochs = 5000;     % Number of training epochs

% Train the LSTM
model = lstmTrain(X, Y, hiddenSize, alpha, epochs);

% Predict Using the Trained LSTM
Y_pred = lstmPredict(model, X);

% Plot True vs. Predicted Time Series
figure;
plot(t_axis, Y, 'b-', 'LineWidth', 2);
hold on;
plot(t_axis, Y_pred, 'r--', 'LineWidth', 2);
xlabel('Time');
ylabel('Output');
title('LSTM Time–Series Prediction');
legend('True Values', 'Predicted Values');
grid on;

% save_all_figs_OPTION('results/lstm','png',1)


function model = lstmTrain(X, Y, hiddenSize, alpha, epochs)
    % lstmTrain trains an LSTM from scratch for time–series prediction.
    %   X: d x T input sequence (each column is a time step; here d=1)
    %   Y: 1 x T target sequence
    %   hiddenSize: number of LSTM neurons
    %   alpha: learning rate
    %   epochs: number of training epochs
    %   model: structure containing the trained parameters
    
    [d, T] = size(X);
    outputSize = 1;  % Regression output
    
    % Initialize LSTM parameters (small random values)
    Wxi = randn(hiddenSize, d) * 0.01;  % Input gate: input weights
    Whi = randn(hiddenSize, hiddenSize) * 0.01;  % Input gate: hidden weights
    bi  = zeros(hiddenSize, 1);
    
    Wxf = randn(hiddenSize, d) * 0.01;  % Forget gate
    Whf = randn(hiddenSize, hiddenSize) * 0.01;
    bf  = zeros(hiddenSize, 1);
    
    Wxo = randn(hiddenSize, d) * 0.01;  % Output gate
    Who = randn(hiddenSize, hiddenSize) * 0.01;
    bo  = zeros(hiddenSize, 1);
    
    Wxc = randn(hiddenSize, d) * 0.01;  % Candidate (cell) gate
    Whc = randn(hiddenSize, hiddenSize) * 0.01;
    bc  = zeros(hiddenSize, 1);
    
    % Output layer parameters
    Why = randn(outputSize, hiddenSize) * 0.01;
    by  = zeros(outputSize, 1);
    
    for epoch = 1:epochs
        % Forward Pass: initialize states
        h = zeros(hiddenSize, T);
        c = zeros(hiddenSize, T);
        h_prev = zeros(hiddenSize, 1);
        c_prev = zeros(hiddenSize, 1);
        y_pred = zeros(1, T);
        
        % For storing gate activations (for BPTT)
        i_store = zeros(hiddenSize, T);
        f_store = zeros(hiddenSize, T);
        o_store = zeros(hiddenSize, T);
        g_store = zeros(hiddenSize, T);
        
        for t = 1:T
            x_t = X(:, t);  % current input (d x 1)
            % LSTM gate computations
            i_t = sigmoid(Wxi * x_t + Whi * h_prev + bi);  % input gate
            f_t = sigmoid(Wxf * x_t + Whf * h_prev + bf);    % forget gate
            o_t = sigmoid(Wxo * x_t + Who * h_prev + bo);    % output gate
            g_t = tanh(Wxc * x_t + Whc * h_prev + bc);         % candidate cell state
            
            % Cell state and hidden state updates
            c_t = f_t .* c_prev + i_t .* g_t;
            h_t = o_t .* tanh(c_t);
            
            % Store activations for backprop
            i_store(:, t) = i_t;
            f_store(:, t) = f_t;
            o_store(:, t) = o_t;
            g_store(:, t) = g_t;
            c(:, t) = c_t;
            h(:, t) = h_t;
            
            % Output prediction at time t
            y_pred(t) = Why * h_t + by;
            
            % Update previous states
            h_prev = h_t;
            c_prev = c_t;
        end
        
        % Compute loss (Mean Squared Error)
        loss = 0.5 * sum((Y - y_pred).^2);
        
        % Backward Pass: initialize gradients
        dWxi = zeros(size(Wxi)); dWhi = zeros(size(Whi)); dbi = zeros(size(bi));
        dWxf = zeros(size(Wxf)); dWhf = zeros(size(Whf)); dbf = zeros(size(bf));
        dWxo = zeros(size(Wxo)); dWho = zeros(size(Who)); dbo = zeros(size(bo));
        dWxc = zeros(size(Wxc)); dWhc = zeros(size(Whc)); dbc = zeros(size(bc));
        dWhy = zeros(size(Why)); dby = zeros(size(by));
        
        dh_next = zeros(hiddenSize, 1);
        dc_next = zeros(hiddenSize, 1);
        
        % Backpropagation Through Time (BPTT)
        for t = T:-1:1
            dy = y_pred(t) - Y(t);  % scalar error at time t
            dWhy = dWhy + dy * h(:, t)';  % (outputSize x hiddenSize)
            dby = dby + dy;
            
            % Backprop into h_t: add gradient from output layer and future time steps
            dh = (Why' * dy) + dh_next;  % (hiddenSize x 1)
            
            % h_t = o_t .* tanh(c_t)
            do = dh .* tanh(c(:, t));
            do = do .* o_store(:, t) .* (1 - o_store(:, t));  % derivative of sigmoid
            
            % Backprop through tanh(c_t)
            dct = dh .* o_store(:, t) .* (1 - tanh(c(:, t)).^2) + dc_next;
            
            % c_t = f_t .* c_prev + i_t .* g_t
            di = dct .* g_store(:, t);
            % Replace ternary operator with if–else:
            if t == 1
                c_prev_val = zeros(hiddenSize,1);
            else
                c_prev_val = c(:, t-1);
            end
            df = dct .* c_prev_val;
            dg = dct .* i_store(:, t);
            dc_prev = dct .* f_store(:, t);
            
            % Backprop through activations:
            di = di .* i_store(:, t) .* (1 - i_store(:, t));  % sigmoid derivative
            df = df .* f_store(:, t) .* (1 - f_store(:, t));
            dg = dg .* (1 - g_store(:, t).^2);  % tanh derivative
            
            % Get x_t and h_prev (from time t-1)
            x_t = X(:, t);
            if t == 1
                h_prev_t = zeros(hiddenSize, 1);
            else
                h_prev_t = h(:, t-1);
            end
            
            % Accumulate gradients for gate parameters
            dWxi = dWxi + di * x_t';
            dWhi = dWhi + di * h_prev_t';
            dbi = dbi + di;
            
            dWxf = dWxf + df * x_t';
            dWhf = dWhf + df * h_prev_t';
            dbf = dbf + df;
            
            dWxo = dWxo + do * x_t';
            dWho = dWho + do * h_prev_t';
            dbo = dbo + do;
            
            dWxc = dWxc + dg * x_t';
            dWhc = dWhc + dg * h_prev_t';
            dbc = dbc + dg;
            
            % Propagate gradients to previous time step
            dh_next = Whi' * di + Whf' * df + Who' * do + Whc' * dg;
            dc_next = dc_prev;
        end
        
        % Average gradients over time steps
        dWxi = dWxi / T;
        dWhi = dWhi / T;
        dbi = dbi / T;
        dWxf = dWxf / T;
        dWhf = dWhf / T;
        dbf = dbf / T;
        dWxo = dWxo / T;
        dWho = dWho / T;
        dbo = dbo / T;
        dWxc = dWxc / T;
        dWhc = dWhc / T;
        dbc = dbc / T;
        dWhy = dWhy / T;
        dby = dby / T;
        
        % Update parameters
        Wxi = Wxi - alpha * dWxi;
        Whi = Whi - alpha * dWhi;
        bi = bi - alpha * dbi;
        
        Wxf = Wxf - alpha * dWxf;
        Whf = Whf - alpha * dWhf;
        bf = bf - alpha * dbf;
        
        Wxo = Wxo - alpha * dWxo;
        Who = Who - alpha * dWho;
        bo = bo - alpha * dbo;
        
        Wxc = Wxc - alpha * dWxc;
        Whc = Whc - alpha * dWhc;
        bc = bc - alpha * dbc;
        
        Why = Why - alpha * dWhy;
        by = by - alpha * dby;
        
        if mod(epoch, 100) == 0
            fprintf('Epoch %d, Loss: %.4f\n', epoch, loss);
        end
    end
    
    % Store trained parameters in the model structure
    model.Wxi = Wxi;
    model.Whi = Whi;
    model.bi = bi;
    model.Wxf = Wxf;
    model.Whf = Whf;
    model.bf = bf;
    model.Wxo = Wxo;
    model.Who = Who;
    model.bo = bo;
    model.Wxc = Wxc;
    model.Whc = Whc;
    model.bc = bc;
    model.Why = Why;
    model.by = by;
    model.hiddenSize = hiddenSize;
end

function y_out = lstmPredict(model, X)
    % lstmPredict generates predictions for input sequence X using the trained LSTM.
    [d, T] = size(X);
    h_prev = zeros(model.hiddenSize, 1);
    c_prev = zeros(model.hiddenSize, 1);
    y_out = zeros(1, T);
    for t = 1:T
        x_t = X(:, t);
        i_t = sigmoid(model.Wxi * x_t + model.Whi * h_prev + model.bi);
        f_t = sigmoid(model.Wxf * x_t + model.Whf * h_prev + model.bf);
        o_t = sigmoid(model.Wxo * x_t + model.Who * h_prev + model.bo);
        g_t = tanh(model.Wxc * x_t + model.Whc * h_prev + model.bc);
        c_t = f_t .* c_prev + i_t .* g_t;
        h_t = o_t .* tanh(c_t);
        y_out(t) = model.Why * h_t + model.by;
        h_prev = h_t;
        c_prev = c_t;
    end
end

function s = sigmoid(x)
    s = 1 ./ (1 + exp(-x));
end
