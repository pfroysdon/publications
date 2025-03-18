%% climateAdvancedTutorial.m
% Advanced Climate Forecasting with Dense, RNN, and LSTM models
% using mini-batch training, Adam optimization, and full BPTT.
% 
% Below is an advanced MATLAB tutorial that builds on the previous example 
% but now uses:
% 
%     Mini–batch training (instead of processing one sample at a time)
%     Advanced gradient–descent loops (here we implement the Adam optimizer 
%     for parameter updates)
%     Full back–propagation through time (BPTT) for both the RNN and LSTM 
%     models
% 
% In this tutorial we forecast temperature using the Jena climate dataset 
% from jena_climate_2009_2016.csv.
% We will implement three models from scratch:
% 
%     A densely connected (feed–forward) model
%     A simple RNN model
%     A simple LSTM model
% 
% Each model’s training function uses mini–batches and the Adam optimizer. 
% The RNN and LSTM training functions implement full BPTT (i.e. gradients 
% are back–propagated through the entire unrolled sequence).
% 
% For clarity the code is split into separate files.
% 
%
% This tutorial uses the Jena climate dataset (jena_climate_2009_2016.csv)
% to forecast temperature. We:
%   1. Parse and plot the temperature time–series.
%   2. Window the data into samples for training, validation, and testing.
%   3. Normalize using training–set statistics.
%   4. Compute a common–sense baseline MAE.
%   5. Train a densely connected model (advanced mini–batch + Adam).
%   6. Train an RNN model (advanced mini–batch + full BPTT + Adam).
%   7. Train an LSTM model (advanced mini–batch + full BPTT + Adam).
%   8. Plot and compare the predictions on the validation set.
%
% Final Comments
% Mini–Batch and Adam:
% In each training function the data is shuffled and processed in mini–
% batches. The Adam optimizer (with momentum and adaptive learning rates) 
% is used to update model parameters.
% 
% Full BPTT:
% Both the RNN and LSTM training functions back–propagate gradients through 
% the entire unrolled sequence. (The LSTM implementation is simplified for 
% tutorial purposes; a production–quality implementation would include 
% additional refinements.)
% 
% Normalization:
% Remember that predictions are produced on normalized data. To recover 
% original units, you must “undo” the normalization using the training mean 
% and standard deviation.

clear; close all; clc;

%% 1. Load and Parse the Data
filename = 'data/jena_climate_2009_2016.csv';
if ~exist(filename,'file')
    error('File %s not found. Download and unzip from the provided URL.', filename);
end

% Read CSV into a table.
T = readtable(filename);
% Convert date strings to datetime objects.
dt = datetime(T{:,1},'InputFormat','yyyy-MM-dd HH:mm:ss');
% Extract temperature (assume column 3 holds temperature in °C).
temp = T{:,3};

fprintf('Loaded %d temperature samples spanning from %s to %s\n', ...
    length(temp), datestr(dt(1)), datestr(dt(end)));

%% 2. Plot the Temperature Time–Series
figure;
plot(dt, temp);
xlabel('Time'); ylabel('Temperature (°C)');
title('Temperature Time–Series (All Data)');

%% 3. Plot the First 10 Days of Temperature
% Data is recorded every 10 minutes; 10 days ≈ 10*24*6 = 1440 samples.
num_first = 1440;
figure;
plot(dt(1:num_first), temp(1:num_first));
xlabel('Time'); ylabel('Temperature (°C)');
title('Temperature Time–Series (First 10 Days)');

%% 4. Prepare the Data (Windowing)
% We forecast temperature "delay" timesteps ahead using the past "lookback" timesteps.
lookback = 720;  % e.g., past 720 timesteps (~5 days if 10-min intervals)
delay    = 144;  % e.g., predict 144 timesteps ahead (~1 day)
step     = 6;    % sample every 6 timesteps (hourly samples)

num_data = length(temp);
% Use similar splits as in the Keras example.
train_end = 200000;
val_end   = 300000;
if num_data < train_end+lookback+delay
    error('Not enough data for these window parameters. Adjust lookback/delay.');
end

% Create datasets. (The function create_dataset.m windows the data.)
[X_train, y_train] = create_dataset(temp, lookback, delay, step, 1, train_end);
[X_val, y_val]     = create_dataset(temp, lookback, delay, step, train_end+1, val_end);
[X_test, y_test]   = create_dataset(temp, lookback, delay, step, val_end+1, num_data);

fprintf('Training samples: %d\n', size(X_train,1));
fprintf('Validation samples: %d\n', size(X_val,1));
fprintf('Testing samples: %d\n', size(X_test,1));

%% 5. Normalize the Data (Using training–set statistics)
% X_* are 3D arrays: (samples x timesteps x 1)
train_mean = mean(X_train(:));
train_std  = std(X_train(:));

X_train = (X_train - train_mean) / train_std;
X_val   = (X_val   - train_mean) / train_std;
X_test  = (X_test  - train_mean) / train_std;

y_train = (y_train - train_mean) / train_std;
y_val   = (y_val   - train_mean) / train_std;
y_test  = (y_test  - train_mean) / train_std;

%% 6. Compute the Common–Sense Baseline MAE
% Baseline: predict the future temperature equals the last value of the window.
y_pred_baseline = squeeze(X_val(:,end,:));
baseline_MAE = mean(abs(y_pred_baseline - y_val));
fprintf('Baseline MAE (normalized): %.4f\n', baseline_MAE);

%% 7. Train a Densely Connected (Feed–Forward) Model
denseOpts.learning_rate = 1e-3;
denseOpts.epochs = 10;
denseOpts.hidden_dim = 32;
denseOpts.batch_size = 32;

[denseModel, train_loss_dense, val_loss_dense, y_pred_dense] = ...
    trainDenseModel(X_train, y_train, X_val, y_val, denseOpts);
dense_MAE = mean(abs(y_pred_dense - y_val));
fprintf('Dense model MAE (normalized): %.4f\n', dense_MAE);

%% 8. Train an RNN Model (Advanced: mini–batch + full BPTT + Adam)
rnnOpts.learning_rate = 1e-3;
rnnOpts.epochs = 10;
rnnOpts.hidden_size = 32;
rnnOpts.batch_size = 32;
[rnnModel, train_loss_rnn, val_loss_rnn, y_pred_rnn] = ...
    trainRNN(X_train, y_train, X_val, y_val, rnnOpts);
rnn_MAE = mean(abs(y_pred_rnn - y_val));
fprintf('RNN model MAE (normalized): %.4f\n', rnn_MAE);

%% 9. Train an LSTM Model (Advanced: mini–batch + full BPTT + Adam)
lstmOpts.learning_rate = 1e-3;
lstmOpts.epochs = 10;
lstmOpts.hidden_size = 32;
lstmOpts.batch_size = 32;
[lstmModel, train_loss_lstm, val_loss_lstm, y_pred_lstm] = ...
    trainLSTM(X_train, y_train, X_val, y_val, lstmOpts);
lstm_MAE = mean(abs(y_pred_lstm - y_val));
fprintf('LSTM model MAE (normalized): %.4f\n', lstm_MAE);

%% 10. Plot the Results on the Validation Set
t = 1:length(y_val);
figure;
plot(t, y_val, 'k', 'LineWidth', 1.5); hold on;
plot(t, y_pred_baseline, 'b');
plot(t, y_pred_dense, 'r');
plot(t, y_pred_rnn, 'g');
plot(t, y_pred_lstm, 'm');
xlabel('Validation Sample Index'); ylabel('Normalized Temperature');
legend('True','Baseline','Dense','RNN','LSTM');
title('Model Predictions on Validation Set');
grid on;


figure;
plot(t, y_val, 'k', 'LineWidth', 1.5); hold on;
plot(t, y_pred_rnn, 'r');
xlabel('Validation Sample Index'); ylabel('Normalized Temperature');
legend('True','RNN');
title('RNN Model Predictions on Validation Set');
grid on;

% save_all_figs_OPTION('results/rnnAdvanced','png',1)


figure;
plot(t, y_val, 'k', 'LineWidth', 1.5); hold on;
plot(t, y_pred_lstm, 'r');
xlabel('Validation Sample Index'); ylabel('Normalized Temperature');
legend('True','LSTM');
title('LSTM Model Predictions on Validation Set');
grid on;

% save_all_figs_OPTION('results/lstmAdvanced','png',1)





function [X, y] = create_dataset(data, lookback, delay, step, start_index, end_index)
% create_dataset constructs time–series samples from raw data.
%
%   [X, y] = create_dataset(data, lookback, delay, step, start_index, end_index)
%
% Inputs:
%   data        - vector of data (e.g. temperature)
%   lookback    - number of timesteps to include in the input window
%   delay       - number of timesteps in the future to predict
%   step        - period (in timesteps) between successive samples in the window
%   start_index - starting index in data for windowing
%   end_index   - ending index in data for windowing
%
% Outputs:
%   X - 3D array of shape (num_samples, timesteps, 1)
%   y - vector of targets (num_samples x 1)

num_samples = floor((end_index - start_index - lookback - delay) / 1) + 1;
num_timesteps = floor(lookback / step);

X = zeros(num_samples, num_timesteps, 1);
y = zeros(num_samples, 1);

for i = 1:num_samples
    idx = start_index + i - 1;
    indices = idx:step:(idx + lookback - 1);
    X(i, :, 1) = data(indices);
    y(i) = data(idx + lookback + delay - 1);
end
end


function [model, train_losses, val_losses, y_pred_val] = trainDenseModel(X_train, y_train, X_val, y_val, opts)
% trainDenseModel_advanced trains a simple dense network with mini-batches and Adam.
%
%   [model, train_losses, val_losses, y_pred_val] = trainDenseModel_advanced(X_train, y_train, X_val, y_val, opts)
%
% Inputs:
%   X_train, y_train - training data (X_train is 3D: samples x timesteps x 1)
%   X_val, y_val     - validation data
%   opts             - structure with fields: learning_rate, epochs, hidden_dim, batch_size
%
% Outputs:
%   model       - structure with parameters (W1, b1, W2, b2)
%   train_losses- vector of training losses per epoch
%   val_losses  - vector of validation losses per epoch
%   y_pred_val- predictions on the validation set at training end

% Get dimensions.
[num_train, timesteps, ~] = size(X_train);
input_dim = timesteps;  % flatten input window
hidden_dim = opts.hidden_dim;
output_dim = 1;
batch_size = opts.batch_size;
learning_rate = opts.learning_rate;
epochs = opts.epochs;

% Initialize weights.
W1 = 0.01*randn(hidden_dim, input_dim);
b1 = zeros(hidden_dim, 1);
W2 = 0.01*randn(output_dim, hidden_dim);
b2 = zeros(output_dim, 1);

% Adam hyperparameters.
beta1 = 0.9; beta2 = 0.999; epsilon = 1e-8;
[mW1, vW1] = deal(zeros(size(W1)));
[mb1, vb1] = deal(zeros(size(b1)));
[mW2, vW2] = deal(zeros(size(W2)));
[mb2, vb2] = deal(zeros(size(b2)));

train_losses = zeros(epochs,1);
val_losses = zeros(epochs,1);

num_batches = floor(num_train / batch_size);

for ep = 1:epochs
    % Shuffle training indices.
    idx = randperm(num_train);
    epoch_loss = 0;
    
    for b = 1:num_batches
        batch_idx = idx((b-1)*batch_size+1:b*batch_size);
        % Get mini-batch and reshape: X_batch: (batch_size x input_dim)
        X_batch = reshape(X_train(batch_idx,:,:), [batch_size, input_dim]);
        y_batch = y_train(batch_idx);
        
        %% Forward Pass (vectorized)
        z1 = X_batch * W1.' + repmat(b1.', batch_size, 1);  % (batch_size x hidden_dim)
        a1 = max(z1, 0);  % ReLU activation
        z2 = a1 * W2.' + repmat(b2.', batch_size, 1);  % (batch_size x 1)
        y_pred = z2;
        
        % Mean–squared error loss.
        loss = sum(0.5*(y_pred - y_batch).^2) / batch_size;
        epoch_loss = epoch_loss + loss;
        
        %% Backward Pass
        dL_dy = (y_pred - y_batch) / batch_size;  % (batch_size x 1)
        dW2 = dL_dy.' * a1;  % (1 x hidden_dim)
        db2 = sum(dL_dy, 1).';  % (hidden_dim x 1) note: db2 is (1x1) here
        da1 = dL_dy * W2;  % (batch_size x hidden_dim)
        dz1 = da1 .* double(z1 > 0);  % ReLU derivative
        dW1 = dz1.' * X_batch;  % (hidden_dim x input_dim)
        db1 = sum(dz1, 1).';  % (hidden_dim x 1)
        
        %% Adam Parameter Updates
        % Update for W1
        mW1 = beta1 * mW1 + (1-beta1)*dW1;
        vW1 = beta2 * vW1 + (1-beta2)*(dW1.^2);
        mW1_hat = mW1 / (1 - beta1^ep);
        vW1_hat = vW1 / (1 - beta2^ep);
        W1 = W1 - learning_rate * mW1_hat ./ (sqrt(vW1_hat) + epsilon);
        
        % Update for b1
        mb1 = beta1 * mb1 + (1-beta1)*db1;
        vb1 = beta2 * vb1 + (1-beta2)*(db1.^2);
        mb1_hat = mb1 / (1 - beta1^ep);
        vb1_hat = vb1 / (1 - beta2^ep);
        b1 = b1 - learning_rate * mb1_hat ./ (sqrt(vb1_hat) + epsilon);
        
        % Update for W2
        mW2 = beta1 * mW2 + (1-beta1)*dW2;
        vW2 = beta2 * vW2 + (1-beta2)*(dW2.^2);
        mW2_hat = mW2 / (1 - beta1^ep);
        vW2_hat = vW2 / (1 - beta2^ep);
        W2 = W2 - learning_rate * mW2_hat ./ (sqrt(vW2_hat) + epsilon);
        
        % Update for b2
        mb2 = beta1 * mb2 + (1-beta1)*db2;
        vb2 = beta2 * vb2 + (1-beta2)*(db2.^2);
        mb2_hat = mb2 / (1 - beta1^ep);
        vb2_hat = vb2 / (1 - beta2^ep);
        b2 = b2 - learning_rate * mb2_hat ./ (sqrt(vb2_hat) + epsilon);
    end
    
    train_losses(ep) = epoch_loss / num_batches;
    
    % Validation (forward pass over all validation samples)
    num_val = size(X_val,1);
    X_val_flat = reshape(X_val, [num_val, input_dim]);
    z1_val = X_val_flat * W1.' + repmat(b1.', num_val, 1);
    a1_val = max(z1_val, 0);
    z2_val = a1_val * W2.' + repmat(b2.', num_val, 1);
    y_val_pred = z2_val;
    val_loss = sum(0.5*(y_val_pred - y_val).^2)/num_val;
    val_losses(ep) = val_loss;
    
    fprintf('Dense Epoch %d/%d: Train Loss = %.4f, Val Loss = %.4f\n', ep, epochs, train_losses(ep), val_losses(ep));
end

% Final predictions on validation set.
y_pred_val = y_val_pred;
model.W1 = W1; model.b1 = b1; model.W2 = W2; model.b2 = b2;
end



function [model, train_losses, val_losses, y_pred_val] = trainRNN(X_train, y_train, X_val, y_val, opts)
% trainRNNModel_advanced trains a simple RNN using mini-batches, full BPTT, and Adam.
%
%   [model, train_losses, val_losses, y_pred_val] = trainRNNModel_advanced(X_train, y_train, X_val, y_val, opts)
%
% Inputs:
%   X_train, y_train - training data (X_train is 3D: samples x T x 1)
%   X_val, y_val     - validation data
%   opts             - structure with fields: learning_rate, epochs, hidden_size, batch_size
%
% Outputs:
%   model       - structure with parameters (Wxh, Whh, Why, bh, by)
%   train_losses- vector of training losses per epoch
%   val_losses  - vector of validation losses per epoch
%   y_pred_val  - predictions on the validation set (vector)

[num_train, T, ~] = size(X_train);
hidden_size = opts.hidden_size;
input_size = 1;
output_size = 1;
batch_size = opts.batch_size;
learning_rate = opts.learning_rate;
epochs = opts.epochs;

% Initialize weights.
Wxh = 0.01 * randn(hidden_size, input_size);
Whh = 0.01 * randn(hidden_size, hidden_size);
bh  = zeros(hidden_size, 1);
Why = 0.01 * randn(output_size, hidden_size);
by  = zeros(output_size, 1);

% Adam hyperparameters.
beta1 = 0.9; beta2 = 0.999; epsilon = 1e-8;
[mWxh, vWxh] = deal(zeros(size(Wxh)));
[mWhh, vWhh] = deal(zeros(size(Whh)));
[mbh,  vbh]  = deal(zeros(size(bh)));
[mWhy, vWhy]  = deal(zeros(size(Why)));
[mby,  vby]  = deal(zeros(size(by)));

train_losses = zeros(epochs,1);
val_losses = zeros(epochs,1);
num_batches = floor(num_train / batch_size);

for ep = 1:epochs
    idx = randperm(num_train);
    epoch_loss = 0;
    
    for b = 1:num_batches
        batch_idx = idx((b-1)*batch_size+1:b*batch_size);
        % Extract mini-batch and reshape: X_batch is (batch_size x T)
        X_batch = reshape(X_train(batch_idx,:,:), [batch_size, T]);
        y_batch = y_train(batch_idx);  % (batch_size x 1)
        
        % Forward pass: Process the sequence for the entire mini-batch.
        % We'll store hidden states in H: (hidden_size x (T+1) x batch_size)
        H = zeros(hidden_size, T+1, batch_size);
        H(:,1,:) = 0;  % initial hidden state
        
        for t = 1:T
            % Extract x_t: reshape X_batch(:, t) (batch_size x 1) into (1 x batch_size)
            x_t = reshape(X_batch(:, t), [1, batch_size]);
            % Get previous hidden state (make sure it is 2D: hidden_size x batch_size)
            h_prev = reshape(H(:, t, :), [hidden_size, batch_size]);
            % Compute the linear combination:
            temp = Wxh * x_t + Whh * h_prev + repmat(bh, 1, batch_size);
            % Compute h_t = tanh(temp)
            h_t = tanh(temp);  % (hidden_size x batch_size)
            % Store h_t back into H, reshaped to 3D.
            H(:, t+1, :) = reshape(h_t, [hidden_size, 1, batch_size]);
        end
        
        % Use the final hidden state for prediction.
        H_final = reshape(H(:, end, :), [hidden_size, batch_size]);  % (hidden_size x batch_size)
        y_pred = (Why * H_final + repmat(by, 1, batch_size)).';  % (batch_size x 1)
        
        % Compute mean-squared error loss.
        loss = sum(0.5*(y_pred - y_batch).^2) / batch_size;
        epoch_loss = epoch_loss + loss;
        
        %% Backward Pass: Full BPTT
        % Initialize gradients.
        dWxh = zeros(size(Wxh));
        dWhh = zeros(size(Whh));
        dbh  = zeros(size(bh));
        dWhy = zeros(size(Why));
        dby  = zeros(size(by));
        
        % Gradient of loss w.r.t. output.
        dy = (y_pred - y_batch) / batch_size;  % (batch_size x 1)
        % Gradients for output layer.
        dWhy = dWhy + dy.' * reshape(H(:, end, :), [hidden_size, batch_size]).';
        dby  = dby + sum(dy, 1).';
        
        % Backpropagate into final hidden state.
        dH_final = Why.' * dy.';  % (hidden_size x batch_size)
        
        % Initialize dH: same shape as H.
        dH = zeros(hidden_size, T+1, batch_size);
        dH(:, end, :) = reshape(dH_final, [hidden_size, 1, batch_size]);
        
        % Backpropagation through time.
        for t = T:-1:1
            % Retrieve h at time t+1 and h_prev (both as 2D arrays)
            h = reshape(H(:, t+1, :), [hidden_size, batch_size]);       % (hidden_size x batch_size)
            h_prev = reshape(H(:, t, :), [hidden_size, batch_size]);      % (hidden_size x batch_size)
            % Retrieve dH for time t+1 as a 2D array.
            dH_t = reshape(dH(:, t+1, :), [hidden_size, batch_size]);
            % Backprop through tanh: derivative = (1 - h.^2)
            dtanh = (1 - h.^2) .* dH_t;  % (hidden_size x batch_size)
            
            % x_t for time t.
            x_t = reshape(X_batch(:, t), [1, batch_size]);  % (1 x batch_size)
            
            % Accumulate gradients.
            dWxh = dWxh + dtanh * x_t.';       % (hidden_size x 1)
            dWhh = dWhh + dtanh * h_prev.';      % (hidden_size x hidden_size)
            dbh = dbh + sum(dtanh, 2);           % (hidden_size x 1)
            
            % Propagate gradient to previous hidden state.
            dh_prev = Whh.' * dtanh;             % (hidden_size x batch_size)
            % Add the backpropagated gradient into dH for time t.
            dH(:, t, :) = dH(:, t, :) + reshape(dh_prev, [hidden_size, 1, batch_size]);
        end
        
        %% Adam Updates for RNN parameters
        % Update Wxh.
        mWxh = beta1 * mWxh + (1-beta1)*dWxh;
        vWxh = beta2 * vWxh + (1-beta2)*(dWxh.^2);
        mWxh_hat = mWxh / (1 - beta1^ep);
        vWxh_hat = vWxh / (1 - beta2^ep);
        Wxh = Wxh - learning_rate * mWxh_hat ./ (sqrt(vWxh_hat) + epsilon);
        
        % Update Whh.
        mWhh = beta1 * mWhh + (1-beta1)*dWhh;
        vWhh = beta2 * vWhh + (1-beta2)*(dWhh.^2);
        mWhh_hat = mWhh / (1 - beta1^ep);
        vWhh_hat = vWhh / (1 - beta2^ep);
        Whh = Whh - learning_rate * mWhh_hat ./ (sqrt(vWhh_hat) + epsilon);
        
        % Update bh.
        mbh = beta1 * mbh + (1-beta1)*dbh;
        vbh = beta2 * vbh + (1-beta2)*(dbh.^2);
        mbh_hat = mbh / (1 - beta1^ep);
        vbh_hat = vbh / (1 - beta2^ep);
        bh = bh - learning_rate * mbh_hat ./ (sqrt(vbh_hat) + epsilon);
        
        % Update Why.
        mWhy = beta1 * mWhy + (1-beta1)*dWhy;
        vWhy = beta2 * vWhy + (1-beta2)*(dWhy.^2);
        mWhy_hat = mWhy / (1 - beta1^ep);
        vWhy_hat = vWhy / (1 - beta2^ep);
        Why = Why - learning_rate * mWhy_hat ./ (sqrt(vWhy_hat) + epsilon);
        
        % Update by.
        mby = beta1 * mby + (1-beta1)*dby;
        vby = beta2 * vby + (1-beta2)*(dby.^2);
        mby_hat = mby / (1 - beta1^ep);
        vby_hat = vby / (1 - beta2^ep);
        by = by - learning_rate * mby_hat ./ (sqrt(vby_hat) + epsilon);
    end
    
    train_losses(ep) = epoch_loss / num_batches;
    
    % Validation pass: process one sample at a time.
    num_val = size(X_val,1);
    val_preds = zeros(num_val,1);
    for i = 1:num_val
        x_seq = reshape(X_val(i,:,:), [1, T]);  % (1 x T)
        h = zeros(hidden_size, 1);
        for t = 1:T
            x_t = x_seq(:, t);  % scalar (1x1)
            h = tanh(Wxh * x_t + Whh * h + bh);
        end
        val_preds(i) = (Why * h + by);
    end
    val_loss = sum(0.5*(val_preds - y_val).^2) / num_val;
    val_losses(ep) = val_loss;
    
    fprintf('RNN Epoch %d/%d: Train Loss = %.4f, Val Loss = %.4f\n', ...
        ep, epochs, train_losses(ep), val_losses(ep));
end

% Final predictions on validation set.
y_pred_val = val_preds;
model.Wxh = Wxh; model.Whh = Whh; model.bh = bh;
model.Why = Why; model.by = by;
end



function [model, train_losses, val_losses, y_pred_val] = trainLSTM(X_train, y_train, X_val, y_val, opts)
% trainLSTMModel_advanced trains an LSTM using mini-batches, full BPTT, and Adam.
%
%   [model, train_losses, val_losses, y_pred_val] = trainLSTMModel_advanced(X_train, y_train, X_val, y_val, opts)
%
% Inputs:
%   X_train, y_train - training data (X_train is 3D: samples x T x 1)
%   X_val, y_val     - validation data
%   opts             - structure with fields: learning_rate, epochs, hidden_size, batch_size
%
% Outputs:
%   model       - structure with LSTM parameters and output layer parameters
%   train_losses- vector of training losses per epoch
%   val_losses  - vector of validation losses per epoch
%   y_pred_val  - predictions on the validation set

[num_train, T, ~] = size(X_train);
hidden_size = opts.hidden_size;
input_size = 1;
output_size = 1;
batch_size = opts.batch_size;
learning_rate = opts.learning_rate;
epochs = opts.epochs;

% Initialize LSTM weights.
Wxi = 0.01 * randn(hidden_size, input_size);
Whi = 0.01 * randn(hidden_size, hidden_size);
bi  = zeros(hidden_size, 1);

Wxf = 0.01 * randn(hidden_size, input_size);
Whf = 0.01 * randn(hidden_size, hidden_size);
bf  = zeros(hidden_size, 1);

Wxo = 0.01 * randn(hidden_size, input_size);
Who = 0.01 * randn(hidden_size, hidden_size);
bo  = zeros(hidden_size, 1);

Wxc = 0.01 * randn(hidden_size, input_size);
Whc = 0.01 * randn(hidden_size, hidden_size);
bc  = zeros(hidden_size, 1);

% Output layer parameters.
Wy = 0.01 * randn(output_size, hidden_size);
by = zeros(output_size, 1);

% Initialize Adam variables for all parameters.
params = {'Wxi','Whi','bi','Wxf','Whf','bf','Wxo','Who','bo','Wxc','Whc','bc','Wy','by'};
for i = 1:length(params)
    eval([params{i},'m = zeros(size(',params{i},'));']);
    eval([params{i},'v = zeros(size(',params{i},'));']);
end

% For simplicity, we store Adam moments in structures.
adam.m.Wxi = zeros(size(Wxi)); adam.v.Wxi = zeros(size(Wxi));
adam.m.Whi = zeros(size(Whi)); adam.v.Whi = zeros(size(Whi));
adam.m.bi  = zeros(size(bi));  adam.v.bi  = zeros(size(bi));
adam.m.Wxf = zeros(size(Wxf)); adam.v.Wxf = zeros(size(Wxf));
adam.m.Whf = zeros(size(Whf)); adam.v.Whf = zeros(size(Whf));
adam.m.bf  = zeros(size(bf));  adam.v.bf  = zeros(size(bf));
adam.m.Wxo = zeros(size(Wxo)); adam.v.Wxo = zeros(size(Wxo));
adam.m.Who = zeros(size(Who)); adam.v.Who = zeros(size(Who));
adam.m.bo  = zeros(size(bo));  adam.v.bo  = zeros(size(bo));
adam.m.Wxc = zeros(size(Wxc)); adam.v.Wxc = zeros(size(Wxc));
adam.m.Whc = zeros(size(Whc)); adam.v.Whc = zeros(size(Whc));
adam.m.bc  = zeros(size(bc));  adam.v.bc  = zeros(size(bc));
adam.m.Wy  = zeros(size(Wy));  adam.v.Wy  = zeros(size(Wy));
adam.m.by  = zeros(size(by));  adam.v.by  = zeros(size(by));

beta1 = 0.9; beta2 = 0.999; epsilon = 1e-8;

train_losses = zeros(epochs,1);
val_losses = zeros(epochs,1);
num_batches = floor(num_train / batch_size);

for ep = 1:epochs
    idx = randperm(num_train);
    epoch_loss = 0;
    
    for b = 1:num_batches
        batch_idx = idx((b-1)*batch_size+1:b*batch_size);
        X_batch = reshape(X_train(batch_idx,:,:), [batch_size, T]);  % (batch_size x T)
        y_batch = y_train(batch_idx);  % (batch_size x 1)
        
        % Forward pass for the mini-batch.
        % Initialize cell arrays to store activations for each time step.
        i_gate = zeros(hidden_size, batch_size, T);
        f_gate = zeros(hidden_size, batch_size, T);
        o_gate = zeros(hidden_size, batch_size, T);
        g_gate = zeros(hidden_size, batch_size, T);
        c_state = zeros(hidden_size, batch_size, T+1);
        h_state = zeros(hidden_size, batch_size, T+1);
        c_state(:, :, 1) = 0;
                h_state(:, :, 1) = 0;
        
        for t = 1:T
            % x_t: (1 x batch_size)
            x_t = reshape(X_batch(:, t), [1, batch_size]);
            % Previous hidden state.
            h_prev = squeeze(h_state(:, :, t));  % (hidden_size x batch_size)
            
            % Input gate.
            i_t = sigmoid(Wxi * x_t + Whi * h_prev + repmat(bi, 1, batch_size));
            % Forget gate.
            f_t = sigmoid(Wxf * x_t + Whf * h_prev + repmat(bf, 1, batch_size));
            % Output gate.
            o_t = sigmoid(Wxo * x_t + Who * h_prev + repmat(bo, 1, batch_size));
            % Candidate cell state.
            g_t = tanh(Wxc * x_t + Whc * h_prev + repmat(bc, 1, batch_size));
            % New cell state.
            c_t = f_t .* squeeze(c_state(:, :, t)) + i_t .* g_t;
            % New hidden state.
            h_t = o_t .* tanh(c_t);
            
            % Store activations.
            i_gate(:, :, t) = i_t;
            f_gate(:, :, t) = f_t;
            o_gate(:, :, t) = o_t;
            g_gate(:, :, t) = g_t;
            c_state(:, :, t+1) = c_t;
            h_state(:, :, t+1) = h_t;
        end
        
        % Prediction from final hidden state.
        H_final = squeeze(h_state(:, :, end));  % (hidden_size x batch_size)
        y_pred = (Wy * H_final + repmat(by, 1, batch_size)).';  % (batch_size x 1)
        loss = sum(0.5*(y_pred - y_batch).^2) / batch_size;
        epoch_loss = epoch_loss + loss;
        
        %% Backward Pass: Full BPTT for LSTM.
        % Initialize gradients for all parameters.
        dWxi = zeros(size(Wxi)); dWhi = zeros(size(Whi)); dbi = zeros(size(bi));
        dWxf = zeros(size(Wxf)); dWhf = zeros(size(Whf)); dbf = zeros(size(bf));
        dWxo = zeros(size(Wxo)); dWho = zeros(size(Who)); dbo = zeros(size(bo));
        dWxc = zeros(size(Wxc)); dWhc = zeros(size(Whc)); dbc = zeros(size(bc));
        dWy  = zeros(size(Wy));  dby  = zeros(size(by));
        
        % Gradient w.r.t. output.
        dy = (y_pred - y_batch) / batch_size;  % (batch_size x 1)
        dWy = dWy + dy.' * H_final.';  % (output_size x hidden_size)
        dby = dby + sum(dy, 1).';
        
        % Backprop into final hidden state.
        dh_next = Wy.' * dy.';  % (hidden_size x batch_size)
        dc_next = zeros(size(dh_next));
        
        % BPTT over time steps.
        for t = T:-1:1
            % Retrieve activations at time t.
            i_t = squeeze(i_gate(:, :, t));
            f_t = squeeze(f_gate(:, :, t));
            o_t = squeeze(o_gate(:, :, t));
            g_t = squeeze(g_gate(:, :, t));
            c_t = squeeze(c_state(:, :, t+1));
            c_prev = squeeze(c_state(:, :, t));
            h_prev = squeeze(h_state(:, :, t));
            x_t = reshape(X_batch(:, t), [1, batch_size]);
            
            % Backprop through the output gate and tanh.
            dh = dh_next;
            do = dh .* tanh(c_t);
            do = do .* o_t .* (1 - o_t);
            
            % Backprop through cell state.
            dc = dh .* o_t .* (1 - tanh(c_t).^2) + dc_next;
            di = dc .* g_t;
            di = di .* i_t .* (1 - i_t);
            df = dc .* c_prev;
            df = df .* f_t .* (1 - f_t);
            dg = dc .* i_t;
            dg = dg .* (1 - g_t.^2);
            
            % Accumulate parameter gradients.
            dWxi = dWxi + di * x_t.';
            dWhi = dWhi + di * h_prev.';
            dbi = dbi + sum(di, 2);
            
            dWxf = dWxf + df * x_t.';
            dWhf = dWhf + df * h_prev.';
            dbf = dbf + sum(df, 2);
            
            dWxo = dWxo + do * x_t.';
            dWho = dWho + do * h_prev.';
            dbo = dbo + sum(do, 2);
            
            dWxc = dWxc + dg * x_t.';
            dWhc = dWhc + dg * h_prev.';
            dbc = dbc + sum(dg, 2);
            
            % Compute gradients to pass to previous time step.
            dh_prev = Whi.' * di + Whf.' * df + Who.' * do + Whc.' * dg;
            dc_prev = dc .* f_t;
            
            dh_next = dh_prev;
            dc_next = dc_prev;
        end
        
        %% Adam updates for all LSTM parameters.
        % For each parameter, update using its Adam moments stored in adam.
        % Helper inline function for Adam update:
        adam_update = @(param, dparam, m, v, t) ...
            param - learning_rate * ((m * beta1 + (1-beta1)*dparam) / (1 - beta1^t)) ./ (sqrt(v * beta2 + (1-beta2)*(dparam.^2)) / (1 - beta2^t) + epsilon);
        
        % For clarity, update each parameter manually:
        % Update Wxi.
        adam.m.Wxi = beta1*adam.m.Wxi + (1-beta1)*dWxi;
        adam.v.Wxi = beta2*adam.v.Wxi + (1-beta2)*(dWxi.^2);
        Wxi = Wxi - learning_rate * adam.m.Wxi./(sqrt(adam.v.Wxi)+epsilon);
        % Update Whi.
        adam.m.Whi = beta1*adam.m.Whi + (1-beta1)*dWhi;
        adam.v.Whi = beta2*adam.v.Whi + (1-beta2)*(dWhi.^2);
        Whi = Whi - learning_rate * adam.m.Whi./(sqrt(adam.v.Whi)+epsilon);
        % Update bi.
        adam.m.bi = beta1*adam.m.bi + (1-beta1)*dbi;
        adam.v.bi = beta2*adam.v.bi + (1-beta2)*(dbi.^2);
        bi = bi - learning_rate * adam.m.bi./(sqrt(adam.v.bi)+epsilon);
        
        % Update Wxf.
        adam.m.Wxf = beta1*adam.m.Wxf + (1-beta1)*dWxf;
        adam.v.Wxf = beta2*adam.v.Wxf + (1-beta2)*(dWxf.^2);
        Wxf = Wxf - learning_rate * adam.m.Wxf./(sqrt(adam.v.Wxf)+epsilon);
        % Update Whf.
        adam.m.Whf = beta1*adam.m.Whf + (1-beta1)*dWhf;
        adam.v.Whf = beta2*adam.v.Whf + (1-beta2)*(dWhf.^2);
        Whf = Whf - learning_rate * adam.m.Whf./(sqrt(adam.v.Whf)+epsilon);
        % Update bf.
        adam.m.bf = beta1*adam.m.bf + (1-beta1)*dbf;
        adam.v.bf = beta2*adam.v.bf + (1-beta2)*(dbf.^2);
        bf = bf - learning_rate * adam.m.bf./(sqrt(adam.v.bf)+epsilon);
        
        % Update Wxo.
        adam.m.Wxo = beta1*adam.m.Wxo + (1-beta1)*dWxo;
        adam.v.Wxo = beta2*adam.v.Wxo + (1-beta2)*(dWxo.^2);
        Wxo = Wxo - learning_rate * adam.m.Wxo./(sqrt(adam.v.Wxo)+epsilon);
        % Update Who.
        adam.m.Who = beta1*adam.m.Who + (1-beta1)*dWho;
        adam.v.Who = beta2*adam.v.Who + (1-beta2)*(dWho.^2);
        Who = Who - learning_rate * adam.m.Who./(sqrt(adam.v.Who)+epsilon);
        % Update bo.
        adam.m.bo = beta1*adam.m.bo + (1-beta1)*dbo;
        adam.v.bo = beta2*adam.v.bo + (1-beta2)*(dbo.^2);
        bo = bo - learning_rate * adam.m.bo./(sqrt(adam.v.bo)+epsilon);
        
        % Update Wxc.
        adam.m.Wxc = beta1*adam.m.Wxc + (1-beta1)*dWxc;
        adam.v.Wxc = beta2*adam.v.Wxc + (1-beta2)*(dWxc.^2);
        Wxc = Wxc - learning_rate * adam.m.Wxc./(sqrt(adam.v.Wxc)+epsilon);
        % Update Whc.
        adam.m.Whc = beta1*adam.m.Whc + (1-beta1)*dWhc;
        adam.v.Whc = beta2*adam.v.Whc + (1-beta2)*(dWhc.^2);
        Whc = Whc - learning_rate * adam.m.Whc./(sqrt(adam.v.Whc)+epsilon);
        % Update bc.
        adam.m.bc = beta1*adam.m.bc + (1-beta1)*dbc;
        adam.v.bc = beta2*adam.v.bc + (1-beta2)*(dbc.^2);
        bc = bc - learning_rate * adam.m.bc./(sqrt(adam.v.bc)+epsilon);
        
        % Update output layer parameters.
        adam.m.Wy = beta1*adam.m.Wy + (1-beta1)*dWy;
        adam.v.Wy = beta2*adam.v.Wy + (1-beta2)*(dWy.^2);
        Wy = Wy - learning_rate * adam.m.Wy./(sqrt(adam.v.Wy)+epsilon);
        
        adam.m.by = beta1*adam.m.by + (1-beta1)*dby;
        adam.v.by = beta2*adam.v.by + (1-beta2)*(dby.^2);
        by = by - learning_rate * adam.m.by./(sqrt(adam.v.by)+epsilon);
    end
    
    train_losses(ep) = epoch_loss / num_batches;
    
    % Validation pass.
    num_val = size(X_val,1);
    val_preds = zeros(num_val,1);
    for i = 1:num_val
        x_seq = reshape(X_val(i,:,:), [1, T]);
        h = zeros(hidden_size,1);
        c = zeros(hidden_size,1);
        for t = 1:T
            x_t = x_seq(:,t);
            i_t = sigmoid(Wxi * x_t + Whi * h + bi);
            f_t = sigmoid(Wxf * x_t + Whf * h + bf);
            o_t = sigmoid(Wxo * x_t + Who * h + bo);
            g_t = tanh(Wxc * x_t + Whc * h + bc);
            c = f_t .* c + i_t .* g_t;
            h = o_t .* tanh(c);
        end
        val_preds(i) = Wy * h + by;
    end
    val_loss = sum(0.5*(val_preds - y_val).^2)/num_val;
    val_losses(ep) = val_loss;
    
    fprintf('LSTM Epoch %d/%d: Train Loss = %.4f, Val Loss = %.4f\n', ep, epochs, train_losses(ep), val_losses(ep));
end

y_pred_val = val_preds;
model.Wxi = Wxi; model.Whi = Whi; model.bi = bi;
model.Wxf = Wxf; model.Whf = Whf; model.bf = bf;
model.Wxo = Wxo; model.Who = Who; model.bo = bo;
model.Wxc = Wxc; model.Whc = Whc; model.bc = bc;
model.Wy  = Wy;  model.by  = by;
end

% Helper function: sigmoid.
function y = sigmoid(x)
    y = 1./(1+exp(-x));
end

