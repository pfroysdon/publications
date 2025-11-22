% File: simple_nn.m % A MATLAB implementation of a simple neural network 
% adapted from the provided Python code.

%% Main script

% Load dataset 
% (This example assumes you have a MAT file 'digits.mat' containing: 
% data - an [n_samples x 64] matrix of pixel values, 
% images - an [8 x 8 x n_samples] array of images, 
% target - an [n_samples x 1] vector with labels 0-9.) 
load('digits.mat');

% Display dataset dimensions 
disp(size(data));

% Display one digit image (show the second image, since Python used index 1) 
figure; 
colormap(gray); 
imagesc(images(:,:,2)); 
title('Sample Digit');
save_all_figs_OPTION('vnn1','png',1)

% Show raw pixel representation of the first sample 
disp('Dataset pixel representations:'); 
disp(data(1,:));

% Scale the data using zscore (similar to StandardScaler) 
X = zscore(data); 
disp('Scaled input data:'); 
disp(X(1,:));

% Set targets 
y = target;

% Split data into training (60%) and testing (40%) sets 
cv = cvpartition(size(X,1),'HoldOut',0.4); 
X_train = X(training(cv),:); X_test = X(test(cv),:); y_train = y(training(cv)); y_test = y(test(cv));

% Convert labels to one-hot encoding (remember to add 1 for MATLAB indices) 
y_v_train = convert_y_to_vect(y_train); 
y_v_test = convert_y_to_vect(y_test);

disp(['y_train(1)= ', num2str(y_train(1))]); 
disp('y_v_train(1)= '); 
disp(y_v_train(1,:));

% Define neural network structure: 
% 64 input nodes, one hidden layer with 20 nodes, and 10 output nodes. 
nn_structure = [64, 20, 10]; 
iter_num = 5000; 
alpha = 0.5;

% Train the neural network 
[W, b, avg_cost_func] = train_nn(nn_structure, X_train, y_v_train, iter_num, alpha);

% Plot the average cost versus iterations 
figure; 
plot(avg_cost_func); 
ylabel('Average Cost (J)'); 
xlabel('Iteration number'); grid on;

save_all_figs_OPTION('vnn2','png',1)

% Predict on the test set. % Note: use n_layers = length(nn_structure) 
y_pred = predict_y(W, b, X_test, length(nn_structure));

% Calculate accuracy 
accuracy = sum(y_test == y_pred) / length(y_test) * 100; 
disp(['Accuracy score = ', num2str(accuracy)]);

%% Function Definitions
function y = f(x) 
% Sigmoid activation function 
y = 1 ./ (1 + exp(-x)); 
end

function y = f_deriv(x) 
% Derivative of the sigmoid function 
fx = f(x); 
y = fx .* (1 - fx); 
end

function [h, z] = feed_forward(x, W, b) 
% Performs feed-forward propagation. 
% x - input column vector. 
% W - cell array of weight matrices. 
% b - cell array of bias vectors. 
% h - cell array of activations (h{1} is input, h{end} is output). 
% z - cell array of linear combinations.

L = length(W) + 1;  % total number of layers (input + layers with weights)
h = cell(L, 1);
z = cell(L, 1);
h{1} = x;
for l = 1:(L-1)
    node_in = h{l};
    z{l+1} = W{l} * node_in + b{l};
    h{l+1} = f(z{l+1});
end
end

function [W, b] = setup_and_init_weights(nn_structure) 
% Initializes weights and biases randomly. 
% nn_structure is a vector [n_input, n_hidden, n_output]. 
L = length(nn_structure); 
W = cell(L-1, 1); 
b = cell(L-1, 1); 
for l = 1:(L-1) 
	% For layer l, weights are of size (nn_structure(l+1) x nn_structure(l)) 
	W{l} = rand(nn_structure(l+1), nn_structure(l)); 
	b{l} = rand(nn_structure(l+1), 1); 
end 
end

function [tri_W, tri_b] = init_tri_values(nn_structure) 
% Initializes the gradient accumulators (tri_W and tri_b) as zeros. 
L = length(nn_structure); 
tri_W = cell(L-1, 1); 
tri_b = cell(L-1, 1); 
for l = 1:(L-1) 
	tri_W{l} = zeros(nn_structure(l+1), nn_structure(l)); 
	tri_b{l} = zeros(nn_structure(l+1), 1); 
end 
end

function delta = calculate_out_layer_delta(y_true, h_out, z_out) 
% Calculates the output layer delta. 
% y_true and h_out are column vectors. 
delta = -(y_true - h_out) .* f_deriv(z_out); 
end

function delta = calculate_hidden_delta(delta_next, w_l, z_l) 
% Calculates the hidden layer delta. 
delta = (w_l' * delta_next) .* f_deriv(z_l); 
end

function y_vect = convert_y_to_vect(y) 
% Converts label vector y into a one-hot encoded matrix. 
% Assumes y contains digits 0-9. 
n = length(y); 
y_vect = zeros(n, 10); 
for i = 1:n 
% MATLAB indices start at 1, so add 1. 
y_vect(i, y(i) + 1) = 1; 
end 
end

function [W, b, avg_cost_func] = train_nn(nn_structure, X, y, iter_num, alpha) 
% Trains the neural network using gradient descent. 
% nn_structure: network structure vector. 
% X: training data (each row is an example). 
% y: one-hot encoded target matrix. 
% iter_num: number of iterations. 
% alpha: learning rate.

[W, b] = setup_and_init_weights(nn_structure);
m = size(X, 1);
avg_cost_func = zeros(iter_num, 1);
fprintf('Starting gradient descent for %d iterations\n', iter_num);

L = length(nn_structure);  % total number of layers (input + weight layers)

for cnt = 0:(iter_num-1)
    if mod(cnt, 1000) == 0
        fprintf('Iteration %d of %d\n', cnt, iter_num);
    end
    
    [tri_W, tri_b] = init_tri_values(nn_structure);
    avg_cost = 0;
    
    % Loop over all training examples
    for i = 1:m
        x_i = X(i, :)';      % convert row to column vector
        y_i = y(i, :)';      % one-hot target as column vector
        [h, z] = feed_forward(x_i, W, b);
        delta = cell(L, 1);
        
        % Backpropagation (looping from output layer down to input layer)
        for l = L:-1:1
            if l == L
                % Output layer delta
                delta{l} = calculate_out_layer_delta(y_i, h{l}, z{l});
                avg_cost = avg_cost + norm(y_i - h{l});
            else
                if l > 1
                    delta{l} = calculate_hidden_delta(delta{l+1}, W{l}, z{l});
                end
                % Accumulate gradients (outer product of delta and activation)
                tri_W{l} = tri_W{l} + delta{l+1} * h{l}';
                tri_b{l} = tri_b{l} + delta{l+1};
            end
        end
    end
    
    % Update weights and biases using the averaged gradients
    for l = 1:(L-1)
        W{l} = W{l} - alpha * (1/m) * tri_W{l};
        b{l} = b{l} - alpha * (1/m) * tri_b{l};
    end
    
    avg_cost_func(cnt + 1) = avg_cost / m;
end
end

function y_pred = predict_y(W, b, X, n_layers) 
% Predicts labels for the data in X. 
% n_layers should equal length(nn_structure) (input+output layers). 
m = size(X, 1); 
y_pred = zeros(m, 1);

for i = 1:m
    x_i = X(i, :)';
    [h, ~] = feed_forward(x_i, W, b);
    [~, idx] = max(h{n_layers});
    % Subtract 1 to convert from MATLAB’s 1-indexing to the original 0–9 labels.
    y_pred(i) = idx - 1;
end

end
