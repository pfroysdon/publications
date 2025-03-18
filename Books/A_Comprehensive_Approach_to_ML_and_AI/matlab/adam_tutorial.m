%% adamTutorial.m
% Adam Optimization Tutorial from Scratch in MATLAB
%
% This script demonstrates the Adam optimization algorithm by minimizing
% the quadratic function:
%
%   f(x) = (x1 - 2)^2 + (x2 + 3)^2
%
% The gradient of f(x) is:
%   grad_f(x) = [2*(x1 - 2); 2*(x2 + 3)]
%
% Adam update rules are used to update the parameter vector x.


close all; clear; clc;

%% Problem Definition
% Define the objective function and its gradient.
f = @(x) (x(1) - 2)^2 + (x(2) + 3)^2;
grad_f = @(x) [2*(x(1) - 2); 2*(x(2) + 3)];

% Global optimum: x = [2; -3] with f(x) = 0

%% Adam Parameters
alpha = 0.1;         % Learning rate
beta1 = 0.9;         % Exponential decay rate for the first moment estimates
beta2 = 0.999;       % Exponential decay rate for the second moment estimates
epsilon = 1e-8;      % Small constant to avoid division by zero
numIterations = 1000;

%% Initialization
x0 = [-5; 5];        % Initial guess
x = x0;
m = zeros(size(x));  % Initialize first moment vector
v = zeros(size(x));  % Initialize second moment vector
lossHistory = zeros(numIterations, 1);

%% Adam Optimization Loop
for t = 1:numIterations
    % Compute gradient at current x
    g = grad_f(x);
    
    % Update biased first moment estimate
    m = beta1 * m + (1 - beta1) * g;
    
    % Update biased second raw moment estimate
    v = beta2 * v + (1 - beta2) * (g.^2);
    
    % Compute bias-corrected first and second moment estimates
    m_hat = m / (1 - beta1^t);
    v_hat = v / (1 - beta2^t);
    
    % Update parameters
    x = x - alpha * m_hat ./ (sqrt(v_hat) + epsilon);
    
    % Store current loss
    lossHistory(t) = f(x);
    
    % Optionally, display progress every 100 iterations
    if mod(t, 100) == 0
        fprintf('Iteration %d: Loss = %.4f, x = [%.4f, %.4f]\n', t, lossHistory(t), x(1), x(2));
    end
end

fprintf('Optimized solution: x = [%.4f, %.4f]\n', x(1), x(2));
fprintf('Final objective value: %.4f\n', f(x));

%% Plot the Loss History
figure;
plot(1:numIterations, lossHistory, 'LineWidth', 2);
xlabel('Iteration');
ylabel('Objective Function Value');
title('ADAM Optimization Convergence');
grid on;

% save_all_figs_OPTION('results/adam','png',1)
