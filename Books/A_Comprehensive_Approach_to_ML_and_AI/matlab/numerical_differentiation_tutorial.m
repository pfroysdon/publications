% Numerical Differentiation Tutorial in MATLAB
%
% This tutorial demonstrates how to compute the numerical gradient of a scalar
% function using central differences. Numerical differentiation is commonly used 
% in machine learning for gradient checking in back-propagation.
%
% In this example, we define a simple function:
%    f(x) = sum(x.^2)
% whose analytical gradient is 2*x.
%
% The function numericalGradient(f, x) computes the gradient at point x using
% central differences:
%
%    grad(i) = (f(x + h*e_i) - f(x - h*e_i)) / (2*h)
%
% We compare the numerical gradient to the analytical gradient.

clear; clc; close all; rng(1);

%% Define the Test Function and Its Analytical Gradient
% Here, we use a simple quadratic function f(x) = sum(x.^2)
f = @(x) sum(x.^2);
% Analytical gradient: grad_f(x) = 2*x.
analyticalGradient = @(x) 2 * x;

% Choose a test point (a vector in R^n)
x0 = randn(5,1);  % a 5-dimensional random vector

%% Compute Numerical Gradient at x0
h = 1e-5;  % step size for central differences
numGrad = numericalGradient(f, x0, h);

%% Display Results
fprintf('Test point x0:\n');
disp(x0);
fprintf('Analytical gradient:\n');
disp(analyticalGradient(x0));
fprintf('Numerical gradient:\n');
disp(numGrad);

% Compute error between numerical and analytical gradients.
errorNorm = norm(numGrad - analyticalGradient(x0));
fprintf('Difference (L2 norm) between gradients: %.6f\n', errorNorm);

%% Plot: Compare Analytical and Numerical Gradients (Bar Graph)
figure;
bar([analyticalGradient(x0), numGrad]);
legend('Analytical','Numerical');
xlabel('Dimension');
ylabel('Gradient Value');
title('Comparison of Analytical and Numerical Gradients');

% save_all_figs_OPTION('results/numericalDiff','png',1)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Local Function Definitions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function grad = numericalGradient(f, x, h)
    % numericalGradient computes the numerical gradient of f at x using central differences.
    %
    % Inputs:
    %   f - Function handle that accepts a column vector and returns a scalar.
    %   x - Column vector at which to compute the gradient.
    %   h - Step size for finite differences (default: 1e-5 if not provided).
    %
    % Output:
    %   grad - Numerical gradient (same size as x).
    
    if nargin < 3
        h = 1e-5;
    end
    
    grad = zeros(size(x));
    for i = 1:length(x)
        % Create unit vector in i-th direction
        e = zeros(size(x));
        e(i) = 1;
        % Central difference approximation:
        grad(i) = (f(x + h*e) - f(x - h*e)) / (2*h);
    end
end
