close all; clear all; clc; 

% Generate synthetic data (position estimation)
N = 10; % Number of measurements
true_position = 5; % True state

% Observation matrix (linear model)
H = ones(N,1); % Assuming direct observation

% Generate noisy measurements
Y = true_position + randn(N,1) * 0.5; % Adding Gaussian noise

% Apply Least Squares Filter
X_est = least_squares_filter(H, Y);

% Display results
fprintf('True Position: %.2f\n', true_position);
fprintf('Estimated Position: %.2f\n', X_est);

% Plot results
figure; hold on;
scatter(1:N, Y, 'ro', 'DisplayName', 'Noisy Measurements');
plot(1:N, ones(N,1) * X_est, 'b', 'LineWidth', 2, 'DisplayName', 'LSF Estimate');
plot(1:N, ones(N,1) * true_position, 'k--', 'LineWidth', 2, 'DisplayName', 'True Position');
xlabel('Measurement Index');
ylabel('Position Estimate');
title('Least Squares Filter Estimation');
legend;
grid on;
hold off;

% save_all_figs_OPTION('results/leastSquares','png',1)


function X_est = least_squares_filter(H, Y)
    % Implements the Least Squares Filter
    % Inputs:
    %   H - Observation matrix (NxM)
    %   Y - Measurement vector (Nx1)
    % Output:
    %   X_est - Estimated state vector (Mx1)

    X_est = (H' * H) \ (H' * Y); % Normal Equation Solution
end

