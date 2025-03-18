% This tutorial demonstrates a simple sliding window filter (moving average)
% applied to daily weather temperature data.
%
% The script:
%   1. Generates synthetic temperature data for one year (365 days) using a 
%      sine wave to mimic seasonal variation plus random noise.
%   2. Implements a sliding window filter from scratch that computes a moving average 
%      over a specified window length.
%   3. Plots both the original and smoothed temperature data.

clear; clc; close all; rng(1);

%% 1. Generate Synthetic Daily Temperature Data
days = 1:365;
% Simulate a seasonal temperature pattern with a sine wave (period = 365 days)
seasonalTemp = 15 + 10*sin(2*pi*(days-80)/365); % average temp ~15°C with amplitude 10°C
% Add random noise
noise = randn(size(days)) * 2;  % standard deviation of 2°C
temperature = seasonalTemp + noise;

%% 2. Define Sliding Window Filter (Moving Average)
% The function slidingWindowFilter computes the moving average with a specified window size.
% For boundary points, we use a simple approach that computes the average over the available points.
windowSize = 7;  % e.g., 7-day moving average

smoothedTemperature7 = slidingWindowFilter(temperature, windowSize);
smoothedTemperature30 = slidingWindowFilter(temperature, 30);

%% 3. Plot the Original and Smoothed Temperature Data
figure;
plot(days, temperature, 'b-', 'LineWidth', 1.5); hold on;
plot(days, smoothedTemperature7, 'r-', 'LineWidth', 2);
xlabel('Day');
ylabel('Temperature (°C)');
title('Daily Temperature and 7-Day Moving Average');
legend('Original Temperature', 'Smoothed Temperature');
grid on;

figure;
plot(days, temperature, 'b-', 'LineWidth', 1.5); hold on;
plot(days, smoothedTemperature7, 'r-', 'LineWidth', 2);
plot(days, smoothedTemperature30, 'k-', 'LineWidth', 2);
xlabel('Day');
ylabel('Temperature (°C)');
title('Daily Temperature and 7-Day Moving Average');
legend('Original Temperature', '7-day Smoothed Temperature', '30-day Smoothed Temperature');
grid on;
xlim([130 230])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Local Function Definitions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function y_filtered = slidingWindowFilter(y, windowSize)
% slidingWindowFilter computes a moving average of the input vector y using a sliding window.
% Inputs:
%   y          - 1D vector of data.
%   windowSize - Size of the sliding window (must be an odd number for symmetric window).
%              If even, the function will use the next odd number.
% Output:
%   y_filtered - Filtered (smoothed) output vector.
%
% The function handles boundary conditions by computing the average over the
% available points.
    if mod(windowSize,2) == 0
        windowSize = windowSize + 1; % ensure windowSize is odd for symmetry
    end
    halfWindow = floor(windowSize/2);
    N = length(y);
    y_filtered = zeros(size(y));
    for i = 1:N
        % Determine window boundaries, ensuring indices are valid.
        idx_start = max(1, i - halfWindow);
        idx_end = min(N, i + halfWindow);
        windowData = y(idx_start:idx_end);
        y_filtered(i) = mean(windowData);
    end
end
