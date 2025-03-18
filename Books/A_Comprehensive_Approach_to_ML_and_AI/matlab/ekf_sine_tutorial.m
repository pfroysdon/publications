%% ekf2DTracking_sinusoid_fixed.m
% Extended Kalman Filter (EKF) for 2D Tracking with a Sinusoidal Trajectory
%
% In this tutorial, we simulate a target moving in 2D with a sinusoidal trajectory.
% The horizontal position increases linearly and the vertical position follows a sine
% wave:
%
%     x(t) = v_x * t,   y(t) = A*sin(omega*t)
%     vx = v_x,         vy = A*omega*cos(omega*t)
%
% Noisy measurements (range and bearing from the origin) are generated.
% The EKF uses a constant velocity model as the process model.
%
% Changes to improve filter stability:
%   - The process noise covariance Q is increased to better account for the model mismatch.
%   - The initial covariance P is set to a larger value.

clear; clc; close all; rng(1);

%% Simulation Parameters
dt = 0.1;             % Time step (s)
T = 20;               % Total simulation time (s)
t = 0:dt:T;         
N = length(t);        % Number of time steps

%% True State Simulation (Sinusoidal Trajectory)
% Define true motion parameters:
v_x = 1.0;            % Constant horizontal velocity (m/s)
A = 2.0;              % Amplitude for vertical sinusoid (m)
omega = 0.5;          % Angular frequency (rad/s)

x_true = zeros(4, N); % State vector: [x; y; vx; vy]
for k = 1:N
    x_true(1, k) = v_x * t(k);              % x position
    x_true(2, k) = A * sin(omega * t(k));     % y position (sinusoidal)
    x_true(3, k) = v_x;                       % x velocity
    x_true(4, k) = A * omega * cos(omega * t(k)); % y velocity
end

%% Measurement Simulation
% Measurement model: z = [range; bearing]
% Measurement noise covariance R_meas:
R_meas = [0.1^2, 0; 0, (pi/180*5)^2];  % 0.1 m std for range; 5° for bearing
z = zeros(2, N);
for k = 1:N
    pos = x_true(1:2, k);
    range = norm(pos);
    bearing = atan2(pos(2), pos(1));
    noise = mvnrnd([0;0], R_meas)';
    z(:, k) = [range; bearing] + noise;
end

%% EKF Initialization
% Process model: constant velocity model.
F = [1, 0, dt, 0; 
     0, 1, 0, dt; 
     0, 0, 1, 0; 
     0, 0, 0, 1];
% Increase initial state covariance to reflect uncertainty:
P = 10 * eye(4);
% Increase process noise covariance to account for model mismatch:
Q = diag([0.005, 0.005, 0.05, 0.05]);

%% Extended Kalman Filter Implementation
x_est = zeros(4, N);
% Initialize estimate slightly off from true state:
x_est(:,1) = [0.2; -0.2; 0.8; 0.6];

for k = 1:N-1
    % Prediction Step:
    x_pred = F * x_est(:, k);
    P_pred = F * P * F' + Q;
    
    % Measurement Prediction:
    x_pos = x_pred(1);
    y_pos = x_pred(2);
    range_pred = sqrt(x_pos^2 + y_pos^2);
    bearing_pred = atan2(y_pos, x_pos);
    z_pred = [range_pred; bearing_pred];
    
    % Compute Jacobian of h(x) at x_pred:
    if range_pred < 1e-4
        H_jacobian = zeros(2,4);
    else
        d_range_dx = x_pos / range_pred;
        d_range_dy = y_pos / range_pred;
        d_bearing_dx = -y_pos / (range_pred^2);
        d_bearing_dy = x_pos / (range_pred^2);
        H_jacobian = [d_range_dx, d_range_dy, 0, 0;
                      d_bearing_dx, d_bearing_dy, 0, 0];
    end
    
    % Innovation:
    y_innov = z(:, k+1) - z_pred;
    % Normalize bearing difference to be within [-pi, pi]:
    y_innov(2) = mod(y_innov(2) + pi, 2*pi) - pi;
    
    % Innovation covariance:
    S = H_jacobian * P_pred * H_jacobian' + R_meas;
    
    % Kalman Gain:
    K = P_pred * H_jacobian' / S;
    
    % Update:
    x_est(:, k+1) = x_pred + K * y_innov;
    P = (eye(4) - K * H_jacobian) * P_pred;
end

%% Plotting Results
time = t;

figure;
subplot(2,1,1);
plot(time, x_true(1,:), 'b-', 'LineWidth',2); hold on;
plot(time, x_est(1,:), 'r--', 'LineWidth',2);
xlabel('Time (s)'); ylabel('x position (m)');
legend('True','Estimated');
title('Cart Position Tracking');
grid on;

subplot(2,1,2);
plot(time, x_true(2,:), 'b-', 'LineWidth',2); hold on;
plot(time, x_est(2,:), 'r--', 'LineWidth',2);
xlabel('Time (s)'); ylabel('y position (m)');
legend('True','Estimated');
title('Pole Position Tracking');
grid on;

figure;
subplot(2,1,1);
plot(time, x_true(3,:), 'b-', 'LineWidth',2); hold on;
plot(time, x_est(3,:), 'r--', 'LineWidth',2);
xlabel('Time (s)'); ylabel('x velocity (m/s)');
legend('True','Estimated');
title('Cart Velocity Tracking');
grid on;

subplot(2,1,2);
plot(time, x_true(4,:), 'b-', 'LineWidth',2); hold on;
plot(time, x_est(4,:), 'r--', 'LineWidth',2);
xlabel('Time (s)'); ylabel('y velocity (m/s)');
legend('True','Estimated');
title('Pole Velocity Tracking');
grid on;

figure;
plot(x_true(1,:), x_true(2,:), 'b-', 'LineWidth',2); hold on;
plot(x_est(1,:), x_est(2,:), 'r--', 'LineWidth',2);
xlabel('x position (m)'); ylabel('y position (m)');
legend('True trajectory','Estimated trajectory');
title('2D Trajectory Tracking');
grid on;

