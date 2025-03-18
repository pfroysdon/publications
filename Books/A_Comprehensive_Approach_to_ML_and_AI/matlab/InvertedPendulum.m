classdef InvertedPendulum < handle
    % InvertedPendulum A self-contained inverted pendulum simulation.
    %
    %   The state is defined as:
    %       state = [theta; theta_dot]
    %   where theta is the angle (in radians) measured from the upright (theta=0)
    %   and theta_dot is the angular velocity.
    %
    %   The observation returned is:
    %       observation = [cos(theta); sin(theta); theta_dot]
    %
    %   Dynamics (Euler integration):
    %       theta_ddot = - (g/l)*sin(theta) + u/(m*l^2)
    %
    %   Reward:
    %       reward = 10 - (theta^2 + 0.1*theta_dot^2 + 0.001*u^2)
    %
    %   Termination:
    %       The episode ends if |theta| > pi/2 (pendulum has fallen) or if
    %       the maximum number of steps is reached.
    
    properties
        % Physical parameters
        m = 1;              % mass (kg)
        l = 1;              % length (m)
        g = 9.81;           % gravitational acceleration (m/s^2)
        dt = 0.02;          % time step (s)
        maxTorque = 2;      % maximum control torque (N*m)
        maxEpisodeSteps = 200;  % maximum steps per episode
        
        % Internal state: [theta; theta_dot]
        state = [0; 0];
        
        % Step counter
        stepCount = 0;
    end
    
    methods
        function obj = InvertedPendulum()
            % Constructor: Initialize the pendulum.
            obj.reset();
        end
        
        function obs = reset(obj)
            % reset Reinitialize the pendulum state.
            %
            % Returns:
            %   obs: the initial observation [cos(theta); sin(theta); theta_dot]
            
            % Set theta to a small deviation from upright and zero angular velocity.
            theta = 0.1 * (rand - 0.5) * 2;  % Uniformly in [-0.1, 0.1] radians
            theta_dot = 0;
            obj.state = [theta; theta_dot];
            obj.stepCount = 0;
            obs = [cos(theta); sin(theta); theta_dot];
        end
        
        function [obs, reward, done] = step(obj, action)
            % step Apply a control action and update the pendulum state.
            %
            % Inputs:
            %   action: control torque (scalar)
            %
            % Returns:
            %   obs: next observation [cos(theta); sin(theta); theta_dot]
            %   reward: reward for this step
            %   done: boolean flag indicating if the episode is terminated
            
            % Clip the action to the allowable range.
            u = max(min(action, obj.maxTorque), -obj.maxTorque);
            
            % Extract current state.
            theta = obj.state(1);
            theta_dot = obj.state(2);
            
            % Compute angular acceleration.
            theta_ddot = - (obj.g / obj.l) * sin(theta) + u / (obj.m * obj.l^2);
            
            % Update state using Euler integration.
            theta = theta + theta_dot * obj.dt;
            theta_dot = theta_dot + theta_ddot * obj.dt;
            
            % Wrap theta to [-pi, pi] using the complex exponential trick.
            theta = angle(exp(1i * theta));
            
            % Update internal state and counter.
            obj.state = [theta; theta_dot];
            obj.stepCount = obj.stepCount + 1;
            
            % Construct observation.
            obs = [cos(theta); sin(theta); theta_dot];
            
            % Compute reward (offset so that near-upright gives high reward).
            % reward = 10 - (theta^2 + 0.1 * theta_dot^2 + 0.001 * u^2);
            damping_bonus = 5 * (abs(theta_dot) < 0.1);
            reward = 10 + damping_bonus - (theta^2 + 0.5*theta_dot^2 + 0.001*u^2);
            
            % Determine if episode is done (pendulum falls or max steps reached).
            done = (abs(theta) > (pi / 2)) || (obj.stepCount >= obj.maxEpisodeSteps);
        end
        
        function render(obj)
            % render Visualize the current state of the pendulum.
            %
            % The pendulum is drawn as a line from the pivot (origin) to the mass.
            
            theta = obj.state(1);
            x = obj.l * sin(theta);
            y = obj.l * cos(theta);
            
            figure(1);
            clf;
            plot([0, x], [0, y], 'b-', 'LineWidth', 2);
            hold on;
            plot(x, y, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
            axis equal;
            xlim([-obj.l * 1.2, obj.l * 1.2]);
            ylim([-obj.l * 1.2, obj.l * 1.2]);
            title(sprintf('Step %d', obj.stepCount));
            xlabel('X'); ylabel('Y');
            drawnow;
        end
    end
end
