%% DDPG Tutorial for Inverted Pendulum (From Scratch, Finite-Difference Gradients)
% This script implements the DDPG algorithm using only basic MATLAB operations.
% All functions (actor/critic forward passes, replay buffer, finite-difference
% gradient estimation, soft updates) are implemented from scratch.
%
% Note: Finite-difference gradient estimation is computationally expensive and
% provided here solely for educational purposes.
%
% for oscillations:
% decrease explorationNoise, tau, actorLR, and criticLR.


clc; clear; close all;
rng(1);

%% Create the Environment
env = InvertedPendulum();

%% Hyperparameters
numEpisodes      = 20;          % number of training episodes
maxSteps         = env.maxEpisodeSteps;
gamma            = 0.99;         % discount factor
actorLR          = 0.001;         % learning rate for actor >> was 0.01
criticLR         = 0.001;         % learning rate for critic >> was 0.01
tau              = 0.01;         % soft update factor for target networks >> was 0.1
batchSize        = 8;           % mini-batch size for updates >> was 32
replayCapacity   = 1000;         % maximum replay buffer capacity
explorationNoise = 0.01;          % scale of Gaussian exploration noise >> was 0.1
delta            = 1e-5;         % finite difference step size for gradient estimation

%% Network Architecture Parameters
obs_dim    = 3;    % observation dimension: [cos(theta); sin(theta); theta_dot]
action_dim = 1;    % action dimension (torque)
hidden_size = 16;  % size of hidden layers (keep small for finite differences)

%% Initialize Actor Network (deterministic policy)
% Architecture: Input (3) -> Hidden (hidden_size) -> Output (1)
actor.W1 = 0.1 * randn(hidden_size, obs_dim);
actor.b1 = zeros(hidden_size, 1);
actor.W2 = 0.1 * randn(action_dim, hidden_size);
actor.b2 = zeros(action_dim, 1);

%% Initialize Critic Network (state-action Q-value)
% Architecture: Input (obs_dim+action_dim=4) -> Hidden (hidden_size) -> Output (1)
critic.W1 = 0.1 * randn(hidden_size, obs_dim+action_dim);
critic.b1 = zeros(hidden_size, 1);
critic.W2 = 0.1 * randn(1, hidden_size);
critic.b2 = 0;

%% Initialize Target Networks (copies of actor and critic)
actor_target = actor;
critic_target = critic;

%% Initialize Replay Buffer as a structure
buffer.count = 0;
buffer.capacity = replayCapacity;
buffer.states = zeros(obs_dim, replayCapacity);
buffer.actions = zeros(action_dim, replayCapacity);
buffer.rewards = zeros(1, replayCapacity);
buffer.next_states = zeros(obs_dim, replayCapacity);
buffer.dones = zeros(1, replayCapacity);

%% Main Training Loop
episodeRewards = zeros(1, numEpisodes);
for ep = 1:numEpisodes
    s = env.reset();
    totalReward = 0;
    
    for t = 1:maxSteps
        % Actor forward pass and add exploration noise.
        a = actor_forward(actor, s);
        a = a + explorationNoise * randn(size(a));
        a = max(min(a, env.maxTorque), -env.maxTorque);
        
        % Step environment.
        [s_next, r, done] = env.step(a);
        totalReward = totalReward + r;
        
        % Store transition in replay buffer.
        buffer.count = buffer.count + 1;
        idx = mod(buffer.count-1, replayCapacity) + 1;
        buffer.states(:, idx) = s;
        buffer.actions(:, idx) = a;
        buffer.rewards(idx) = r;
        buffer.next_states(:, idx) = s_next;
        buffer.dones(idx) = done;
        
        s = s_next;
        if done
            break;
        end
        
        % Update networks if we have enough samples.
        if buffer.count >= batchSize
            % Sample a mini-batch.
            indices = randi(min(buffer.count, replayCapacity), 1, batchSize);
            batch.states = buffer.states(:, indices);
            batch.actions = buffer.actions(:, indices);
            batch.rewards = buffer.rewards(indices);
            batch.next_states = buffer.next_states(:, indices);
            batch.dones = buffer.dones(indices);
            
            % Compute target Q-values for mini-batch.
            target_Q = zeros(1, batchSize);
            for i = 1:batchSize
                s_next_i = batch.next_states(:, i);
                r_i = batch.rewards(i);
                done_i = batch.dones(i);
                a_next = actor_forward(actor_target, s_next_i);
                Q_next = critic_forward(critic_target, [s_next_i; a_next]);
                if done_i
                    target_Q(i) = r_i;
                else
                    target_Q(i) = r_i + gamma * Q_next;
                end
            end
            
            % Update Critic using mini-batch (finite differences).
            [critic, critic_loss] = update_critic(critic, batch.states, batch.actions, target_Q, criticLR, delta);
            
            % Update Actor using mini-batch.
            [actor, actor_loss] = update_actor(actor, batch.states, critic, actorLR, delta);
            
            % Soft update target networks.
            actor_target = soft_update(actor, actor_target, tau);
            critic_target = soft_update(critic, critic_target, tau);
        end
    end
    
    episodeRewards(ep) = totalReward;
    fprintf('Episode %d, Total Reward: %.2f\n', ep, totalReward);
end

%% Plot Convergence of Episode Rewards
figure;
plot(1:numEpisodes, episodeRewards, 'b-', 'LineWidth', 1);
xlabel('Episode'); ylabel('Total Reward');
title('DDPG Convergence on Inverted Pendulum');
grid on;
hold on;
plot(1:numEpisodes, movmean(episodeRewards, 10), 'r-', 'LineWidth', 2);
legend('Episode Reward','Moving Average');
hold off;

% save_all_figs_OPTION('results/ddpg_convergence','png',1)

%% Final Test Run and Trajectory Plot
s = env.reset();
trajectory = [];
done = false;
while ~done
    a = actor_forward(actor, s);
    a = max(min(a, env.maxTorque), -env.maxTorque);
    [s, r, done] = env.step(a);
    trajectory = [trajectory; env.state'];
end

dt = env.dt;
time = (0:size(trajectory,1)-1) * dt;
figure;
subplot(2,1,1);
plot(time, trajectory(:,1), 'b', 'LineWidth', 2);
xlabel('Time (s)'); ylabel('Pole Angle (rad)');
title('Pole Angle Trajectory');
grid on;
subplot(2,1,2);
plot(time, trajectory(:,2), 'b', 'LineWidth', 2);
xlabel('Time (s)'); ylabel('Angular Velocity (rad/s)');
title('Angular Velocity Trajectory');
grid on;

% save_all_figs_OPTION('results/ddpg_trajectory','png',1)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Local Function Definitions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function a = actor_forward(actor, s)
    % Forward pass through the actor network.
    % s: observation vector (3x1)
    z1 = actor.W1 * s + actor.b1;
    h1 = max(0, z1);   % ReLU activation
    a = actor.W2 * h1 + actor.b2;
end

function Q = critic_forward(critic, sa)
    % Forward pass through the critic network.
    % sa: concatenated [state; action] vector ((obs_dim+action_dim)x1)
    z1 = critic.W1 * sa + critic.b1;
    h1 = max(0, z1);
    Q = critic.W2 * h1 + critic.b2;
end

function g = finiteDiffGradient(param, lossFunc, delta)
    % Compute finite-difference gradient of lossFunc with respect to param.
    g = zeros(size(param));
    for i = 1:numel(param)
        orig = param(i);
        param(i) = orig + delta;
        loss_plus = lossFunc(param);
        param(i) = orig - delta;
        loss_minus = lossFunc(param);
        g(i) = (loss_plus - loss_minus) / (2*delta);
        param(i) = orig;
    end
end

function [actor, loss] = update_actor(actor, states, critic, actorLR, delta)
    % Update actor network using the deterministic policy gradient objective.
    % The objective is to maximize Q(s, actor(s)) over the mini-batch, so we
    % minimize -mean(Q(s, actor(s))).
    N = size(states, 2);
    loss = 0;
    for i = 1:N
        s = states(:, i);
        a = actor_forward(actor, s);
        Q_val = critic_forward(critic, [s; a]);
        loss = loss - Q_val;
    end
    loss = loss / N;
    
    % Update each actor parameter via finite differences.
    actor.W1 = actor.W1 - actorLR * finiteDiffGradient(actor.W1, @(W) update_actor_loss_wrapper(setfield(actor, 'W1', W), states, critic), delta);
    actor.b1 = actor.b1 - actorLR * finiteDiffGradient(actor.b1, @(b) update_actor_loss_wrapper(setfield(actor, 'b1', b), states, critic), delta);
    actor.W2 = actor.W2 - actorLR * finiteDiffGradient(actor.W2, @(W) update_actor_loss_wrapper(setfield(actor, 'W2', W), states, critic), delta);
    actor.b2 = actor.b2 - actorLR * finiteDiffGradient(actor.b2, @(b) update_actor_loss_wrapper(setfield(actor, 'b2', b), states, critic), delta);
end

function loss = update_actor_loss_wrapper(actor, states, critic)
    % Compute actor loss (negative mean Q-value) over the mini-batch.
    N = size(states,2);
    loss = 0;
    for i = 1:N
        s = states(:,i);
        a = actor_forward(actor, s);
        Q_val = critic_forward(critic, [s; a]);
        loss = loss - Q_val;
    end
    loss = loss / N;
end

function [critic, loss] = update_critic(critic, states, actions, target_Q, criticLR, delta)
    % Update critic network using MSE loss: loss = mean((Q(s,a) - target_Q)^2)
    N = size(states, 2);
    loss = 0;
    for i = 1:N
        s = states(:, i);
        a = actions(:, i);
        Q_val = critic_forward(critic, [s; a]);
        loss = loss + (Q_val - target_Q(i))^2;
    end
    loss = loss / N;
    
    % Update each critic parameter via finite differences.
    critic.W1 = critic.W1 - criticLR * finiteDiffGradient(critic.W1, @(W) update_critic_loss_wrapper(setfield(critic, 'W1', W), states, actions, target_Q), delta);
    critic.b1 = critic.b1 - criticLR * finiteDiffGradient(critic.b1, @(b) update_critic_loss_wrapper(setfield(critic, 'b1', b), states, actions, target_Q), delta);
    critic.W2 = critic.W2 - criticLR * finiteDiffGradient(critic.W2, @(W) update_critic_loss_wrapper(setfield(critic, 'W2', W), states, actions, target_Q), delta);
    critic.b2 = critic.b2 - criticLR * finiteDiffGradient(critic.b2, @(b) update_critic_loss_wrapper(setfield(critic, 'b2', b), states, actions, target_Q), delta);
end

function loss = update_critic_loss_wrapper(critic, states, actions, target_Q)
    % Compute critic loss (MSE) over the mini-batch.
    N = size(states,2);
    loss = 0;
    for i = 1:N
        s = states(:, i);
        a = actions(:, i);
        Q_val = critic_forward(critic, [s; a]);
        loss = loss + (Q_val - target_Q(i))^2;
    end
    loss = loss / N;
end

function target_net = soft_update(net, target_net, tau)
    % Soft update: target = tau*net + (1-tau)*target.
    fields = fieldnames(net);
    for i = 1:length(fields)
        fld = fields{i};
        target_net.(fld) = tau * net.(fld) + (1 - tau) * target_net.(fld);
    end
end
