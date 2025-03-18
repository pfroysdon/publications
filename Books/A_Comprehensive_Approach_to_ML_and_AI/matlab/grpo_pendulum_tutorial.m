%% GRPO Tutorial for Inverted Pendulum (From Scratch)
% This script implements a basic version of Group Relative Policy Optimization (GRPO)
% using our custom inverted pendulum environment (InvertedPendulum.m). All functions 
% (policy forward pass, GRPO loss, finite-difference gradients, critic update, etc.)
% are implemented from scratch.
%
% References for GRPO:
%   - https://arxiv.org/pdf/2402.03300
%   - https://epichka.com/blog/2025/grpo/
%   - https://community.aws/content/2rJrpj6m2eh591fjMcRZ3ushpB7/deep-dive-into-group-relative-policy-optimization-grpo
%
% Note: This is a simplified implementation for demonstration and educational purposes.
% 

clc; clear; close all;
rng(1);

%% Create the Environment
env = InvertedPendulum();

%% Hyperparameters
numEpisodes      = 150;      % Number of training episodes
maxSteps         = env.maxEpisodeSteps;
gamma            = 0.99;     % Discount factor
actorLR          = 1e-3;     % Learning rate for actor updates
criticLR         = 1e-3;     % Learning rate for critic updates
explorationNoise = 0.1;      % Exploration noise scale
delta            = 1e-5;     % Finite difference step size

%% Network Architecture Parameters
obs_dim    = 3;    % Observation dimension: [cos(theta); sin(theta); theta_dot]
action_dim = 1;    % Action dimension (torque)
hidden_size = 16;  % Size of hidden layer (keep small for finite-difference estimation)

%% Initialize Actor Network (Deterministic Policy)
% Architecture: Input (3) -> Hidden (hidden_size, with ReLU) -> Output (1)
actor.W1 = 0.1 * randn(hidden_size, obs_dim);
actor.b1 = zeros(hidden_size, 1);
actor.W2 = 0.1 * randn(action_dim, hidden_size);
actor.b2 = zeros(action_dim, 1);
% We also learn a log standard deviation for a Gaussian policy
actor.log_std = -0.5 * ones(action_dim, 1);

%% Initialize Critic Network (State Value Estimator)
% Architecture: Input (obs_dim) -> Hidden (hidden_size) -> Output (1)
critic.W1 = 0.1 * randn(hidden_size, obs_dim);
critic.b1 = zeros(hidden_size, 1);
critic.W2 = 0.1 * randn(1, hidden_size);
critic.b2 = 0;

%% Training Loop Storage
episodeRewards = zeros(1, numEpisodes);

for ep = 1:numEpisodes
    % Reset environment and initialize trajectory storage
    s = env.reset();
    traj_obs = [];     % will store observations (3 x T)
    traj_actions = []; % will store actions (1 x T)
    traj_rewards = []; % row vector of rewards
    traj_logp = [];    % row vector of log probabilities
    
    done = false;
    while ~done
        % --- Actor Forward Pass with Exploration ---
        % Get action mean and standard deviation.
        [mean_a, std_a] = actor_forward(actor, s);
        % Sample action from Gaussian policy and add exploration noise.
        a = mean_a + std_a .* randn(size(std_a));
        % Clip action within allowed torque.
        a = max(min(a, env.maxTorque), -env.maxTorque);
        
        % Compute log probability of taken action.
        logp = gaussian_log_prob(a, mean_a, exp(actor.log_std));
        
        % Step the environment.
        [s_next, r, done] = env.step(a);
        
        % Store trajectory data.
        traj_obs = [traj_obs, s];
        traj_actions = [traj_actions, a];
        traj_rewards = [traj_rewards, r];
        traj_logp = [traj_logp, logp];
        
        s = s_next;
    end
    
    % --- Compute Returns ---
    T = length(traj_rewards);
    returns = zeros(1, T);
    G = 0;
    for t = T:-1:1
        G = traj_rewards(t) + gamma * G;
        returns(t) = G;
    end
    
    episodeRewards(ep) = sum(traj_rewards);
    
    % --- GRPO Update ---
    % Here we use the entire trajectory as our mini-batch.
    % Compute the median return.
    med_return = median(returns);
    % Identify indices for "good" and "bad" groups.
    good_idx = find(returns >= med_return);
    bad_idx = find(returns < med_return);
    
    if ~isempty(good_idx)
        mu_good = mean(returns(good_idx));
    else
        mu_good = med_return;
    end
    if ~isempty(bad_idx)
        mu_bad = mean(returns(bad_idx));
    else
        mu_bad = med_return;
    end
    
    % Compute GRPO loss over the trajectory.
    loss_grpo = grpo_loss(actor, traj_obs, traj_actions, returns, traj_logp, mu_good, mu_bad);
    
    % Update actor parameters using finite-difference gradient estimation.
    actor = update_actor_grpo(actor, traj_obs, traj_actions, returns, traj_logp, mu_good, mu_bad, actorLR, delta);
    
    % --- Critic Update (MSE Loss) ---
    critic = update_critic(critic, traj_obs, returns, criticLR, delta);
    
    fprintf('Episode %d, Total Reward: %.2f, GRPO Loss: %.4f\n', ep, episodeRewards(ep), loss_grpo);
end

%% Plot Convergence of Episode Rewards
figure;
plot(1:numEpisodes, episodeRewards, 'b-', 'LineWidth', 1);
xlabel('Episode'); ylabel('Total Reward');
title('GRPO Convergence on Inverted Pendulum');
grid on;
hold on;
plot(1:numEpisodes, movmean(episodeRewards, 10), 'r-', 'LineWidth', 2);
legend('Episode Reward','Moving Average');
hold off;

% save_all_figs_OPTION('results/grpo_convergence','png',1)

%% Final Test Run and Trajectory Plot
s = env.reset();
trajectory = [];
done = false;
while ~done
    % Use the actor (deterministic) to select an action.
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

% save_all_figs_OPTION('results/grpo_trajectory','png',1)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Local Function Definitions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [mean_a, std_a] = actor_forward(actor, s)
    % Compute the actor network forward pass.
    % s: observation vector (3x1)
    z1 = actor.W1 * s + actor.b1;
    h1 = max(0, z1);  % ReLU activation
    mean_a = actor.W2 * h1 + actor.b2;
    std_a = exp(actor.log_std);
end

function logp = gaussian_log_prob(a, mean, std)
    % Compute log probability of action 'a' under a Gaussian with given mean and std.
    logp = -0.5 * log(2*pi) - log(std) - 0.5 * ((a - mean)./std).^2;
    logp = sum(logp, 1);  % sum over action dimensions (scalar action)
end

function loss = grpo_loss(actor, obs_batch, actions_batch, returns, logprobs_batch, mu_good, mu_bad)
    % Compute the GRPO loss over a batch of samples.
    % obs_batch: [3 x T], actions_batch: [1 x T], returns: [1 x T], logprobs_batch: [1 x T]
    T = size(obs_batch,2);
    loss_sum = 0;
    for i = 1:T
        s = obs_batch(:, i);
        a = actions_batch(:, i);
        % Compute predicted log probability using the current actor.
        mean_a = actor_forward(actor, s);
        pred_logp = gaussian_log_prob(a, mean_a, exp(actor.log_std));
        if returns(i) >= mu_good
            advantage = returns(i) - mu_good;
            loss_sum = loss_sum - pred_logp * advantage;
        else
            advantage = mu_bad - returns(i);
            loss_sum = loss_sum + pred_logp * advantage;
        end
    end
    loss = loss_sum / T;
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
        g(i) = (loss_plus - loss_minus) / (2 * delta);
        param(i) = orig;
    end
end

function actor = update_actor_grpo(actor, obs_batch, actions_batch, returns, logprobs_batch, mu_good, mu_bad, actorLR, delta)
    % Update actor network parameters using finite-difference gradients
    % computed from the GRPO loss.
    
    % Update actor.W1
    lossFunc = @(W) grpo_loss(setfield(actor, 'W1', W), obs_batch, actions_batch, returns, logprobs_batch, mu_good, mu_bad);
    grad_W1 = finiteDiffGradient(actor.W1, lossFunc, delta);
    actor.W1 = actor.W1 - actorLR * grad_W1;
    
    % Update actor.b1
    lossFunc = @(b) grpo_loss(setfield(actor, 'b1', b), obs_batch, actions_batch, returns, logprobs_batch, mu_good, mu_bad);
    grad_b1 = finiteDiffGradient(actor.b1, lossFunc, delta);
    actor.b1 = actor.b1 - actorLR * grad_b1;
    
    % Update actor.W2
    lossFunc = @(W) grpo_loss(setfield(actor, 'W2', W), obs_batch, actions_batch, returns, logprobs_batch, mu_good, mu_bad);
    grad_W2 = finiteDiffGradient(actor.W2, lossFunc, delta);
    actor.W2 = actor.W2 - actorLR * grad_W2;
    
    % Update actor.b2
    lossFunc = @(b) grpo_loss(setfield(actor, 'b2', b), obs_batch, actions_batch, returns, logprobs_batch, mu_good, mu_bad);
    grad_b2 = finiteDiffGradient(actor.b2, lossFunc, delta);
    actor.b2 = actor.b2 - actorLR * grad_b2;
    
    % Update actor.log_std
    lossFunc = @(l) grpo_loss(setfield(actor, 'log_std', l), obs_batch, actions_batch, returns, logprobs_batch, mu_good, mu_bad);
    grad_logstd = finiteDiffGradient(actor.log_std, lossFunc, delta);
    actor.log_std = actor.log_std - actorLR * grad_logstd;
end

function Q = critic_forward(critic, s)
    % Forward pass through critic network.
    % s: observation vector (3x1)
    z1 = critic.W1 * s + critic.b1;
    h1 = max(0, z1);
    Q = critic.W2 * h1 + critic.b2;
end

function loss = critic_loss(critic, obs_batch, returns)
    % Compute mean-squared error loss for critic over a batch.
    T = size(obs_batch,2);
    loss_sum = 0;
    for i = 1:T
        s = obs_batch(:, i);
        Q_pred = critic_forward(critic, s);
        loss_sum = loss_sum + (Q_pred - returns(i))^2;
    end
    loss = loss_sum / T;
end

function critic = update_critic(critic, obs_batch, returns, criticLR, delta)
    % Update critic network parameters using finite-difference gradients.
    % Update critic.W1
    lossFunc = @(W) critic_loss(setfield(critic, 'W1', W), obs_batch, returns);
    grad_W1 = finiteDiffGradient(critic.W1, lossFunc, delta);
    critic.W1 = critic.W1 - criticLR * grad_W1;
    
    % Update critic.b1
    lossFunc = @(b) critic_loss(setfield(critic, 'b1', b), obs_batch, returns);
    grad_b1 = finiteDiffGradient(critic.b1, lossFunc, delta);
    critic.b1 = critic.b1 - criticLR * grad_b1;
    
    % Update critic.W2
    lossFunc = @(W) critic_loss(setfield(critic, 'W2', W), obs_batch, returns);
    grad_W2 = finiteDiffGradient(critic.W2, lossFunc, delta);
    critic.W2 = critic.W2 - criticLR * grad_W2;
    
    % Update critic.b2
    lossFunc = @(b) critic_loss(setfield(critic, 'b2', b), obs_batch, returns);
    grad_b2 = finiteDiffGradient(critic.b2, lossFunc, delta);
    critic.b2 = critic.b2 - criticLR * grad_b2;
end

function target_net = soft_update(net, target_net, tau)
    % Soft-update target network parameters.
    fields = fieldnames(net);
    for i = 1:length(fields)
        fld = fields{i};
        target_net.(fld) = tau * net.(fld) + (1 - tau) * target_net.(fld);
    end
end
