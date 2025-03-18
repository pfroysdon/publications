%% PPO Tutorial for Inverted Pendulum (From Scratch, No Built-in ML Toolbox)
% This script demonstrates a from-scratch implementation of Proximal Policy
% Optimization (PPO) using finite-difference gradient estimation.
% It uses a custom inverted pendulum environment defined in InvertedPendulum.m.
% Note: Finite difference gradient estimation is very slow and is provided only for educational purposes.

clc; clear; close all;
rng(1);

%% Create the Environment
env = InvertedPendulum();

%% Hyperparameters
numEpisodes      = 50;    % For finite-difference updates, use fewer episodes
maxEpisodeSteps  = env.maxEpisodeSteps;
gamma            = 0.99;  % Discount factor
ppoEpsilon       = 0.2;   % Clipping parameter
actorLR          = 1e-3;  % Actor learning rate
criticLR         = 1e-3;  % Critic learning rate
numEpochs        = 5;     % PPO update epochs per episode
miniBatchSize    = 8;     % Mini-batch size for updates
entropyCoef      = 0.01;  % Entropy bonus weight
delta            = 1e-5;  % Finite difference delta

%% Network Architecture Parameters
obs_dim    = 3;  % Observation: [cos(theta); sin(theta); theta_dot]
action_dim = 1;  % Action: torque
hidden_size = 16; % Use a smaller network for finite differences

%% Initialize Actor Network (as numeric arrays)
actor.W1 = 0.1 * randn(hidden_size, obs_dim);
actor.b1 = zeros(hidden_size, 1);
actor.W2 = 0.1 * randn(action_dim, hidden_size);
actor.b2 = zeros(action_dim, 1);
actor.log_std = -0.5 * ones(action_dim, 1);

%% Initialize Critic Network
critic.W1 = 0.1 * randn(hidden_size, obs_dim);
critic.b1 = zeros(hidden_size, 1);
critic.W2 = 0.1 * randn(1, hidden_size);
critic.b2 = 0;

%% Storage for Episode Rewards
episodeRewards = zeros(1, numEpisodes);

%% Main Training Loop
for ep = 1:numEpisodes
    % --- Collect a Trajectory ---
    obs_batch      = [];  % Each column is an observation
    actions_batch  = [];  % Each column is an action
    rewards_batch  = [];  % Row vector of rewards
    logprobs_batch = [];  % Row vector of log probabilities (from actor)
    values_batch   = [];  % Row vector of critic values
    
    s = env.reset();  % Get initial observation (3x1)
    done = false;
    while ~done
        % Actor forward pass
        [mean_a, std_a] = actor_forward(actor, s);
        % Sample an action from Gaussian
        a = mean_a + std_a .* randn(size(std_a));
        % Clip action within allowable torque limits
        a = max(min(a, env.maxTorque), -env.maxTorque);
        
        % Compute log probability of the action
        logp = gaussian_log_prob(a, mean_a, std_a);
        % Critic forward pass
        V = critic_forward(critic, s);
        
        % Step environment
        [s_next, r, done] = env.step(a);
        
        % Store data
        obs_batch      = [obs_batch, s];
        actions_batch  = [actions_batch, a];
        rewards_batch  = [rewards_batch, r];
        logprobs_batch = [logprobs_batch, logp];
        values_batch   = [values_batch, V];
        
        s = s_next;

        % env.render();
    end
    
    % --- Compute Returns and Advantages ---
    T = length(rewards_batch);
    returns = zeros(1, T);
    advantages = zeros(1, T);
    G = 0;
    for t = T:-1:1
        G = rewards_batch(t) + gamma * G;
        returns(t) = G;
        advantages(t) = G - values_batch(t);
    end
    advantages = (advantages - mean(advantages)) / (std(advantages)+1e-8);
    
    episodeRewards(ep) = sum(rewards_batch);
    
    % --- PPO Updates (using mini-batches and finite differences) ---
    num_samples = T;
    indices = 1:num_samples;
    for epoch = 1:numEpochs
        indices = indices(randperm(num_samples));
        for i = 1:miniBatchSize:num_samples
            batch_idx = indices(i:min(i+miniBatchSize-1, num_samples));
            obs_mb      = obs_batch(:, batch_idx);
            actions_mb  = actions_batch(:, batch_idx);
            old_logp_mb = logprobs_batch(batch_idx);
            adv_mb      = advantages(batch_idx);
            returns_mb  = returns(batch_idx);
            
            [actor, ~] = update_actor(actor, obs_mb, actions_mb, old_logp_mb, adv_mb, ppoEpsilon, actorLR, entropyCoef, delta);
            [critic, ~] = update_critic(critic, obs_mb, returns_mb, criticLR, delta);
        end
    end
    fprintf('Episode %d, Total Reward: %.2f\n', ep, episodeRewards(ep));

    % env.render();
end

%% Plot Convergence
figure;
plot(1:numEpisodes, episodeRewards, 'b-', 'LineWidth', 1);
xlabel('Episode'); ylabel('Total Episode Reward');
title('PPO Convergence on Inverted Pendulum (Finite Differences)');
grid on;
hold on;
plot(1:numEpisodes, movmean(episodeRewards, 5), 'r-', 'LineWidth', 2);
legend('Episode Reward', 'Moving Average');
hold off;

% save_all_figs_OPTION('results/ppo_convergence','png',1)

env.render();

%% Final Test Run and Trajectory Plot

% Reset the environment and initialize storage for state trajectory.
s = env.reset();
trajectory = [];  % Will store [theta, theta_dot] for each time step

% Run an episode with a test policy.
% (For demonstration, we use a random policy here. Replace with your trained policy as needed.)
done = false;
while ~done
    % Select a random action within allowable limits.
    action = -env.maxTorque + 2 * env.maxTorque * rand;
    
    % Take a step in the environment.
    [s, reward, done] = env.step(action);
    
    % Record the state from the environment.
    % Note: env.state stores [theta; theta_dot].
    trajectory = [trajectory; env.state'];
end

% Create a time vector using the environment time step.
dt = env.dt;
time = (0:size(trajectory,1)-1) * dt;

% Plot the trajectories.
figure;
subplot(2,1,1);
plot(time, trajectory(:,1), 'b', 'LineWidth', 2);
xlabel('Time (s)'); 
ylabel('Pole Angle (rad)');
title('Pole Angle Trajectory');
grid on;

subplot(2,1,2);
plot(time, trajectory(:,2), 'b', 'LineWidth', 2);
xlabel('Time (s)'); 
ylabel('Angular Velocity (rad/s)');
title('Angular Velocity Trajectory');
grid on;

save_all_figs_OPTION('results/ppo_trajectory','png',1)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Local Function Definitions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [mean_a, std_a] = actor_forward(actor, s)
    % Forward pass through actor network.
    % s is a column vector of size [obs_dim x 1]
    z1 = actor.W1 * s + actor.b1;
    h1 = max(0, z1);  % ReLU activation
    mean_a = actor.W2 * h1 + actor.b2;
    std_a = exp(actor.log_std);
end

function logp = gaussian_log_prob(a, mean, std)
    % Compute log probability of action a under a Gaussian distribution.
    logp = -0.5 * log(2*pi) - log(std) - 0.5 * ((a - mean)./std).^2;
    logp = sum(logp, 1);  % Sum over action dimensions (here, scalar)
end

function V = critic_forward(critic, s)
    % Forward pass through critic network.
    z1 = critic.W1 * s + critic.b1;
    h1 = max(0, z1);
    V = critic.W2 * h1 + critic.b2;
end

function loss = modelLoss_actor(actor, obs, actions, old_logp, adv, ppoEpsilon, entropyCoef)
    % Compute the PPO surrogate loss for the actor over a mini-batch.
    % obs: [obs_dim x N], actions: [action_dim x N]
    N = size(obs,2);
    losses = zeros(1, N);
    for i = 1:N
        x = obs(:, i);
        a = actions(:, i);
        [mean_a, std_a] = actor_forward(actor, x);
        new_logp = gaussian_log_prob(a, mean_a, std_a);
        ratio = exp(new_logp - old_logp(i));
        % Compute surrogate objectives.
        surr1 = ratio * adv(i);
        surr2 = min(max(ratio, 1-ppoEpsilon), 1+ppoEpsilon) * adv(i);
        losses(i) = -min(surr1, surr2);
    end
    actor_loss = mean(losses);
    % Entropy bonus (using mean of std as a rough proxy)
    entropy = mean(exp(actor.log_std));
    loss = actor_loss - entropyCoef * entropy;
end

function loss = modelLoss_critic(critic, obs, returns)
    % Compute mean-squared error loss for the critic over a mini-batch.
    N = size(obs,2);
    errors = zeros(1, N);
    for i = 1:N
        x = obs(:, i);
        V_pred = critic_forward(critic, x);
        errors(i) = (V_pred - returns(i))^2;
    end
    loss = mean(errors);
end

function grad = finiteDiffGradient(param, lossFunc, delta)
    % Compute finite-difference gradient of a function with respect to param.
    grad = zeros(size(param));
    for i = 1:numel(param)
        original = param(i);
        param(i) = original + delta;
        loss_plus = lossFunc(param);
        param(i) = original - delta;
        loss_minus = lossFunc(param);
        grad(i) = (loss_plus - loss_minus) / (2 * delta);
        param(i) = original;  % Restore original value
    end
end

function [actor, loss] = update_actor(actor, obs, actions, old_logp, adv, ppoEpsilon, actorLR, entropyCoef, delta)
    % Update actor parameters using finite-difference gradients.
    loss = modelLoss_actor(actor, obs, actions, old_logp, adv, ppoEpsilon, entropyCoef);
    
    % Compute gradients for each parameter using finite differences.
    % Update actor.W1
    lossFunc_W1 = @(W1) modelLoss_actor(setfield(actor, 'W1', W1), obs, actions, old_logp, adv, ppoEpsilon, entropyCoef);
    grad_W1 = finiteDiffGradient(actor.W1, lossFunc_W1, delta);
    actor.W1 = actor.W1 - actorLR * grad_W1;
    
    % Update actor.b1
    lossFunc_b1 = @(b1) modelLoss_actor(setfield(actor, 'b1', b1), obs, actions, old_logp, adv, ppoEpsilon, entropyCoef);
    grad_b1 = finiteDiffGradient(actor.b1, lossFunc_b1, delta);
    actor.b1 = actor.b1 - actorLR * grad_b1;
    
    % Update actor.W2
    lossFunc_W2 = @(W2) modelLoss_actor(setfield(actor, 'W2', W2), obs, actions, old_logp, adv, ppoEpsilon, entropyCoef);
    grad_W2 = finiteDiffGradient(actor.W2, lossFunc_W2, delta);
    actor.W2 = actor.W2 - actorLR * grad_W2;
    
    % Update actor.b2
    lossFunc_b2 = @(b2) modelLoss_actor(setfield(actor, 'b2', b2), obs, actions, old_logp, adv, ppoEpsilon, entropyCoef);
    grad_b2 = finiteDiffGradient(actor.b2, lossFunc_b2, delta);
    actor.b2 = actor.b2 - actorLR * grad_b2;
    
    % Update actor.log_std
    lossFunc_logstd = @(log_std) modelLoss_actor(setfield(actor, 'log_std', log_std), obs, actions, old_logp, adv, ppoEpsilon, entropyCoef);
    grad_logstd = finiteDiffGradient(actor.log_std, lossFunc_logstd, delta);
    actor.log_std = actor.log_std - actorLR * grad_logstd;
end

function [critic, loss] = update_critic(critic, obs, returns, criticLR, delta)
    % Update critic parameters using finite-difference gradients.
    loss = modelLoss_critic(critic, obs, returns);
    
    lossFunc_W1 = @(W1) modelLoss_critic(setfield(critic, 'W1', W1), obs, returns);
    grad_W1 = finiteDiffGradient(critic.W1, lossFunc_W1, delta);
    critic.W1 = critic.W1 - criticLR * grad_W1;
    
    lossFunc_b1 = @(b1) modelLoss_critic(setfield(critic, 'b1', b1), obs, returns);
    grad_b1 = finiteDiffGradient(critic.b1, lossFunc_b1, delta);
    critic.b1 = critic.b1 - criticLR * grad_b1;
    
    lossFunc_W2 = @(W2) modelLoss_critic(setfield(critic, 'W2', W2), obs, returns);
    grad_W2 = finiteDiffGradient(critic.W2, lossFunc_W2, delta);
    critic.W2 = critic.W2 - criticLR * grad_W2;
    
    lossFunc_b2 = @(b2) modelLoss_critic(setfield(critic, 'b2', b2), obs, returns);
    grad_b2 = finiteDiffGradient(critic.b2, lossFunc_b2, delta);
    critic.b2 = critic.b2 - criticLR * grad_b2;
end
