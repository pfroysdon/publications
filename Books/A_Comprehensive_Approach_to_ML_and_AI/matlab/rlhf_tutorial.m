% Reinforcement Learning with Human Feedback (RLHF) Tutorial in MATLAB
%
% In this simplified RLHF example, the environment is one-dimensional with states 0 to 10.
% The agent starts at state 0 and must reach state 10. At each time step, the agent chooses
% an action: move left (-1) or move right (+1). The episode terminates when state 10 is reached.
%
% A human provides feedback via a reward function, which is simulated here as:
%       r_human(s) = s.
% Instead of using this directly, the agent learns a reward model:
%       r_model(s) = theta0 + theta1 * s,
% where theta0 and theta1 are learned via linear regression on collected (state, human reward)
% pairs.
%
% The policy is parameterized as a logistic (sigmoid) function:
%       p(right|s) = sigmoid( phi * s + b ),
% so that the probability of moving right increases with the state value.
% The agent uses REINFORCE (policy gradient) to update the policy using the reward
% provided by the learned reward model.
%
% After training, the learned policy is evaluated by simulating an episode,
% and the resulting state trajectory is plotted. Additionally, a plot compares the
% learned reward model to the human reward function.


clear; clc; close all; rng(1);

%% Environment Parameters
stateMin = 0;
stateMax = 10;
startState = 0;
goalState = 10;

%% Human Reward Function (Simulated)
% Here, the human reward is simply the state value.
r_human = @(s) s;

%% Reward Model Initialization
% r_model(s) = theta0 + theta1 * s.
theta = [0; 0];  % 2x1 vector

%% Policy Initialization
% The policy is a logistic policy:
%   p(right|s) = sigmoid( phi * s + b )
% where phi and b are scalar parameters.
phi = 0;
b = 0;

%% Hyperparameters
numEpisodes = 1000;   % Number of training episodes
maxSteps = 20;        % Maximum steps per episode
alpha_policy = 0.01;  % Learning rate for policy update
alpha_model = 0.001;  % Learning rate for reward model update (via regression)
gamma = 0.99;         % Discount factor

% Memory for reward model training
model_states = [];
model_rewards = [];

%% Training Loop (RLHF)
episodeRewards = zeros(numEpisodes, 1);
for ep = 1:numEpisodes
    s = startState;
    states_ep = [];
    actions_ep = [];
    rewards_ep = [];
    
    % Generate an episode
    for t = 1:maxSteps
        states_ep = [states_ep, s]; %#ok<AGROW>
        
        % Compute policy: probability of moving right.
        p_right = sigmoid(phi * s + b);
        % Sample action: with probability p_right, move right (+1); else, move left (-1).
        if rand < p_right
            a = 1;
        else
            a = -1;
        end
        actions_ep = [actions_ep, a]; %#ok<AGROW>
        
        % Transition: new state = s + a, clipped to [stateMin, stateMax]
        s_next = min(max(s + a, stateMin), stateMax);
        
        % Get human reward for state s.
        r = r_human(s);
        rewards_ep = [rewards_ep, r]; %#ok<AGROW>
        
        s = s_next;
        if s == goalState
            break;
        end
    end
    
    % Compute discounted returns for this episode.
    T_ep = length(rewards_ep);
    G = zeros(1, T_ep);
    G(T_ep) = rewards_ep(T_ep);
    for t = T_ep-1:-1:1
        G(t) = rewards_ep(t) + gamma * G(t+1);
    end
    episodeRewards(ep) = sum(rewards_ep);
    
    % Append episode states and human rewards for reward model training.
    model_states = [model_states, states_ep]; %#ok<AGROW>
    model_rewards = [model_rewards, arrayfun(r_human, states_ep)]; %#ok<AGROW>
    
    % Update reward model via linear regression (closed-form solution with gradient descent step)
    % Our model: r_model(s) = theta0 + theta1 * s.
    X_model = [ones(1, length(model_states)); model_states];  % 2 x N
    y_model = model_rewards;  % 1 x N
    % Using gradient descent on mean squared error:
    pred = theta(1) + theta(2) * model_states;
    error_model = pred - model_rewards;
    grad_theta = [mean(error_model); mean(error_model .* model_states)];
    theta = theta - alpha_model * grad_theta;
    
    % Update policy using REINFORCE with reward model.
    % Use the learned reward model as the reward signal.
    r_model_ep = theta(1) + theta(2) * states_ep;  % 1 x T_ep
    baseline = mean(G);
    for t = 1:T_ep
        % Policy gradient for logistic policy: gradient of log p(right|s) = (I(a==+1) - p_right)*s.
        s_t = states_ep(t);
        p_right = sigmoid(phi * s_t + b);
        if actions_ep(t) == 1
            grad_log = (1 - p_right);
        else
            grad_log = (0 - p_right);
        end
        advantage = G(t) - baseline;  % advantage signal
        phi = phi + alpha_policy * grad_log * s_t * advantage;
        b = b + alpha_policy * grad_log * advantage;
    end
    
    if mod(ep,100)==0
        fprintf('Episode %d, Total Reward: %.2f\n', ep, sum(rewards_ep));
    end
end

%% Evaluation: Simulate an Episode Using the Greedy Policy
s = startState;
trajectory = s;
for t = 1:maxSteps
    p_right = sigmoid(phi * s + b);
    if p_right >= 0.5
        a = 1;
    else
        a = -1;
    end
    s = min(max(s + a, stateMin), stateMax);
    trajectory = [trajectory, s];
    if s == goalState
        break;
    end
end

%% Plot the Trajectory
figure;
plot(0:length(trajectory)-1, trajectory, '-o', 'LineWidth',2);
xlabel('Time Step');
ylabel('State');
title('Agent Trajectory under Learned Policy');
grid on;

% save_all_figs_OPTION('results/rlhf1','png',1)


%% Plot Reward Model vs. Human Reward
figure;
s_vals = linspace(stateMin, stateMax, 100);
r_model_vals = theta(1) + theta(2)*s_vals;
r_human_vals = arrayfun(r_human, s_vals);
plot(s_vals, r_model_vals, 'r-', 'LineWidth',2); hold on;
plot(s_vals, r_human_vals, 'b--', 'LineWidth',2);
xlabel('State');
ylabel('Reward');
legend('Learned Reward Model', 'Human Reward', 'Location', 'SE');
title('Reward Model vs. Human Reward');
grid on;


% save_all_figs_OPTION('results/rlhf2','png',1)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Local Function Definitions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function s = sigmoid(x)
    s = 1 ./ (1 + exp(-x));
end
