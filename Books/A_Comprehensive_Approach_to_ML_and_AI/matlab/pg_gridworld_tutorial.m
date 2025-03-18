% This tutorial demonstrates the REINFORCE policy gradient algorithm
% on a simple 5x5 gridworld environment.
%
% Environment:
%   - 25 states (0-based: 0 to 24, represented as 1 to 25 in MATLAB).
%   - Start at state 0 (MATLAB index 1) and goal at state 24 (MATLAB index 25).
%   - Actions: 1=Up, 2=Right, 3=Down, 4=Left.
%   - Each step: reward = -1; reaching the goal gives +10 and terminates the episode.
%
% Policy:
%   - Parameterized by Θ (4×25 matrix). For state s, 
%     p(a|s) = softmax(Θ(:, s)).
%
% Learning:
%   - The REINFORCE (policy gradient) algorithm is used to update Θ.
%
% Visualization:
%   - The grid is displayed with 0-based state labels.
%   - Arrows indicate the best (greedy) action in each state.
%   - A convergence figure shows episode reward over time.


clear; clc; close all; rng(1);

%% Environment Parameters
gridRows = 5;
gridCols = 5;
numStates = gridRows * gridCols;  % 25 states
startState = 1;  % 1-based index for state 0
goalState = numStates;  % 1-based index for state 24
numActions = 4;  % Up, Right, Down, Left
maxSteps = 50;   % Maximum steps per episode
gamma = 0.99;    % Discount factor

%% Policy Parameters
% Θ is a 4x25 matrix (each column for one state)
Theta = randn(numActions, numStates) * 0.01;

%% Hyperparameters for REINFORCE
alpha = 0.01;       % Learning rate for policy gradient
numEpisodes = 10000;  % Number of episodes for training

% To track convergence, record the total reward per episode.
episodeRewards = zeros(numEpisodes, 1);

%% Training Loop (REINFORCE)
for ep = 1:numEpisodes
    % Generate one episode using the current policy.
    s = startState;  % start state (1-based, corresponds to state 0)
    states = [];
    actions = [];
    rewards = [];
    
    for t = 1:maxSteps
        states(end+1) = s; %#ok<AGROW>
        % Select an action using the current policy (softmax over Θ(:, s)).
        probs = softmax(Theta(:, s));
        a = sampleAction(probs);  % a is in {1,2,3,4}
        actions(end+1) = a; %#ok<AGROW>
        % Take the action and observe next state, reward, and done flag.
        [s_next, r, done] = step(s, a, gridRows, gridCols, goalState);
        rewards(end+1) = r; %#ok<AGROW>
        s = s_next;
        if done
            break;
        end
    end
    
    % Record the total reward for this episode.
    episodeRewards(ep) = sum(rewards);
    
    % Compute discounted returns for the episode.
    T_ep = length(rewards);
    G = zeros(1, T_ep);
    G(T_ep) = rewards(T_ep);
    for t = T_ep-1:-1:1
        G(t) = rewards(t) + gamma * G(t+1);
    end
    
    % Update policy parameters using REINFORCE update rule.
    for t = 1:T_ep
        s_t = states(t);  % 1-based state index
        a_t = actions(t); % action index (1 to 4)
        % Gradient of log p(a|s) is (oneHot(a) - p) for state s.
        grad_log = oneHot(a_t, numActions) - softmax(Theta(:, s_t));
        % Update the parameters for state s_t.
        Theta(:, s_t) = Theta(:, s_t) + alpha * grad_log * G(t);
    end
end

%% Plot Convergence: Episode Reward vs. Episode Number
figure;
plot(1:numEpisodes, episodeRewards, 'b-', 'LineWidth', 1);
xlabel('Episode'); ylabel('Total Episode Reward');
title('Policy Gradient Convergence Over Time');
grid on;
% Compute and overlay a moving average (window of 50 episodes)
window = 50;
avgRewards = movmean(episodeRewards, window);
hold on;
plot(1:numEpisodes, avgRewards, 'r-', 'LineWidth', 2);
legend('Episode Reward', sprintf('Moving Average (%d episodes)', window));
hold off;

% save_all_figs_OPTION('results/pg_convergence','png',1)

%% Evaluate Learned Policy: Greedy Trajectory
s = startState;
optimalPath = s;
for t = 1:maxSteps
    % Use greedy policy (choose action with highest probability).
    [~, a] = max(Theta(:, s));
    [s_next, ~, done] = step(s, a, gridRows, gridCols, goalState);
    optimalPath(end+1) = s_next;
    s = s_next;
    if done
        break;
    end
end
fprintf('Optimal path (1-based, 0-based shown in parentheses):\n');
disp([optimalPath; optimalPath - 1]);

%% Visualize the Grid and Learned Policy
figure;
% Display grid with 0-based state labels.
stateMatrix = reshape(1:numStates, gridRows, gridCols);
imagesc(stateMatrix);
colormap(gray);
colorbar;
% Replace tick labels with 0-based numbers.
xticks(1:gridCols); xticklabels(0:gridCols-1);
yticks(1:gridRows); yticklabels(0:gridRows-1);
hold on;
% Overlay arrows for the learned policy: use greedy action from each state.
for s = 1:numStates
    [row, col] = ind2sub([gridRows, gridCols], s);
    p = softmax(Theta(:, s));
    [~, bestA] = max(p);
    % Determine arrow direction based on best action.
    switch bestA
        case 1, dx = 0; dy = -0.4; % up
        case 2, dx = 0.4; dy = 0;  % right
        case 3, dx = 0; dy = 0.4;  % down
        case 4, dx = -0.4; dy = 0; % left
    end
    quiver(col, row, dx, dy, 0, 'r', 'LineWidth', 2, 'MaxHeadSize', 2.5);
end
hold off;
title('Learned Policy: Gridworld with 0-based State Labels and Arrows');

% save_all_figs_OPTION('results/pg_gridWorld','png',1)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Local Function Definitions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function p = softmax(x)
    x = x - max(x);
    p = exp(x) / sum(exp(x));
end

function a = sampleAction(probs)
    r = rand;
    cumulative = cumsum(probs);
    a = find(cumulative >= r, 1, 'first');
end

function y = oneHot(a, numActions)
    y = zeros(numActions, 1);
    y(a) = 1;
end

% The discount function is provided for clarity but is not used here.
function G = discount(rewards, gamma)
    T = length(rewards);
    G = zeros(1, T);
    G(T) = rewards(T);
    for t = T-1:-1:1
        G(t) = rewards(t) + gamma * G(t+1);
    end
end

function [s_next, r, done] = step(s, action, gridRows, gridCols, goalState)
    % step returns the next state, reward, and done flag.
    % s: current state (1-based index).
    % action: integer 1 (up), 2 (right), 3 (down), 4 (left).
    % gridRows, gridCols: grid dimensions.
    % goalState: goal state index (1-based).
    % If the action would move the agent off the grid, the agent stays in place.
    
    [row, col] = ind2sub([gridRows, gridCols], s);
    switch action
        case 1  % up
            newRow = max(row - 1, 1);
            newCol = col;
        case 2  % right
            newRow = row;
            newCol = min(col + 1, gridCols);
        case 3  % down
            newRow = min(row + 1, gridRows);
            newCol = col;
        case 4  % left
            newRow = row;
            newCol = max(col - 1, 1);
        otherwise
            newRow = row;
            newCol = col;
    end
    s_next = sub2ind([gridRows, gridCols], newRow, newCol);
    
    if s_next == goalState
        r = 10;
        done = true;
    else
        r = -1;
        done = false;
    end
end
