% qLearningPlotOptimalPath.m
%
% This script demonstrates Q-learning on a directed graph with 6 states
% (states 0 through 5, using 0-based numbering). The reward matrix R defines
% valid transitions (R(s,a) ≠ -1) and rewards (with 100 awarded when reaching
% the goal state 5). Q-values are learned using the update rule:
%
%   Q(s,a) = Q(s,a) + α [ R(s,a) + γ max_a' Q(s',a') - Q(s,a) ]
%
% Each episode begins at state 2 (0-based) and ends when state 5 is reached.
% After training, the optimal (greedy) policy is extracted, and the directed
% graph is plotted using MATLAB's digraph, with nodes numbered 1–6. The optimal
% path (converted to 1-based indices) is highlighted in red.

clear; clc; close all; rng(1);

%% 1. Define the Reward Matrix R (6x6)
% States (0-based): 0,1,2,3,4,5. In MATLAB we use 1-based indexing:
% index 1 corresponds to state 0, and index 6 corresponds to state 5.
R = [ -1 , -1 , -1 , -1 ,  0 , -1;
      -1 , -1 , -1 ,  0 , -1 , 100;
      -1 , -1 , -1 ,  0 , -1 , -1;
      -1 ,  0 ,  0 , -1 ,  0 , -1;
       0 , -1 , -1 ,  0 , -1 , 100;
      -1 ,  0 , -1 , -1 ,  0 , 100];

numStates = 6;   % MATLAB indices 1 to 6 (states 0 to 5)
startState = 2;  % (0-based state 2, MATLAB index = 3)
goalState  = 5;  % (0-based state 5, MATLAB index = 6)

%% 2. Q-Learning Hyperparameters
alpha = 0.1;       % Learning rate
gamma = 0.8;       % Discount factor
epsilon = 0.1;     % Epsilon-greedy exploration
numEpisodes = 400; % Number of training episodes

%% 3. Initialize Q-Table
% Q is a 6x6 matrix where Q(s,a) is the Q-value for transitioning from state s to state a.
Q = zeros(numStates, numStates);
Q_out = zeros(numEpisodes,1);

%% 4. Q-Learning Loop
for episode = 1:numEpisodes
    s = startState;  % Start each episode at state 2 (0-based)
    while s ~= goalState
        s_idx = s + 1;  % Convert to 1-based index for Q-table
        % Build valid action set: actions 'a' where R(s_idx,a) ~= -1.
        validActions = find(R(s_idx,:) ~= -1) - 1;  % Convert back to 0-based
        if isempty(validActions)
            break;
        end
        
        % Epsilon-greedy action selection
        if rand < epsilon
            a = validActions(randi(length(validActions)));
        else
            % Among valid actions, choose the one with highest Q-value.
            Q_valid = Q(s_idx, validActions+1);
            [~, bestIdx] = max(Q_valid);
            a = validActions(bestIdx);
        end
        
        a_idx = a + 1;
        s_next = a;  % The action indicates the next state (0-based)
        rew = R(s_idx, a_idx);
        
        if s_next == goalState
            Q(s_idx,a_idx) = Q(s_idx,a_idx) + alpha * (rew - Q(s_idx,a_idx));
            break;
        else
            next_idx = s_next + 1;
            Q(s_idx,a_idx) = Q(s_idx,a_idx) + alpha * (rew + gamma * max(Q(next_idx,:)) - Q(s_idx,a_idx));
        end
        s = s_next;
    end
    Q_out(episode,1) = norm(Q);  
end

%% 5. Display the Learned Q-Table
disp('Learned Q-values (rows: current state, cols: next state):');
disp(Q);

plot(Q_out, 'b', 'LineWidth', 2)
xlabel('iteration')
ylabel('$\|Q\|_2$','Interpreter','latex')
title('2-norm of the Q value at each iteration')

% save_all_figs_OPTION('results/q_learning_iterations','png',1)

%% 6. Extract the Optimal Path from State 2 to State 5
optimalPath = startState;  % Start at state 2 (0-based)
s = startState;
while s ~= goalState
    s_idx = s + 1;
    [~, a] = max(Q(s_idx,:));
    optimalPath(end+1) = a - 1;  % Convert from 1-based to 0-based
    s = a - 1;  % a is 1-based, convert to 0-based
end
fprintf('Optimal path (0-based): ');
disp(optimalPath);

%% 7. Plot the Directed Graph and Highlight the Optimal Path
% Explanation
% 
% Edge List Construction:
% We iterate over the reward matrix RR (which is 6×6) and, for every valid 
% transition (i.e. where R(i,j)≠−1R(i,j)=−1), we store the starting node 
% index ii and the ending node index jj in numeric arrays (edgesFrom and 
% edgesTo). We also store the corresponding reward value in a numeric array 
% (edgeWeights).
% 
% Graph Construction:
% We then pass these numeric arrays to MATLAB’s digraph function. This 
% ensures that the edge weights are numeric and avoids the error.
% 
% Optimal Path Highlighting:
% After constructing the directed graph, we highlight the edges 
% corresponding to the optimal path (converted to 1‑based indices) using 
% the highlight function.
% 
% Build edge lists from R: include an edge from state (i-1) to state (j-1)
% if R(i,j) ~= -1. Here we store 0-based indices.
edgesFrom0 = [];
edgesTo0 = [];
edgeWeights = [];
for i = 1:numStates
    for j = 1:numStates
        if R(i,j) ~= -1
            edgesFrom0(end+1) = i - 1;  %#ok<AGROW>
            edgesTo0(end+1) = j - 1;    %#ok<AGROW>
            edgeWeights(end+1) = R(i,j);  %#ok<AGROW>
        end
    end
end

% Since digraph requires positive integers, add 1 to get 1-based indices.
G = digraph(edgesFrom0+1, edgesTo0+1, edgeWeights);

% Create node labels that are 0-based.
nodeLabels = arrayfun(@(x) num2str(x-1), 1:numStates, 'UniformOutput', false);

% Plot the graph using a layered layout.
figure;
hG = plot(G, 'Layout', 'layered', 'Direction', 'right','LineWidth', 1.5);
hG.NodeLabel = nodeLabels;  % relabel nodes to 0-based
hG.EdgeColor = 'b';
hG.NodeColor = 'b';
title('Directed Graph with Optimal Path from State 2 to State 5');
xlabel('State (0-based)');
ylabel('Layer');

% Convert the optimal path from 0-based to 1-based for internal indexing.
optimalPathIndices = optimalPath + 1;

hold on;
% Highlight optimal path edges in red.
for k = 1:length(optimalPathIndices)-1
    s_from = optimalPathIndices(k);
    s_to = optimalPathIndices(k+1);
    % Find the edge index in G that goes from s_from to s_to.
    edgeIdx = find((G.Edges.EndNodes(:,1) == s_from) & (G.Edges.EndNodes(:,2) == s_to));
    if ~isempty(edgeIdx)
        highlight(hG, s_from, s_to, 'EdgeColor', 'r', 'LineWidth', 2);
    end
end
axis tight
hold off;

% save_all_figs_OPTION('results/q_learning_digraph','png',1)

%% 8. Visualize the Learned Q-Table and the Optimal Path using imagesc
% 
% Explanation
% 
% imagesc(Q):
% Displays the Q‑table as a color-coded image. The x‑axis represents the 
% "next state" (action) and the y‑axis represents the "current state" (from 
% which the action is taken).
% 
% colormap(jet) & colorbar:
% Apply a colorful colormap and show a colorbar for reference.
% 
% Optimal Path Markers:
% The optimal path (learned by Q‑learning) is stored in the variable 
% optimalPath in 0‑based indices. We convert it to MATLAB’s 1‑based 
% indexing and then plot markers (black circles) at the matrix cell 
% corresponding to each transition in the optimal path.
% (For example, if the optimal path is [2, 4, 5] in 0‑based numbering, 
% then the markers will be plotted at (column 5, row 3) for the transition 
% from state 2 to state 4 and at (column 6, row 5) for the transition from 
% state 4 to state 5.)
% 
% This visualization provides an alternate view of the Q‑learning 
% optimization process and the learned policy. 

% Display the Q-table using imagesc.
figure;
imagesc(Q);
colormap(jet);
colorbar;
xlabel('Next State (0-based)');
ylabel('Current State (0-based)');
title('Learned Q-values Matrix');

% Set tick marks and labels to 0-based.
numStates = size(Q,1);
xticks(1:numStates);
xticklabels(0:numStates-1);
yticks(1:numStates);
yticklabels(0:numStates-1);

% Overlay markers for the optimal path.
% 'optimalPath' is assumed to be 0-based (e.g. [2, 4, 5, ...]).
% For plotting, we convert it to 1-based indices.
optimalPath1 = optimalPath + 1;

hold on;
% For each transition in the optimal path, plot a marker at the cell center.
for i = 1:length(optimalPath1)-1
    currentState = optimalPath1(i);
    nextState    = optimalPath1(i+1);
    % Plot a marker at (nextState, currentState) in the Q-table
    plot(nextState, currentState, 'ko', 'MarkerSize', 10, 'LineWidth', 2);
end
hold off;


% save_all_figs_OPTION('results/q_learning_imagesc','png',1)