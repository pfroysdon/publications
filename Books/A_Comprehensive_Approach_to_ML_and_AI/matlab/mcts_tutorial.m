% Monte Carlo Tree Search (MCTS) Tutorial using a Struct Array in MATLAB
%
% Game Description:
%   - State: a scalar number (starting at 0).
%   - Actions: add 1 or add 2.
%   - Terminal condition: state >= 10.
%         * If state == 10, reward = +1 (win).
%         * If state > 10, reward = -1 (loss).
%
% The algorithm performs:
%   1. Selection using UCT.
%   2. Expansion of an untried action.
%   3. Random simulation (rollout) until terminal state.
%   4. Backpropagation of the simulation reward.
%
% The best action from the root is selected based on highest average reward.

clear; clc; close all; rng(1);

%% Game Parameters
target = 10;         % Terminal state is when state >= 10
actions = [1, 2];    % Available actions: add 1 or add 2

%% MCTS Parameters
numIterations = 1000;  % Total number of MCTS iterations
uctConstant = 1.41;    % Exploration constant for UCT

%% Initialize Tree as a Struct Array
% Each node has the following fields:
%   state           - current state (scalar)
%   parent          - index of parent node (0 for root)
%   children        - vector of indices of child nodes
%   visits          - number of visits to this node
%   totalReward     - cumulative reward from rollouts through this node
%   untriedActions  - actions not yet tried at this node
%   actionFromParent- action taken from parent to reach this node
nodeCount = 1;
tree(nodeCount) = createNode(0, 0, actions, NaN);  % Root node (state 0, no parent)

%% Run MCTS Iterations
for iter = 1:numIterations
    % --- SELECTION ---
    current = 1;  % Start at root (index 1)
    % Traverse until a node with untried actions or a terminal node is found.
    while ~isTerminal(tree(current).state, target) && isempty(tree(current).untriedActions) && ~isempty(tree(current).children)
        current = selectChildUCT(tree, current, uctConstant);
    end
    
    % --- EXPANSION ---
    if ~isTerminal(tree(current).state, target) && ~isempty(tree(current).untriedActions)
        % Choose one untried action (pop the first one)
        action = tree(current).untriedActions(1);
        tree(current).untriedActions(1) = [];  % Remove this action from untried list
        % Create a new child node for this action
        newState = nextState(tree(current).state, action);
        nodeCount = nodeCount + 1;
        newNode = createNode(newState, current, actions, action);
        tree(nodeCount) = newNode;
        % Add the new node index to the parent's children list
        tree(current).children = [tree(current).children, nodeCount];
        % Set current node for simulation to the newly created node
        current = nodeCount;
    end
    
    % --- SIMULATION (ROLL-OUT) ---
    reward = rollout(tree(current).state, target, actions);
    
    % --- BACKPROPAGATION ---
    % Propagate reward up from current node to root using node indices.
    idx = current;
    while idx ~= 0
        tree(idx).visits = tree(idx).visits + 1;
        tree(idx).totalReward = tree(idx).totalReward + reward;
        idx = tree(idx).parent;
    end
end

%% Choose the Best Action from the Root
bestAvg = -Inf;
bestChild = -1;
for i = 1:length(tree(1).children)
    childIdx = tree(1).children(i);
    avgReward = tree(childIdx).totalReward / tree(childIdx).visits;
    if avgReward > bestAvg
        bestAvg = avgReward;
        bestChild = childIdx;
    end
end
bestAction = tree(bestChild).actionFromParent;
fprintf('From state %d, the best action is: +%d (avg reward: %.2f)\n', tree(1).state, bestAction, bestAvg);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Local Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function node = createNode(state, parent, actions, actionFromParent)
    % createNode constructs a node with given state, parent index, available actions, etc.
    node.state = state;
    node.parent = parent;           % 0 for root
    node.children = [];             % Initially no children
    node.visits = 0;
    node.totalReward = 0;
    node.untriedActions = actions;  % All actions are available initially
    node.actionFromParent = actionFromParent;
end

function childIdx = selectChildUCT(tree, parentIdx, uctConst)
    % selectChildUCT selects a child of tree(parentIdx) using the UCT formula.
    % UCT = (child.totalReward/child.visits) + uctConst * sqrt( log(tree(parentIdx).visits+1) / child.visits )
    % For any child with 0 visits, we assign UCT = Inf.
    children = tree(parentIdx).children;
    bestUCT = -Inf;
    childIdx = children(1);
    for i = 1:length(children)
        cIdx = children(i);
        if tree(cIdx).visits == 0
            uctValue = Inf;
        else
            avgReward = tree(cIdx).totalReward / tree(cIdx).visits;
            uctValue = avgReward + uctConst * sqrt(log(tree(parentIdx).visits + 1) / tree(cIdx).visits);
        end
        if uctValue > bestUCT
            bestUCT = uctValue;
            childIdx = cIdx;
        end
    end
end

function term = isTerminal(state, target)
    % isTerminal returns true if state is terminal (state >= target).
    term = (state >= target);
end

function next = nextState(state, action)
    % nextState computes the next state given current state and action.
    next = state + action;
end

function reward = rollout(state, target, actions)
    % rollout simulates a random play-out from the given state until a terminal state.
    % Returns reward: +1 if state == target; -1 if state > target.
    while ~isTerminal(state, target)
        a = actions(randi(length(actions)));
        state = nextState(state, a);
    end
    if state == target
        reward = 1;
    else
        reward = -1;
    end
end
