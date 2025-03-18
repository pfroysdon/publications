%% gcnKarateTutorial_advanced.m
% Graph Convolutional Network (GCN) on Zachary's Karate Club in MATLAB
%
% This script extends the previous GCN example by:
%   1. Using TWO hidden layers for higher model capacity.
%   2. Training ONLY on a subset of nodes (partial supervision).
%   3. Refining the 4-group labeling for the Karate Club graph.
%   4. Increasing the number of epochs and using a smaller learning rate.
%
% 
% Below is an updated MATLAB tutorial for a Graph Convolutional Network on Zachary’s Karate Club that aims for higher training accuracy. In this version we make several changes to improve performance:
% 
%     Full Supervision: All nodes are used for training (i.e. the training mask is all true).
%     Increased Model Capacity: We use larger hidden dimensions (32 instead of 16) in both hidden layers.
%     Longer Training: We increase the number of epochs.
%     Reduced Learning Rate: A smaller learning rate (0.001) is used to stabilize learning.
% 
% 
% Explanation
% 
%     Full Supervision:
%     Instead of training on a small subset of nodes, we now train on all nodes (i.e. train_mask = true(numNodes,1)). This gives the model access to more training signals.
% 
%     Increased Model Capacity:
%     Both hidden layers have their dimension increased to 32. This added capacity should help the network capture the graph structure better.
% 
%     Longer and More Stable Training:
%     We use 3000 epochs with a reduced learning rate of 0.001. The smaller learning rate allows for more stable convergence over many epochs.
% 
%     Visualization:
%     The graph layout is computed once and used consistently. During training, we update the plot every 100 epochs to show progress.
% 
% These modifications typically yield significantly higher training accuracy (often above 80%). You may further experiment with hyperparameters, regularization, or alternative label partitions for even better performance. Enjoy exploring Graph Convolutional Networks in MATLAB!
% 
% http://konect.cc/networks/ucidata-zachary/
% https://towardsdatascience.com/graph-convolutional-networks-introduction-to-gnns-24b3f60d6c95/

clear; clc; close all; rng(1);

%% 1. Load the Karate Club Graph
A = loadKarateClubAdjacency();   % 34x34 adjacency matrix
numNodes = size(A,1);

%% 2. Refined 4-Group Labeling
% We'll define a partition that tries to reflect some natural substructures.
% You can adapt these group definitions if you know a better partition.
% The actual "Karate Club" is often used with 2 classes, but here we do 4.
group1 = [1,2,3,4,8,14];                      % 6 nodes
group2 = [5,6,7,11,12,13,17];                 % 7 nodes
group3 = [9,10,15,16,19,21,23,25,27,29,30,31,33,34];  % 14 nodes
group4 = [18,20,22,24,26,28,32];              % 7 nodes
labels = zeros(numNodes,1);
labels(group1) = 1;
labels(group2) = 2;
labels(group3) = 3;
labels(group4) = 4;
numClasses = 4;

%% 3. Node Features (Identity)
% We'll use an identity matrix as node features.
X = eye(numNodes);
d_input = numNodes;

%% 4. Normalized Adjacency
I = eye(numNodes);
A_tilde = A + I;
D_tilde = diag(sum(A_tilde,2));
D_inv_sqrt = diag(1 ./ sqrt(diag(D_tilde)));
A_norm = D_inv_sqrt * A_tilde * D_inv_sqrt;

%% 5. Define a PARTIAL Training Set
% We label only a handful of nodes from each group. The rest are unlabeled
% (we won't compute loss for them).
train_mask = false(numNodes,1);

% We'll pick 8 labeled nodes total (2 from each group).
train_mask([1,2]) = true;    % from group1
train_mask([5,6]) = true;    % from group2
train_mask([9,10])= true;    % from group3
train_mask([18,20])=true;    % from group4

% We'll measure accuracy on ALL nodes, but only compute cross-entropy
% for the nodes in train_mask.

%% 6. GCN Weights
% We'll use TWO hidden layers + final output layer:
%   H1 = ReLU(A_norm * X * W1)
%   H2 = ReLU(A_norm * H1 * W2)
%   H3 = A_norm * H2 * W3 => row-wise softmax => predictions
d_hidden1 = 30;
d_hidden2 = 30;
W1 = 0.01 * randn(d_input, d_hidden1);
W2 = 0.01 * randn(d_hidden1, d_hidden2);
W3 = 0.01 * randn(d_hidden2, numClasses);

%% 7. Training Setup
learningRate = 0.02;
epochs = 3000;
[coordsX, coordsY] = layoutGraph(A);  % 2D layout for plotting

%% 8. Training Loop
figure('Name','GCN Karate Club (Advanced)');
for ep = 1:epochs
    % ---------- Forward Pass ----------
    % Layer 1
    Z1 = X * W1;            % shape: (34 x d_hidden1)
    H1 = A_norm * Z1;       % shape: (34 x d_hidden1)
    H1 = max(H1,0);         % ReLU
    
    % Layer 2
    Z2 = H1 * W2;           % shape: (34 x d_hidden2)
    H2 = A_norm * Z2;       % shape: (34 x d_hidden2)
    H2 = max(H2,0);         % ReLU
    
    % Output Layer
    Z3 = H2 * W3;           % shape: (34 x numClasses)
    H3 = A_norm * Z3;       % shape: (34 x numClasses)
    Y_pred = softmaxRows(H3);
    
    % ---------- Loss & Accuracy ----------
    L = crossEntropyLoss(Y_pred(train_mask,:), labels(train_mask), numClasses);
    [~, pred_labels] = max(Y_pred,[],2);
    acc = mean(pred_labels == labels) * 100;   % measure on ALL nodes
    
    % ---------- Backprop ----------
    % We'll compute gradient w.r.t. W3, W2, W1.
    
    % dL/dH3
    dH3 = Y_pred;
    train_indices = find(train_mask);
    for i = 1:length(train_indices)
        node_i = train_indices(i);
        true_c = labels(node_i);
        dH3(node_i,true_c) = dH3(node_i,true_c) - 1;
    end
    dH3 = dH3 / sum(train_mask);  % scale by #labeled
    
    % H3 = A_norm * Z3 => dZ3 = A_norm' * dH3 (A_norm is symmetric)
    dZ3 = A_norm' * dH3;          % shape: (34 x numClasses)
    gradW3 = H2' * dZ3;           % W3 shape: (d_hidden2 x numClasses)
    
    % dH2 = dZ3 * W3' => apply ReLU mask
    dH2 = (dZ3 * W3');
    mask2 = (H2 > 0);
    dH2(~mask2) = 0;
    % H2 = A_norm * Z2 => dZ2 = A_norm' * dH2
    dZ2 = A_norm' * dH2;          % shape: (34 x d_hidden2)
    gradW2 = H1' * dZ2;           % W2 shape: (d_hidden1 x d_hidden2)
    
    % dH1 = dZ2 * W2'
    dH1 = dZ2 * W2';
    mask1 = (H1 > 0);
    dH1(~mask1) = 0;
    % H1 = A_norm * Z1 => dZ1 = A_norm' * dH1
    dZ1 = A_norm' * dH1;          % shape: (34 x d_hidden1)
    gradW1 = X' * dZ1;            % W1 shape: (d_input x d_hidden1)
    
    % ---------- Parameter Updates ----------
    W3 = W3 - learningRate * gradW3;
    W2 = W2 - learningRate * gradW2;
    W1 = W1 - learningRate * gradW1;
    
    % ---------- Visualization ----------
    if mod(ep,300)==0 || ep==epochs
        clf;
        G = graph(A);
        plotG = plot(G, 'XData', coordsX, 'YData', coordsY, ...
                     'NodeCData', pred_labels, 'MarkerSize', 7, ...
                     'EdgeColor', [0.7 0.7 0.7], 'LineWidth', 1.0);
        colormap(jet(numClasses));
        caxis([1 numClasses]);
        title(sprintf('Epoch %d | Loss: %.3f | Acc: %.2f%%', ep, L, acc));
        drawnow;
        fprintf('Epoch %d | Loss: %.3f | Acc: %.2f%%\n', ep, L, acc);
    end
    
end

fprintf('Final training accuracy (all nodes): %.2f%%\n', acc);

% save_all_figs_OPTION('results/gcnn3','png',1)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Local Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function A = loadKarateClubAdjacency()
% loadKarateClubAdjacency returns adjacency matrix for Zachary's Karate Club (34 nodes).
% Hard-coded edges from known dataset.
A_data = [
1 2
1 3
2 3
1 4
2 4
3 4
1 5
1 6
1 7
5 7
6 7
1 8
2 8
3 8
4 8
1 9
3 9
3 10
1 11
5 11
6 11
1 12
1 13
4 13
1 14
2 14
3 14
4 14
6 17
7 17
1 18
2 18
1 20
2 20
1 22
2 22
24 26
25 26
3 28
24 28
25 28
3 29
24 30
27 30
2 31
9 31
1 32
25 32
26 32
29 32
3 33
9 33
15 33
16 33
19 33
21 33
23 33
24 33
30 33
31 33
32 33
9 34
10 34
14 34
15 34
16 34
19 34
20 34
21 34
23 34
24 34
27 34
28 34
29 34
30 34
31 34
32 34
33 34
];
N=34;
A = zeros(N,N);
for i = 1:size(A_data,1)
    v1 = A_data(i,1);
    v2 = A_data(i,2);
    A(v1,v2)=1; 
    A(v2,v1)=1;
end
end

function [x, y] = layoutGraph(A)
% layoutGraph returns 2D coordinates for nodes in adjacency A using a force layout.
    G = graph(A);
    p = plot(G,'Layout','force','Iterations',100,'UseGravity',true);
    x = p.XData; 
    y = p.YData;
    close;
end

function S = softmaxRows(X)
% softmaxRows applies row-wise softmax on matrix X.
    X_exp = exp(X - max(X,[],2));
    S = X_exp ./ sum(X_exp,2);
end

function L = crossEntropyLoss(predProbs, labels, numClasses)
% crossEntropyLoss calculates average cross-entropy for predicted vs. true labels.
    n = size(predProbs,1);
    idx = sub2ind(size(predProbs), (1:n)', labels);
    chosen = predProbs(idx);
    L = -sum(log(chosen + 1e-15)) / n;
end
