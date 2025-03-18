% Advanced GCN on Zachary's Karate Club with 4 hidden layers
%
% This tutorial implements a Graph Convolutional Network with four hidden
% layers (plus an output layer) for node classification on the Karate Club graph.
% The network architecture is:
%
%   H1 = ReLU( A_norm * (X * W1) )
%   H2 = ReLU( A_norm * (H1 * W2) )
%   H3 = ReLU( A_norm * (H2 * W3) )
%   H4 = ReLU( A_norm * (H3 * W4) )
%   H5 = A_norm * (H4 * W5)
%   Y_pred = softmaxRows(H5)
%
% We train the network using full supervision (all nodes labeled) with a 4‑group partition.

% Below is an advanced MATLAB tutorial that extends our previous GCN 
% example on Zachary’s Karate Club by adding two additional hidden layers 
% (for a total of four hidden layers before the output). This extra 
% capacity can help the model capture more complex graph features and 
% (hopefully) yield higher accuracy. In this example we use full 
% supervision (all nodes labeled) with a refined 4‑group partition. The 
% network architecture is as follows:
% 
%     Layer 1: H1=ReLU(Anorm (X W1))H1​=ReLU(Anorm​(XW1​))
%     Layer 2: H2=ReLU(Anorm (H1 W2))H2​=ReLU(Anorm​(H1​W2​))
%     Layer 3: H3=ReLU(Anorm (H2 W3))H3​=ReLU(Anorm​(H2​W3​))
%     Layer 4: H4=ReLU(Anorm (H3 W4))H4​=ReLU(Anorm​(H3​W4​))
%     Output: H5=Anorm (H4 W5)H5​=Anorm​(H4​W5​) followed by row‑wise softmax for class probabilities.
% 
% http://konect.cc/networks/ucidata-zachary/
% https://towardsdatascience.com/graph-convolutional-networks-introduction-to-gnns-24b3f60d6c95/

clear; clc; close all; rng(1);

%% 1. Load the Karate Club Graph
A = loadKarateClubAdjacency();   % 34x34 adjacency matrix
numNodes = size(A,1);

%% 2. Define a Refined 4-Group Labeling
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

%% 3. Define Node Features (Identity)
X = eye(numNodes);   % Use identity: each node as a one-hot vector.
d_input = numNodes;

%% 4. Compute Normalized Adjacency Matrix
I = eye(numNodes);
A_tilde = A + I;
D_tilde = diag(sum(A_tilde, 2));
D_inv_sqrt = diag(1 ./ sqrt(diag(D_tilde)));
A_norm = D_inv_sqrt * A_tilde * D_inv_sqrt;

%% 5. Use Full Supervision
train_mask = true(numNodes,1);

%% 6. Initialize GCN Weights
% We now use 4 hidden layers.
d_hidden1 = 32;
d_hidden2 = 32;
d_hidden3 = 32;
d_hidden4 = 32;
W1 = 0.01 * randn(d_input, d_hidden1);
W2 = 0.01 * randn(d_hidden1, d_hidden2);
W3 = 0.01 * randn(d_hidden2, d_hidden3);
W4 = 0.01 * randn(d_hidden3, d_hidden4);
W5 = 0.01 * randn(d_hidden4, numClasses);

%% 7. Training Setup
learningRate = 0.02;
epochs = 3000;
[coordsX, coordsY] = layoutGraph(A);

%% 8. Training Loop
figure('Name','Advanced GCN with 4 Hidden Layers');
for ep = 1:epochs
    % --- Forward Pass ---
    % Layer 1
    Z1 = X * W1;            
    H1 = A_norm * Z1;       
    H1 = max(H1, 0);         % ReLU
    
    % Layer 2
    Z2 = H1 * W2;
    H2 = A_norm * Z2;
    H2 = max(H2, 0);         % ReLU
    
    % Layer 3
    Z3 = H2 * W3;
    H3 = A_norm * Z3;
    H3 = max(H3, 0);         % ReLU
    
    % Layer 4
    Z4 = H3 * W4;
    H4 = A_norm * Z4;
    H4 = max(H4, 0);         % ReLU
    
    % Output layer
    Z5 = H4 * W5;
    H5 = A_norm * Z5;
    Y_pred = softmaxRows(H5);
    
    % --- Loss & Accuracy ---
    L = crossEntropyLoss(Y_pred(train_mask,:), labels(train_mask), numClasses);
    [~, pred_labels] = max(Y_pred, [], 2);
    acc = mean(pred_labels(train_mask) == labels(train_mask)) * 100;
    
    % --- Backpropagation ---
    % Start from output: dL/dH5
    dH5 = Y_pred;
    for i = 1:numNodes
        true_class = labels(i);
        dH5(i, true_class) = dH5(i, true_class) - 1;
    end
    dH5 = dH5 / numNodes;  % scale
    
    % Backprop through Output layer:
    dZ5 = A_norm' * dH5;         % shape: (numNodes x numClasses)
    gradW5 = H4' * dZ5;          % W5: (d_hidden4 x numClasses)
    
    % Backprop into Layer 4:
    dH4 = dZ5 * W5';             % shape: (numNodes x d_hidden4)
    mask4 = (H4 > 0);
    dH4(~mask4) = 0;
    dZ4 = A_norm' * dH4;         % shape: (numNodes x d_hidden4)
    gradW4 = H3' * dZ4;          % W4: (d_hidden3 x d_hidden4)
    
    % Backprop into Layer 3:
    dH3 = dZ4 * W4';             % shape: (numNodes x d_hidden3)
    mask3 = (H3 > 0);
    dH3(~mask3) = 0;
    dZ3 = A_norm' * dH3;         % shape: (numNodes x d_hidden3)
    gradW3 = H2' * dZ3;          % W3: (d_hidden2 x d_hidden3)
    
    % Backprop into Layer 2:
    dH2 = dZ3 * W3';             % shape: (numNodes x d_hidden2)
    mask2 = (H2 > 0);
    dH2(~mask2) = 0;
    dZ2 = A_norm' * dH2;         % shape: (numNodes x d_hidden2)
    gradW2 = H1' * dZ2;          % W2: (d_hidden1 x d_hidden2)
    
    % Backprop into Layer 1:
    dH1 = dZ2 * W2';             % shape: (numNodes x d_hidden1)
    mask1 = (H1 > 0);
    dH1(~mask1) = 0;
    dZ1 = A_norm' * dH1;         % shape: (numNodes x d_hidden1)
    gradW1 = X' * dZ1;           % W1: (d_input x d_hidden1)
    
    % --- Parameter Updates ---
    W5 = W5 - learningRate * gradW5;
    W4 = W4 - learningRate * gradW4;
    W3 = W3 - learningRate * gradW3;
    W2 = W2 - learningRate * gradW2;
    W1 = W1 - learningRate * gradW1;
    
    % --- Visualization ---
    if mod(ep,200)==0 || ep==epochs
        clf;
        G = graph(A);
        plotG = plot(G, 'XData', coordsX, 'YData', coordsY, ...
                     'NodeCData', pred_labels, 'MarkerSize', 8, ...
                     'EdgeColor', [0.7 0.7 0.7], 'LineWidth', 1.2);
        colormap(jet(numClasses));
        caxis([1 numClasses]);
        title(sprintf('Epoch %d | Loss: %.3f | Acc: %.2f%%', ep, L, acc));
        drawnow;
        fprintf('Epoch %d | Loss: %.3f | Acc: %.2f%%\n', ep, L, acc);
    end
end

fprintf('Final training accuracy: %.2f%%\n', acc);

% save_all_figs_OPTION('results/gcnn4','png',1)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Local Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function A = loadKarateClubAdjacency()
    % loadKarateClubAdjacency returns the adjacency matrix for Zachary's Karate Club.
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
    N = 34;
    A = zeros(N,N);
    for i = 1:size(A_data,1)
        v1 = A_data(i,1);
        v2 = A_data(i,2);
        A(v1,v2) = 1; 
        A(v2,v1) = 1;
    end
end

function [x, y] = layoutGraph(A)
    % layoutGraph returns 2D coordinates for nodes in adjacency A using a force layout.
    G = graph(A);
    p = plot(G, 'Layout', 'force', 'Iterations', 100, 'UseGravity', true);
    x = p.XData;
    y = p.YData;
    close;
end

function S = softmaxRows(X)
    % softmaxRows applies the softmax function row-wise on matrix X.
    X_exp = exp(X - max(X,[],2));
    S = X_exp ./ sum(X_exp,2);
end

function L = crossEntropyLoss(predProbs, labels, numClasses)
    % crossEntropyLoss calculates the average cross-entropy loss.
    n = size(predProbs,1);
    idx = sub2ind(size(predProbs), (1:n)', labels);
    chosen = predProbs(idx);
    L = -sum(log(chosen + 1e-15)) / n;
end
