% Graph Convolutional Network (GCN) on Zachary's Karate Club in MATLAB
% 
% Explanation of Key Steps
% 
%     Loading the Karate Club Graph:
%     The function loadKarateClubAdjacency defines the edges for the 34-node karate club network. The result is a 34×34 adjacency matrix AA.
% 
%     Classes and Features:
%         We define four arbitrary class labels (1 to 4) for each node, purely for demonstration.
%         We use the identity matrix as node features (so each node is one-hot).
% 
%     GCN Normalization:
%         We add self-loops (A+IA+I).
%         Compute D−1/2(A+I)D−1/2D−1/2(A+I)D−1/2 to get the normalized adjacency AnormAnorm​.
% 
%     GCN Forward Pass:
%     H(1)=ReLU(AnormXW1),H(2)=AnormH(1)W2,
%     ​H(1)=ReLU(Anorm​XW1​),H(2)=Anorm​H(1)W2​,​
% 
%     then apply row-wise softmax for class probabilities.
% 
%     Loss and Backprop:
%         We compute cross-entropy with the known labels.
%         A simple gradient descent is done for the two weight matrices {W1,W2}{W1​,W2​}.
% 
%     Visualization:
%         We compute a 2D “force” layout once and store node coordinates.
%         After each training epoch (or every 10 epochs), we update the figure, color nodes by predicted class, and show the epoch, loss, and accuracy in the title.
% 
% Running this code should produce a figure (updated during training) similar to the attached example. Once training completes, you’ll see a final classification of the nodes into four groups. Enjoy experimenting with GCNs in MATLAB!

% This improved version:
%   - Defines a more coherent 4-group partition of the Karate Club nodes.
%   - Uses a larger hidden dimension, more epochs, and a smaller learning rate.
%   - Often achieves substantially higher training accuracy.
%
% http://konect.cc/networks/ucidata-zachary/
% https://towardsdatascience.com/graph-convolutional-networks-introduction-to-gnns-24b3f60d6c95/

clear; clc; close all; rng(1);

%% 1. Load the Karate Club Graph
A = loadKarateClubAdjacency();   % 34x34 adjacency matrix
numNodes = size(A,1);

%% 2. Define a 4-Group Labeling
% We define a partition into 4 groups that reflect some substructures.
% (This is arbitrary, but more coherent than random labeling.)
labels = zeros(numNodes,1);

group1 = [1,2,3,4,8,14];                                  % 6 nodes
group2 = [5,6,7,11,12,13,17];                             % 7 nodes
group3 = [9,10,15,16,19,21,23,25,27,29,30,31,33,34];       % 14 nodes
group4 = [18,20,22,24,26,28,32];                          % 7 nodes

labels(group1) = 1;
labels(group2) = 2;
labels(group3) = 3;
labels(group4) = 4;
numClasses = 4;

%% 3. Node Features (Identity)
% We'll use an identity matrix as node features (one-hot per node).
X = eye(numNodes);
d_input = numNodes;  % each node is a one-hot of length 34

%% 4. Normalized Adjacency
I = eye(numNodes);
A_tilde = A + I;
D_tilde = diag(sum(A_tilde,2));
D_inv_sqrt = diag(1 ./ sqrt(diag(D_tilde)));
A_norm = D_inv_sqrt * A_tilde * D_inv_sqrt;

%% 5. GCN Weights
% Increase hidden dimension for more capacity.
d_hidden = 30;
W1 = 0.01 * randn(d_input, d_hidden);
W2 = 0.01 * randn(d_hidden, numClasses);

%% 6. Training Setup
learningRate = 0.01;
epochs =1000;
train_mask = true(numNodes,1);  % Train on all nodes (full supervision).

% We'll compute a 2D layout once for plotting
[coordsX, coordsY] = layoutGraph(A);

%% 7. Training Loop
figure('Name','GCN Training Progress');
for ep = 1:epochs
    % Forward pass
    % Layer 1
    H1 = A_norm * (X * W1);
    H1 = max(H1, 0);  % ReLU

    % Layer 2
    H2 = A_norm * (H1 * W2);
    Y_pred = softmaxRows(H2);
    
    % Cross-entropy loss
    L = crossEntropyLoss(Y_pred(train_mask,:), labels(train_mask), numClasses);
    
    % Accuracy
    [~, pred_labels] = max(Y_pred, [], 2);
    acc = mean(pred_labels(train_mask) == labels(train_mask)) * 100;
    
    % Backprop
    dH2 = Y_pred;
    idxTrain = find(train_mask);
    for i = 1:length(idxTrain)
        node_i = idxTrain(i);
        true_c = labels(node_i);
        dH2(node_i,true_c) = dH2(node_i,true_c) - 1;
    end
    dH2 = dH2 / numNodes;  % scale
    
    % H2 = A_norm * (H1 * W2)
    dZ2 = A_norm' * dH2;    % A_norm is symmetric, so A_norm' = A_norm
    gradW2 = H1' * dZ2;     % W2 shape: (d_hidden x numClasses)
    
    dH1 = dZ2 * W2';        % shape: (numNodes x d_hidden)
    mask = (H1 > 0);
    dH1(~mask) = 0;
    dZ1 = A_norm' * dH1;
    gradW1 = X' * dZ1;
    
    % Update
    W2 = W2 - learningRate * gradW2;
    W1 = W1 - learningRate * gradW1;
    
    % Plot every 50 epochs (or final)
    if mod(ep,50)==0 || ep==epochs
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

fprintf('Final training accuracy: %.2f%%\n', acc);

% save_all_figs_OPTION('results/gcnn2','png',1)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Local Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function A = loadKarateClubAdjacency()
% loadKarateClubAdjacency returns adjacency matrix for the 34-node Zachary's Karate Club.
% Hardcoded edges. (Same as in the previous examples.)
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
