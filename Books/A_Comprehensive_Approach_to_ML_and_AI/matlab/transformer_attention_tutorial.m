% This script visualizes the multi-head scaled dot-product attention
% weights from a Transformer. We generate random queries, keys, and values,
% split them into multiple heads, compute the attention weights for each head,
% and then plot the weights in subplots.
%
% The attention for each head is computed as:
%    Attention(Q, K, V) = softmax( Q*K' / sqrt(d_k) ) * V


clear; clc; close all; rng(1);

%% Parameters
seq_len = 8;         % Number of tokens in the sequence
d_model = 16;        % Model (embedding) dimension
num_heads = 4;       % Number of attention heads (d_model must be divisible by num_heads)
d_k = d_model / num_heads;  % Dimension per head

%% Generate Random Q, K, V Matrices
% Each of size (seq_len x d_model)
Q = randn(seq_len, d_model);
K = randn(seq_len, d_model);
V = randn(seq_len, d_model);

%% Compute Attention Weights for Each Head
headWeights = cell(num_heads,1);
for h = 1:num_heads
    % Extract the dimensions for head h
    idx = (h-1)*d_k + 1 : h*d_k;
    Q_h = Q(:, idx);  % (seq_len x d_k)
    K_h = K(:, idx);  % (seq_len x d_k)
    V_h = V(:, idx);  % (seq_len x d_k)
    
    % Compute scaled dot-product attention; we ignore the output here
    [~, attn] = scaledDotProductAttention(Q_h, K_h, V_h);
    
    % Save the attention weights for visualization
    headWeights{h} = attn;
end

%% Plot Attention Weights for Each Head
figure;
for h = 1:num_heads
    subplot(2,2,h);
    imagesc(headWeights{h});
    colorbar;
    title(sprintf('Head %d Attention Weights', h));
    xlabel('Key Index'); ylabel('Query Index');
end
sgtitle('Multi-Head Attention Weights Visualization');


% save_all_figs_OPTION('results/transformer_attention','png',1)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Local Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output, attn_weights] = scaledDotProductAttention(Q, K, V, mask)
% scaledDotProductAttention computes the scaled dot-product attention.
%
%   [output, attn_weights] = scaledDotProductAttention(Q, K, V, mask)
%
% Inputs:
%   Q    - Query matrix (n_q x d_k)
%   K    - Key matrix (n_k x d_k)
%   V    - Value matrix (n_k x d_v)
%   mask - (Optional) mask matrix (n_q x n_k)
%
% Outputs:
%   output       - Attention output (n_q x d_v)
%   attn_weights - Attention weights after softmax (n_q x n_k)
%
% Steps:
%   1. Compute scores = Q*K' / sqrt(d_k)
%   2. Optionally add mask to scores.
%   3. Apply softmax row-wise to obtain attention weights.
%   4. Compute output = attentionWeights * V.

    if nargin < 4
        mask = [];
    end
    [n_q, d_k] = size(Q);
    scores = Q * K' / sqrt(d_k);
    if ~isempty(mask)
        scores = scores + mask;
    end
    attn_weights = softmax(scores, 2);
    output = attn_weights * V;
end

function S = softmax(X, dim)
% softmax applies the softmax function along dimension 'dim' of X.
    if nargin < 2, dim = 1; end
    X_max = max(X, [], dim);
    X_exp = exp(X - X_max);
    S = X_exp ./ sum(X_exp, dim);
end
