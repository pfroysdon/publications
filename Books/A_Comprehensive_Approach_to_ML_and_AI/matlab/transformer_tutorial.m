% Below is a complete MATLAB tutorial that implements a simplified—but 
% fully functional—Transformer network from scratch. This example builds 
% both encoder and decoder blocks with multi‑head self‑attention, 
% position‑wise feed‑forward networks, residual connections, and layer 
% normalization. (For clarity, we omit dropout and advanced training 
% routines, focusing instead on the forward pass.)
% 
% In this tutorial, we:
% 1. Create dummy input sequences for both the encoder and decoder.
% 2. Add sinusoidal positional encodings.
% 3. Define functions for:
%     Scaled dot‑product attention and multi‑head attention.
%     A two‑layer feed‑forward network.
%     Layer normalization.
% 4. Build an encoder block (self‑attention followed by feed‑forward, with 
%    residual connections).
% 5. Build a decoder block (masked self‑attention, encoder‑decoder 
%    attention, and feed‑forward, each with residual connections).
% 6. Assemble a small Transformer with two encoder layers and two decoder 
%    layers, then run a forward pass.
% 
% Explanation:
% Positional Encoding:
% The positionalEncoding function computes a sinusoidal encoding for each 
% token position and is added to the input embeddings.
% 
% Attention Functions:
% scaledDotProductAttention computes the dot‑product attention (optionally 
% using a mask for the decoder). multiHeadAttention splits inputs into 
% multiple heads, computes attention for each, concatenates, and applies a 
% final projection.
% 
% Feed‑Forward and Layer Norm:
% The positionwiseFeedForward and layerNorm functions implement the feed‑
% forward network and normalization used in each Transformer block.
% 
% Encoder and Decoder Blocks:
% The encoderBlock applies multi‑head self‑attention followed by a feed‑
% forward network (each with residual connections and normalization).
% The decoderBlock first performs masked self‑attention, then attends over 
% the encoder’s output, and finally applies its feed‑forward network.
% 
% Transformer Assembly:
% The main function creates dummy encoder and decoder inputs, adds 
% positional encodings, builds a stack of two encoder layers and two 
% decoder layers, and performs a forward pass. Finally, it displays the 
% first three tokens of the decoder output.
% 
% This self‑contained MATLAB tutorial demonstrates how to build a 
% simplified Transformer network—including both encoder and decoder 
% blocks—from scratch. You can extend it further by incorporating a 
% training loop with backpropagation, dropout, and additional layers to 
% tackle real sequence‐to‐sequence tasks.

function transformerNetworkTutorial
    % TRANSFORMERNETWORKTUTORIAL
    %
    % This tutorial implements a simplified Transformer network (encoder–decoder)
    % from scratch, including multi-head self-attention, feed-forward networks,
    % positional encodings, layer normalization, and residual connections.
    %
    % We simulate a forward pass using dummy input sequences.
    
    clc; clear; close all; rng(1);
    
    %% Hyperparameters
    seq_len_enc = 10;    % Encoder sequence length
    seq_len_dec = 10;    % Decoder sequence length
    d_model = 32;        % Model (embedding) dimension
    num_heads = 4;       % Number of attention heads (must divide d_model)
    d_ff = 64;           % Feed-forward inner-layer dimension
    num_enc_layers = 2;  % Number of encoder layers
    num_dec_layers = 2;  % Number of decoder layers
    epsilon = 1e-6;      % For layer normalization
    
    %% Dummy Input Sequences
    % In practice these would be token embeddings.
    encoder_input = randn(seq_len_enc, d_model);
    decoder_input = randn(seq_len_dec, d_model);
    
    % Add positional encodings
    encoder_input = encoder_input + positionalEncoding(seq_len_enc, d_model);
    decoder_input = decoder_input + positionalEncoding(seq_len_dec, d_model);
    
    %% Initialize Encoder Layer Weights
    % For each encoder layer, we need weights for multi-head attention and feed-forward.
    enc_layers = cell(num_enc_layers,1);
    for i = 1:num_enc_layers
        % Multi-head self-attention weights (all dimensions: d_model x d_model)
        enc_layers{i}.Wq = randn(d_model, d_model);
        enc_layers{i}.Wk = randn(d_model, d_model);
        enc_layers{i}.Wv = randn(d_model, d_model);
        enc_layers{i}.Wo = randn(d_model, d_model);
        % Feed-forward network weights
        enc_layers{i}.W1 = randn(d_model, d_ff);
        enc_layers{i}.b1 = randn(1, d_ff);
        enc_layers{i}.W2 = randn(d_ff, d_model);
        enc_layers{i}.b2 = randn(1, d_model);
    end
    
    %% Initialize Decoder Layer Weights
    dec_layers = cell(num_dec_layers,1);
    for i = 1:num_dec_layers
        % Masked self-attention (decoder) weights
        dec_layers{i}.Wq_self = randn(d_model, d_model);
        dec_layers{i}.Wk_self = randn(d_model, d_model);
        dec_layers{i}.Wv_self = randn(d_model, d_model);
        dec_layers{i}.Wo_self = randn(d_model, d_model);
        % Encoder-decoder attention weights
        dec_layers{i}.Wq_encdec = randn(d_model, d_model);
        dec_layers{i}.Wk_encdec = randn(d_model, d_model);
        dec_layers{i}.Wv_encdec = randn(d_model, d_model);
        dec_layers{i}.Wo_encdec = randn(d_model, d_model);
        % Feed-forward network weights
        dec_layers{i}.W1 = randn(d_model, d_ff);
        dec_layers{i}.b1 = randn(1, d_ff);
        dec_layers{i}.W2 = randn(d_ff, d_model);
        dec_layers{i}.b2 = randn(1, d_model);
    end
    
    %% Encoder Forward Pass
    enc_output = encoder_input;
    for i = 1:num_enc_layers
        enc_output = encoderBlock(enc_output, num_heads, d_model, d_ff, ...
                        enc_layers{i}.Wq, enc_layers{i}.Wk, enc_layers{i}.Wv, enc_layers{i}.Wo, ...
                        enc_layers{i}.W1, enc_layers{i}.b1, enc_layers{i}.W2, enc_layers{i}.b2, epsilon);
    end
    
    %% Create Mask for Decoder Self-Attention
    % Create a lower-triangular mask to prevent attending to future tokens.
    mask = triu(ones(seq_len_dec, seq_len_dec), 1);
    mask(mask==1) = -Inf;  % Masked positions set to -Inf
    mask(mask==0) = 0;
    
    %% Decoder Forward Pass
    dec_output = decoder_input;
    for i = 1:num_dec_layers
        dec_output = decoderBlock(dec_output, enc_output, num_heads, d_model, d_ff, ...
                        dec_layers{i}.Wq_self, dec_layers{i}.Wk_self, dec_layers{i}.Wv_self, dec_layers{i}.Wo_self, ...
                        dec_layers{i}.Wq_encdec, dec_layers{i}.Wk_encdec, dec_layers{i}.Wv_encdec, dec_layers{i}.Wo_encdec, ...
                        dec_layers{i}.W1, dec_layers{i}.b1, dec_layers{i}.W2, dec_layers{i}.b2, epsilon, mask);
    end
    
    %% Final Output
    % The final decoder output would then be projected to a vocabulary (not shown).
    fprintf('Transformer Network Output (first 3 tokens of decoder):\n');
    disp(dec_output(1:3, :));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                   Positional Encoding Function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function posEnc = positionalEncoding(seq_len, d_model)
    % POSITIONALENCODING computes sinusoidal positional encodings.
    % posEnc: (seq_len x d_model)
    posEnc = zeros(seq_len, d_model);
    for pos = 1:seq_len
        for i = 1:d_model
            angle = pos / (10000^((i-1)/d_model));
            if mod(i,2)==1  % odd indices: sine
                posEnc(pos,i) = sin(angle);
            else            % even indices: cosine
                posEnc(pos,i) = cos(angle);
            end
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                Scaled Dot-Product Attention Function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output, attn_weights] = scaledDotProductAttention(Q, K, V, mask)
    % SCALED_DOT_PRODUCT_ATTENTION computes attention as:
    %   Attention(Q,K,V) = softmax((Q*K'/sqrt(d_k)) + mask) * V
    d_k = size(K,2);
    scores = Q * K' / sqrt(d_k);  % (seq_len x seq_len)
    if nargin == 4 && ~isempty(mask)
        scores = scores + mask;
    end
    attn_weights = softmax(scores, 2);
    output = attn_weights * V;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                       Multi-Head Attention Function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function output = multiHeadAttention(Q, K, V, num_heads, Wq, Wk, Wv, Wo, mask)
    % MULTI_HEAD_ATTENTION computes multi-head attention.
    % Q, K, V: (seq_len x d_model)
    [seq_len, d_model] = size(Q);
    d_k = d_model / num_heads;
    
    % Linear projections
    Q_proj = Q * Wq;
    K_proj = K * Wk;
    V_proj = V * Wv;
    
    % Reshape into (seq_len x num_heads x d_k)
    Q_reshaped = reshape(Q_proj, seq_len, num_heads, d_k);
    K_reshaped = reshape(K_proj, seq_len, num_heads, d_k);
    V_reshaped = reshape(V_proj, seq_len, num_heads, d_k);
    
    head_outputs = zeros(seq_len, num_heads, d_k);
    for h = 1:num_heads
        Q_h = squeeze(Q_reshaped(:,h,:));
        K_h = squeeze(K_reshaped(:,h,:));
        V_h = squeeze(V_reshaped(:,h,:));
        [head_out, ~] = scaledDotProductAttention(Q_h, K_h, V_h, mask);
        head_outputs(:,h,:) = head_out;
    end
    
    % Concatenate heads: reshape to (seq_len x d_model)
    concatenated = reshape(head_outputs, seq_len, d_model);
    % Final linear projection
    output = concatenated * Wo;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                    Position-wise Feed-Forward Network
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function output = positionwiseFeedForward(X, W1, b1, W2, b2)
    % POSITIONWISE_FEED_FORWARD applies a two-layer feed-forward network to X.
    hidden = max(0, X * W1 + repmat(b1, size(X,1), 1));  % ReLU activation
    output = hidden * W2 + repmat(b2, size(X,1), 1);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                          Layer Normalization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function output = layerNorm(X, epsilon)
    % LAYERNORM applies layer normalization to X (row-wise normalization).
    mu = mean(X, 2);
    sigma = std(X, 0, 2);
    output = (X - repmat(mu, 1, size(X,2))) ./ (repmat(sigma, 1, size(X,2)) + epsilon);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                           Encoder Block
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function output = encoderBlock(X, num_heads, d_model, d_ff, Wq, Wk, Wv, Wo, W1, b1, W2, b2, epsilon)
    % ENCODERBLOCK applies a Transformer encoder block:
    %   1. Multi-head self-attention with residual connection and layer norm.
    %   2. Position-wise feed-forward network with residual connection and layer norm.
    
    % Multi-head self-attention
    attn_out = multiHeadAttention(X, X, X, num_heads, Wq, Wk, Wv, Wo, []);
    out1 = layerNorm(X + attn_out, epsilon);
    
    % Feed-forward network
    ff_out = positionwiseFeedForward(out1, W1, b1, W2, b2);
    output = layerNorm(out1 + ff_out, epsilon);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                           Decoder Block
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function output = decoderBlock(X, encoderOutput, num_heads, d_model, d_ff, ...
                        Wq_self, Wk_self, Wv_self, Wo_self, ...
                        Wq_encdec, Wk_encdec, Wv_encdec, Wo_encdec, ...
                        W1, b1, W2, b2, epsilon, mask)
    % DECODERBLOCK applies a Transformer decoder block:
    %   1. Masked multi-head self-attention (with look-ahead mask).
    %   2. Encoder-decoder attention.
    %   3. Position-wise feed-forward network.
    
    % Masked self-attention on decoder input
    self_attn = multiHeadAttention(X, X, X, num_heads, Wq_self, Wk_self, Wv_self, Wo_self, mask);
    out1 = layerNorm(X + self_attn, epsilon);
    
    % Encoder-decoder attention: query from out1, keys/values from encoder output.
    encdec_attn = multiHeadAttention(out1, encoderOutput, encoderOutput, num_heads, Wq_encdec, Wk_encdec, Wv_encdec, Wo_encdec, []);
    out2 = layerNorm(out1 + encdec_attn, epsilon);
    
    % Feed-forward network
    ff_out = positionwiseFeedForward(out2, W1, b1, W2, b2);
    output = layerNorm(out2 + ff_out, epsilon);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                             Softmax Function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function S = softmax(X, dim)
    if nargin < 2, dim = 1; end
    X_max = max(X, [], dim);
    X_exp = exp(X - X_max);
    S = X_exp ./ sum(X_exp, dim);
end
