% Tutorial Overview
% 
% Corpus Preprocessing:
% A small corpus of sentences is defined, converted to lowercase, 
% punctuation is removed, and the text is split into individual words.
% 
% Vocabulary Building:
% Unique words are extracted to build the vocabulary, and a mapping from 
% word to index is created.
% 
% Training Pair Creation (Skip-Gram):
% For each word in the corpus, a context window is defined (here, 2 words 
% before and after). Training pairs are generated as (center word, context 
% word).
% 
% Weight Initialization:
% The input embedding matrix W_in (size V×DV×D) and the output matrix W_out 
% (size D×VD×V) are initialized with small random values.
% 
% Model Training:
% For each training pair, the model computes the hidden representation 
% (embedding) for the center word. It then computes scores for all 
% vocabulary words, applies softmax, and computes the negative log‑
% likelihood loss for the correct context word. Gradients are computed and 
% parameters are updated using stochastic gradient descent (SGD).
% 
% Visualization:
% Finally, the learned embeddings (rows of W_in) are projected to 2D using 
% PCA and plotted with the word labels.
% 
% This self-contained tutorial provides a basic implementation of the 
% Word2Vec skip-gram model in MATLAB from scratch. Enjoy experimenting with 
% the code, changing the corpus, or extending the model (e.g., using 
% negative sampling)!

% Word2Vec Skip-Gram Model Tutorial from Scratch in MATLAB
%
% This tutorial implements a simple version of the Word2Vec skip-gram model.
% Steps:
%   1. Define a small corpus and preprocess it (lowercase, remove punctuation).
%   2. Build the vocabulary and map each word to a unique index.
%   3. Generate training pairs (center, context) using a fixed window size.
%   4. Initialize the embedding matrices (input and output weights).
%   5. Train the model using stochastic gradient descent with softmax loss.
%   6. Visualize the learned word embeddings (projected to 2D via PCA).


clear; clc; close all; rng(1);

%% Step 1: Define and Preprocess Corpus
corpus = { 'The quick brown fox jumps over the lazy dog', ...
           'I love natural language processing', ...
           'Word embeddings capture semantic similarity', ...
           'Deep learning for NLP is fascinating' };

% Convert to lowercase and remove punctuation
for i = 1:length(corpus)
    corpus{i} = lower(corpus{i});
    corpus{i} = regexprep(corpus{i}, '[^a-z\s]', '');  % keep letters and spaces
end

% Split sentences into words and combine into a single list
words = {};
for i = 1:length(corpus)
    w = strsplit(corpus{i});
    words = [words, w]; %#ok<AGROW>
end

%% Step 2: Build Vocabulary
vocab = unique(words);
V = length(vocab);  % Vocabulary size
fprintf('Vocabulary size: %d\n', V);
% Create mapping: word -> index
word2idx = containers.Map(vocab, 1:V);

%% Step 3: Generate Training Pairs (Skip-Gram Model)
% For each word in the corpus, consider a context window of size windowSize.
windowSize = 2;  % Look at 2 words before and after the center word
trainingPairs = [];  % Each row: [center, context]
for i = 1:length(words)
    center = word2idx(words{i});
    for j = max(1, i-windowSize) : min(length(words), i+windowSize)
        if j == i, continue; end
        context = word2idx(words{j});
        trainingPairs = [trainingPairs; center, context]; %#ok<AGROW>
    end
end
numPairs = size(trainingPairs,1);
fprintf('Number of training pairs: %d\n', numPairs);

%% Step 4: Initialize Weight Matrices
D = 10;  % Embedding dimension
% Input embedding matrix: V x D (each row is a word embedding)
W_in = 0.01 * randn(V, D);
% Output weight matrix: D x V
W_out = 0.01 * randn(D, V);

%% Step 5: Train the Skip-Gram Model using SGD
learningRate = 0.05;
numEpochs = 1000;
lossHistory = zeros(numEpochs, 1);

for epoch = 1:numEpochs
    totalLoss = 0;
    % Shuffle training pairs for each epoch
    idx = randperm(numPairs);
    for i = 1:numPairs
        pair = trainingPairs(idx(i), :);
        center = pair(1);
        context = pair(2);
        
        % Forward Pass:
        % Get the embedding for the center word (D x 1)
        h = W_in(center, :)';  
        % Compute scores for all words: (V x 1)
        scores = W_out' * h;
        % Compute softmax probabilities (V x 1)
        probs = softmax(scores);
        
        % Compute loss: negative log probability for the actual context word
        loss = -log(probs(context) + 1e-10);
        totalLoss = totalLoss + loss;
        
        % Backward Pass:
        % Gradient with respect to scores (V x 1)
        dscores = probs;
        dscores(context) = dscores(context) - 1;
        
        % Gradients for output weight matrix: dW_out (D x V)
        dW_out = h * dscores';
        % Gradient for hidden layer: dh (D x 1)
        dh = W_out * dscores;
        
        % Gradient for input embedding of the center word is dh.
        % Update parameters with SGD:
        W_in(center, :) = W_in(center, :) - learningRate * dh';
        W_out = W_out - learningRate * dW_out;
    end
    lossHistory(epoch) = totalLoss / numPairs;
    if mod(epoch, 100) == 0
        fprintf('Epoch %d/%d, Loss: %.4f\n', epoch, numEpochs, lossHistory(epoch));
    end
end

%% Step 6: Visualize Learned Word Embeddings
% Use PCA to project the D-dimensional embeddings to 2D for visualization.
[coeff, score] = pca(W_in);
figure;
scatter(score(:,1), score(:,2), 50, 'filled');
% Annotate points with corresponding words
for i = 1:V
    text(score(i,1)+0.05, score(i,2), vocab{i});
end
xlabel('Principal Component 1'); ylabel('Principal Component 2');
title('Learned Word Embeddings (Projected to 2D via PCA)');
grid on;

% save_all_figs_OPTION('results/word2vec','png',1)

%% Local Function: Softmax
function s = softmax(x)
% softmax computes the softmax of vector x.
    x = x - max(x);  % for numerical stability
    ex = exp(x);
    s = ex / sum(ex);
end
