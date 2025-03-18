close all; clear all; clc; 

% Suppose we have a very simple corpus with three documents and a vocabulary size of 15.
% Each document is represented as an array of word indices.
docs = { [1 5 3 2 7 5], [4 2 5 6 8 2 5], [7 8 2 3 1 4], [3 6 1 7 4 2 9] };
[theta, phi, Z] = myLDA(docs, 15, 4, 0.1, 0.01, 100);
% Plot the topic distribution for the first document:
figure;
bar(theta(1,:));
title('Topic Distribution for Document 1');
xlabel('Topic'); ylabel('Probability');

% save_all_figs_OPTION('results/latent_dirichlet_allocation','png',1)

function [theta, phi, Z] = myLDA(docs, V, K, alpha, beta, T)
% myLDA Perform Latent Dirichlet Allocation using collapsed Gibbs sampling.
%   [theta, phi, Z] = myLDA(docs, V, K, alpha, beta, T) takes as input:
%       docs  - A cell array of documents, each document is an array of word indices.
%       V     - Vocabulary size.
%       K     - Number of topics.
%       alpha - Hyperparameter for document-topic distribution.
%       beta  - Hyperparameter for topic-word distribution.
%       T     - Number of Gibbs sampling iterations.
%
%   The function outputs:
%       theta - Document-topic distribution (M x K matrix).
%       phi   - Topic-word distribution (K x V matrix).
%       Z     - Cell array of topic assignments for each document.
%
% Example usage:
%   docs = { [1 2 3 4 5], [2 3 3 4 6], [1 4 5 6 7] };
%   V = 10; K = 2; alpha = 0.1; beta = 0.01; T = 100;
%   [theta, phi, Z] = myLDA(docs, V, K, alpha, beta, T);
%   % Visualize topic distribution for the first document:
%   figure;
%   bar(theta(1,:));
%   title('Document 1 Topic Distribution');
%   xlabel('Topic'); ylabel('Probability');

    M = length(docs);
    % Initialize count matrices
    ndk = zeros(M, K);         % document-topic counts
    nkw = zeros(K, V);         % topic-word counts
    nk = zeros(K, 1);          % topic counts
    
    Z = cell(M, 1);           % topic assignments for each document

    % Randomly initialize topic assignments for each word in each document
    for d = 1:M
        N_d = length(docs{d});
        Z{d} = zeros(1, N_d);
        for n = 1:N_d
            w = docs{d}(n);
            topic = randi(K);
            Z{d}(n) = topic;
            ndk(d, topic) = ndk(d, topic) + 1;
            nkw(topic, w) = nkw(topic, w) + 1;
            nk(topic) = nk(topic) + 1;
        end
    end

    % Gibbs sampling iterations
    for t = 1:T
        for d = 1:M
            N_d = length(docs{d});
            for n = 1:N_d
                w = docs{d}(n);
                topic = Z{d}(n);
                
                % Remove current assignment
                ndk(d, topic) = ndk(d, topic) - 1;
                nkw(topic, w) = nkw(topic, w) - 1;
                nk(topic) = nk(topic) - 1;
                
                % Compute conditional distribution
                p = zeros(1, K);
                for k = 1:K
                    p(k) = (ndk(d, k) + alpha) * ...
                           (nkw(k, w) + beta) / (nk(k) + V * beta);
                end
                p = p / sum(p);
                
                % Sample new topic
                new_topic = find(mnrnd(1, p) == 1);
                
                % Update assignments and counts
                Z{d}(n) = new_topic;
                ndk(d, new_topic) = ndk(d, new_topic) + 1;
                nkw(new_topic, w) = nkw(new_topic, w) + 1;
                nk(new_topic) = nk(new_topic) + 1;
            end
        end
    end

    % Estimate theta (document-topic distributions)
    theta = (ndk + alpha);
    theta = theta ./ repmat(sum(theta, 2), 1, K);
    
    % Estimate phi (topic-word distributions)
    phi = (nkw + beta);
    phi = phi ./ repmat(sum(phi, 2), 1, V);
end
