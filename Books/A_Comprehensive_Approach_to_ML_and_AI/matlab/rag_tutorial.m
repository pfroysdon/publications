% Retrieval-Augmented Generation (RAG) Tutorial in MATLAB
%
% This tutorial demonstrates a simple version of the RAG algorithm.
%
% Steps:
%   1. Build a small text corpus (documents).
%   2. Construct a vocabulary from the corpus.
%   3. Given a query, convert the query and each document to bag-of-words vectors.
%   4. Compute cosine similarity between the query vector and each document vector.
%   5. Retrieve the top-k most relevant documents.
%   6. Generate a response by combining the query and the retrieved documents.
%
% All functions (vocabulary building, text-to-vector conversion, cosine similarity,
% retrieval, and generation) are implemented from scratch.

clear; clc; close all; rng(1);

%% 1. Define a Small Corpus
corpus = { ...
    'The cat sat on the mat.' , ...
    'Dogs are loyal and friendly.' , ...
    'The weather today is sunny and bright.' , ...
    'Artificial intelligence is transforming the world.' , ...
    'MATLAB is a powerful tool for engineering and data analysis.' , ...
    'Machine learning techniques can solve complex problems.' };

% Number of documents in the corpus
numDocs = length(corpus);

%% 2. Build Vocabulary from the Corpus
vocab = buildVocabulary(corpus);
fprintf('Vocabulary (%d words):\n', length(vocab));
disp(vocab');

%% 3. Define a Query and Convert to Vector
query = 'What are the benefits of machine learning?';
queryVec = textToVector(query, vocab);

%% 4. Retrieve Top-k Documents
k = 2;
[retrievedIdx, similarities] = retrieveDocuments(query, corpus, vocab, k);
fprintf('Top %d retrieved documents (indices):\n', k);
disp(retrievedIdx);
fprintf('Similarity scores:\n');
disp(similarities);

retrievedDocs = corpus(retrievedIdx);

%% 5. Generate a Response Based on the Retrieved Documents
response = generateResponse(query, retrievedDocs);
fprintf('\nGenerated Response:\n%s\n', response);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Local Function Definitions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function vocab = buildVocabulary(corpus)
    % buildVocabulary constructs a vocabulary (cell array of unique words)
    % from a cell array of text documents.
    words = {};
    for i = 1:length(corpus)
        % Convert to lowercase and remove punctuation
        doc = lower(corpus{i});
        doc = regexprep(doc, '[^\w\s]', '');
        w = strsplit(doc);
        words = [words, w];  %#ok<AGROW>
    end
    vocab = unique(words);
end

function vec = textToVector(text, vocab)
    % textToVector converts a text string into a bag-of-words vector based on the vocabulary.
    % The vector is of length equal to the number of vocabulary words, containing counts.
    text = lower(text);
    text = regexprep(text, '[^\w\s]', '');
    words = strsplit(text);
    vec = zeros(length(vocab), 1);
    for i = 1:length(vocab)
        vec(i) = sum(strcmp(vocab{i}, words));
    end
end

function sim = cosineSimilarity(v1, v2)
    % cosineSimilarity computes the cosine similarity between two vectors.
    if norm(v1)==0 || norm(v2)==0
        sim = 0;
    else
        sim = dot(v1, v2) / (norm(v1) * norm(v2));
    end
end

function [topIdx, topSims] = retrieveDocuments(query, corpus, vocab, k)
    % retrieveDocuments retrieves the indices of the top k documents from the corpus
    % that are most similar to the query, based on cosine similarity of bag-of-words vectors.
    queryVec = textToVector(query, vocab);
    sims = zeros(length(corpus), 1);
    for i = 1:length(corpus)
        docVec = textToVector(corpus{i}, vocab);
        sims(i) = cosineSimilarity(queryVec, docVec);
    end
    [sortedSims, sortedIdx] = sort(sims, 'descend');
    topIdx = sortedIdx(1:min(k, length(corpus)));
    topSims = sortedSims(1:min(k, length(corpus)));
end

function response = generateResponse(query, retrievedDocs)
    % generateResponse generates a simple response by combining the query
    % with the retrieved documents.
    response = ['Your query was: "', query, '". '];
    if isempty(retrievedDocs)
        response = [response, 'No relevant documents were found.'];
    else
        response = [response, 'I found the following relevant information: '];
        for i = 1:length(retrievedDocs)
            response = [response, ' [', retrievedDocs{i}, ']'];
        end
    end
end
