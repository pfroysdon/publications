% This script demonstrates a recurrent neural network for sentence
% completion using "Alice's Adventures in Wonderland" as the dataset.
%
% The text file "alice-in-wonderland.txt" must be in your MATLAB path.
%
% Steps:
%   1. Load and preprocess the text.
%   2. Tokenize the text and build a vocabulary.
%   3. Create training examples: for each sequence of three words, the
%      target is the next (fourth) word.
%   4. Split the examples into training and validation sets.
%   5. Define and train an LSTM network for next-word prediction.
%   6. At inference, given three words, the network predicts the next word.
%
% Run the script to train the network and then see sample predictions.

%% 1. Load and Preprocess Text
text = fileread('data/alice-in-wonderland.txt');
text = lower(text); % convert to lower case
% Remove punctuation: keep only letters, numbers, and whitespace.
text = regexprep(text, '[^a-z0-9\s]', '');
% Tokenize: split on whitespace.
tokens = strsplit(text);
% Remove empty tokens.
tokens = tokens(~cellfun('isempty', tokens));

%% 2. Build Vocabulary and Create Mappings
vocab = unique(tokens);
vocabSize = numel(vocab);
% Create mappings: word -> index and index -> word.
word2idx = containers.Map(vocab, 1:vocabSize);
idx2word = containers.Map(1:vocabSize, vocab);

%% 3. Create Training Sequences
% Create training examples using a sliding window:
% For every 3 consecutive words, the target is the next (4th) word.
sequenceLength = 3;
numTokens = numel(tokens);
numSequences = numTokens - sequenceLength;
X = cell(numSequences, 1);
Y = cell(numSequences, 1);
for i = 1:numSequences
    % Input: three words
    seq = tokens(i:i+sequenceLength-1);
    seqIdx = zeros(1, sequenceLength);
    for j = 1:sequenceLength
        seqIdx(j) = word2idx(seq{j});
    end
    X{i} = seqIdx;
    % Target: the word immediately after the three-word sequence.
    targetWord = tokens{i+sequenceLength};
    % Store target as numeric index.
    Y{i} = word2idx(targetWord);
end

%% 4. Split Data into Training and Validation Sets
rng(1); % For reproducibility
numExamples = numel(X);
shuffledIdx = randperm(numExamples);
trainRatio = 0.9;
numTrain = round(trainRatio * numExamples);
trainIdx = shuffledIdx(1:numTrain);
valIdx = shuffledIdx(numTrain+1:end);
XTrain = X(trainIdx);
% Convert the numeric labels to categorical using a fixed value set.
YTrain = categorical([Y{trainIdx}], 1:vocabSize)';
XVal = X(valIdx);
YVal = categorical([Y{valIdx}], 1:vocabSize)';

%% 5. Define and Train the LSTM Network
% The network architecture:
%   - sequenceInputLayer: accepts scalar inputs (word indices).
%   - wordEmbeddingLayer: learns an embedding for each word.
%   - lstmLayer: processes the sequence (output mode 'last').
%   - fullyConnectedLayer: maps to vocabSize classes.
%   - softmaxLayer and classificationLayer.
embeddingDimension = 50;
numHiddenUnits = 100;

layers = [ ...
    sequenceInputLayer(1, 'Name', 'input')
    wordEmbeddingLayer(embeddingDimension, vocabSize, 'Name', 'embed')
    lstmLayer(numHiddenUnits, 'OutputMode', 'last', 'Name', 'lstm')
    fullyConnectedLayer(vocabSize, 'Name', 'fc')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classoutput')];

options = trainingOptions('adam', ...
    'MaxEpochs', 10, ...               % Increase epochs for improved performance
    'MiniBatchSize', 128, ...
    'ValidationData', {XVal, YVal}, ...
    'ValidationFrequency', 1000, ...
    'Verbose', 1, ...
    'Plots', 'training-progress');

net = trainNetwork(XTrain, YTrain, layers, options);

%% 6. Inference: Complete a Prompt with the Next Word
% The function completeSentence (defined at the end) takes a three-word
% prompt and returns the predicted next word.
disp('Example Predictions:');
prompt1 = 'alice was beginning';
next1 = completeSentence(prompt1, net, word2idx, idx2word);
disp(['Prompt: "', prompt1, '" --> Next word: ', next1]);

prompt2 = 'today is very';
next2 = completeSentence(prompt2, net, word2idx, idx2word);
disp(['Prompt: "', prompt2, '" --> Next word: ', next2]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Local Function: completeSentence
function nextWord = completeSentence(prompt, net, word2idx, idx2word)
% completeSentence Given a three-word prompt, predict the next word.
%   nextWord = completeSentence(prompt, net, word2idx, idx2word)
%
% Inputs:
%   prompt  - a string containing exactly three words.
%   net     - the trained network.
%   word2idx- a containers.Map mapping words to indices.
%   idx2word- a containers.Map mapping indices to words.
%
% Output:
%   nextWord- the predicted next word (as a char array).

    prompt = lower(prompt);
    prompt = regexprep(prompt, '[^a-z0-9\s]', '');
    words = strsplit(prompt);
    if numel(words) ~= 3
        error('Prompt must contain exactly three words.');
    end
    seq = zeros(1,3);
    for j = 1:3
        if isKey(word2idx, words{j})
            seq(j) = word2idx(words{j});
        else
            % If word not in vocabulary, assign a default index (here, 1).
            seq(j) = 1;
        end
    end
    % The network expects the input as a cell array of row vectors.
    pred = classify(net, {seq});
    predIdx = double(pred);
    nextWord = idx2word(predIdx);
end
