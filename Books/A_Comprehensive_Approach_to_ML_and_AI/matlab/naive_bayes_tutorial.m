close all; clear all; clc; 

% Generate synthetic 2D data for two classes.
X = [randn(50,2); randn(50,2)+3];
y = [ones(50,1); 2*ones(50,1)];

model = myNaiveBayes(X, y);

% Classify a new point.
newPoint = [0, 0];
predictedClass = classifyNaiveBayes(model, newPoint);

% Visualize decision boundaries for 2D data.
x1range = linspace(min(X(:,1))-1, max(X(:,1))+1, 100);
x2range = linspace(min(X(:,2))-1, max(X(:,2))+1, 100);
[X1, X2] = meshgrid(x1range, x2range);
gridPoints = [X1(:), X2(:)];

predictions = arrayfun(@(i) classifyNaiveBayes(model, gridPoints(i,:)), 1:size(gridPoints,1));
Z = reshape(predictions, size(X1));

% Plots
figure;
scatter(X(y==1,1), X(y==1,2), 50, 'bo');
hold on;
scatter(X(y==2,1), X(y==2,2), 50, 'bo');
title('Naive Bayes Classification');
xlabel('Feature 1'); ylabel('Feature 2');
hold off;
legend('Clusters','Location','SE');

% save_all_figs_OPTION('results/naive_bayes1','png',1)

% Plots
figure;
scatter(X(y==1,1), X(y==1,2), 50, 'ro');
hold on;
scatter(X(y==2,1), X(y==2,2), 50, 'bo');
% contourf(X1, X2, Z, 'k', 'LineWidth', 1, 'FaceAlpha', 0.1); % Decision boundary
contourf(X1, X2, Z, 'LineWidth', 0.8, 'FaceAlpha', 0.1); % Decision boundary
colormap([1 0.8 0.8; 0.8 0.8 1]);
title('Naive Bayes Classification - Decision Boundaries');
xlabel('Feature 1'); ylabel('Feature 2');
hold off;
legend('Class 0','Class 1','Location','SE');

% save_all_figs_OPTION('results/naive_bayes2','png',1)


function model = myNaiveBayes(trainData, trainLabels)
% myNaiveBayes Trains a Naive Bayes classifier.
%   model = myNaiveBayes(trainData, trainLabels) returns a model structure
%   containing the prior probabilities and the mean and variance estimates for
%   each feature for each class.
%
%   Inputs:
%       trainData   - An N x d matrix where each row is a data instance.
%       trainLabels - An N x 1 vector of class labels.
%
%   Output:
%       model - A structure containing:
%           .classes     - Unique class labels.
%           .priors      - Prior probabilities for each class.
%           .means       - A cell array of mean vectors for each class.
%           .variances   - A cell array of variance vectors for each class.

    classes = unique(trainLabels);
    numClasses = length(classes);
    [N, d] = size(trainData);
    
    priors = zeros(numClasses, 1);
    means = cell(numClasses, 1);
    variances = cell(numClasses, 1);
    
    for k = 1:numClasses
        idx = (trainLabels == classes(k));
        priors(k) = sum(idx) / N;
        means{k} = mean(trainData(idx, :), 1);
        variances{k} = var(trainData(idx, :), 1); % Using 1/N normalization
    end
    
    model.classes = classes;
    model.priors = priors;
    model.means = means;
    model.variances = variances;
end

function predictedClass = classifyNaiveBayes(model, x)
% classifyNaiveBayes Classifies a new instance using a trained Naive Bayes model.
%   predictedClass = classifyNaiveBayes(model, x) returns the predicted class label
%   for the feature vector x.
%
%   Inputs:
%       model - A structure containing the trained model parameters.
%       x     - A 1 x d feature vector.
%
%   Output:
%       predictedClass - The predicted class label.

    numClasses = length(model.classes);
    scores = zeros(numClasses, 1);
    
    for k = 1:numClasses
        mu = model.means{k};
        sigma2 = model.variances{k};
        % Compute Gaussian likelihood for each feature (assuming independence)
        likelihood = prod((1./sqrt(2*pi*sigma2)) .* exp(-((x - mu).^2)./(2*sigma2)));
        scores(k) = log(model.priors(k)) + log(likelihood);
    end
    
    [~, idx] = max(scores);
    predictedClass = model.classes(idx);
end


