% Feature Engineering Pipeline Tutorial in MATLAB
%
% This tutorial demonstrates a complete feature engineering pipeline for a
% binary classification problem. The pipeline includes:
%
%   1. Data Cleaning: Handling missing values (via mean imputation) and
%      removing anomalies (samples with feature values beyond a threshold).
%
%   2. Feature Transformation: Normalizing the data using z-score scaling.
%
%   3. Feature Extraction: Applying Principal Component Analysis (PCA) to extract
%      principal components.
%
%   4. Feature Derivation: Creating polynomial features (degree 2) by adding
%      squared terms and pairwise products.
%
% The synthetic dataset is generated from two Gaussian distributions in 2D,
% representing two classes. Missing values and anomalies are introduced
% intentionally to demonstrate the cleaning process.


clear; clc; close all; rng(1);

%% 1. Generate Synthetic Dataset
N = 200; % total number of samples
% Generate features from two Gaussian distributions for two classes.
% Class 0: centered at (1,1)
X_class0 = randn(2, N/2) * 0.5 + repmat([1; 1], 1, N/2);
% Class 1: centered at (3,3)
X_class1 = randn(2, N/2) * 0.5 + repmat([3; 3], 1, N/2);
X = [X_class0, X_class1];  % 2 x N data matrix
y = [zeros(1, N/2), ones(1, N/2)];  % binary labels: 0 for Class 0, 1 for Class 1

% Introduce missing values: randomly set 5\% of entries in X to NaN.
numMissing = round(0.05 * numel(X));
missingIndices = randperm(numel(X), numMissing);
X(missingIndices) = NaN;

% Introduce anomalies: randomly select 5 samples and multiply feature 1 by 10.
anomalyIndices = randperm(N, 5);
X(1, anomalyIndices) = X(1, anomalyIndices) * 10;

%% 2. Data Cleaning
% Impute missing values by replacing NaN in each column with the column mean.
X_clean = imputeMissing(X);

% Remove anomalies: remove samples with any feature value beyond 3 standard deviations.
[X_clean, y_clean] = removeOutliers(X_clean, y, 2);

%% 3. Feature Transformation
% Apply z-score normalization to the cleaned data.
[X_norm, mu_X, sigma_X] = zscoreNormalize(X_clean);

%% 4. Feature Extraction
% Apply PCA to the normalized data.
% Note: MATLAB's pca function expects observations in rows.
[coeff, score, latent] = pca(X_norm');  
% For illustration, select the first 2 principal components.
X_pca = score(:, 1:2)';  % now 2 x (number of samples)

%% 5. Feature Derivation
% Create polynomial features (degree 2) from the normalized data.
X_poly = createPolynomialFeatures(X_norm, 2); % returns new features (d_new x N)

%% 6. Visualization
figure;
subplot(1,2,1);
    scatter(X(1,:), X(2,:), 40, y, 'filled');
    xlabel('Feature 1'); ylabel('Feature 2');
    title('Original Data');
subplot(1,2,2);
    scatter(X_clean(1,:), X_clean(2,:), 40, y_clean, 'filled');
    xlabel('Feature 1'); ylabel('Feature 2');
    title('Cleaned Data');
    fh = gcf;
    fh.Position = [850 950 850 350];
    % save_all_figs_OPTION('results/fetEng1','png',1)

figure;
subplot(1,2,1);
    scatter(X_pca(1,:), X_pca(2,:), 40, y_clean, 'filled');
    xlabel('PC1'); ylabel('PC2');
    title('PCA Extracted Features');
subplot(1,2,2);
    % For visualization, plot the first two polynomial features.
    scatter(X_poly(1,:), X_poly(2,:), 40, y_clean, 'filled');
    xlabel('Poly Feature 1'); ylabel('Poly Feature 2');
    title('Polynomial Derived Features');
    fh = gcf;
    fh.Position = [850 500 850 350];
    % save_all_figs_OPTION('results/fetEng2','png',1)




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Local Function Definitions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function X_imputed = imputeMissing(X)
    % imputeMissing replaces missing values (NaN) in each column of X with the column mean.
    X_imputed = X;
    [nRows, nCols] = size(X);
    for j = 1:nCols
        col = X(:, j);
        missing = isnan(col);
        if any(missing)
            colMean = mean(col(~missing));
            col(missing) = colMean;
            X_imputed(:, j) = col;
        end
    end
end

function [X_clean, y_clean] = removeOutliers(X, y, threshold)
    % removeOutliers removes samples with any feature value that is beyond the
    % threshold (in terms of standard deviations from the mean).
    % X: features (d x N), y: 1 x N labels.
    [d, N] = size(X);
    X_clean = [];
    y_clean = [];
    % For each sample, check if any feature is more than threshold standard deviations from its column mean.
    mu = mean(X, 2);
    sigma = std(X, 0, 2);
    for i = 1:N
        sample = X(:, i);
        if all(abs((sample - mu) ./ sigma) < threshold)
            X_clean = [X_clean, sample];
            y_clean = [y_clean, y(i)];
        end
    end
end

function [X_norm, mu_X, sigma_X] = zscoreNormalize(X)
    % zscoreNormalize performs z-score normalization on data X (d x N).
    mu_X = mean(X, 2);
    sigma_X = std(X, 0, 2);
    X_norm = (X - repmat(mu_X, 1, size(X,2))) ./ repmat(sigma_X, 1, size(X,2));
end

function X_poly = createPolynomialFeatures(X, degree)
    % createPolynomialFeatures creates polynomial features up to the specified degree.
    % X: original features (d x N), degree: maximum degree (currently supports degree 2)
    % Returns X_poly: new feature matrix including original and polynomial features.
    [d, N] = size(X);
    X_poly = X;  % Start with original features.
    if degree >= 2
        % Add square terms.
        for i = 1:d
            X_poly = [X_poly; X(i,:).^2];
        end
        % Add pairwise interaction terms.
        for i = 1:d
            for j = i+1:d
                X_poly = [X_poly; X(i,:) .* X(j,:)];
            end
        end
    end
    % Higher degree terms can be added similarly.
end
