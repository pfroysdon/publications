function bootstrap_vs_bagging_vs_boosting_demo
    % This demo illustrates bootstrap, bagging, and boosting using
    % decision stumps on a simple binary classification problem.
    %
    % We generate 2D synthetic data from two Gaussian clusters:
    %   - Class -1: centered at (-1, -1)
    %   - Class +1: centered at (1, 1)
    %
    % Then, we train:
    %   1. A bagging ensemble (using bootstrap samples and majority vote).
    %   2. A boosting ensemble (using AdaBoost with decision stumps).
    %
    % Finally, we plot the decision boundaries (using a grid) and report
    % the training accuracies.

    rng(1); % For reproducibility
    
    %% 1. Generate Synthetic Data
    N = 200;
    % Generate 100 points for each class
    X_class1 = randn(100,2) + [-1 -1];  % Class -1
    X_class2 = randn(100,2) + [1 1];     % Class +1
    X = [X_class1; X_class2];
    y = [-ones(100,1); ones(100,1)];     % Labels: -1 and +1
    
    %% 2. Train Ensemble Models
    T = 50;  % Number of ensemble members

    % ----- BAGGING (using bootstrap samples) -----
    bagging_models = baggingEnsembleTrain(X, y, T);
    y_pred_bagging = baggingEnsemblePredict(bagging_models, X);
    acc_bagging = mean(y_pred_bagging == y);
    fprintf('Bagging Training Accuracy: %.2f%%\n', acc_bagging*100);
    
    % ----- BOOSTING (AdaBoost with decision stumps) -----
    boosting_models = boostingEnsembleTrain(X, y, T);
    y_pred_boosting = boostingEnsemblePredict(boosting_models, X);
    acc_boosting = mean(y_pred_boosting == y);
    fprintf('Boosting Training Accuracy: %.2f%%\n', acc_boosting*100);
    
    %% 3. Plot Decision Boundaries
    % Create a grid over the feature space
    x1_range = linspace(min(X(:,1))-1, max(X(:,1))+1, 100);
    x2_range = linspace(min(X(:,2))-1, max(X(:,2))+1, 100);
    [xx, yy] = meshgrid(x1_range, x2_range);
    grid_points = [xx(:), yy(:)];
    
    % Obtain predictions on the grid from each ensemble
    preds_bagging = baggingEnsemblePredict(bagging_models, grid_points);
    preds_boosting = boostingEnsemblePredict(boosting_models, grid_points);
    
    % Reshape predictions for contour plotting
    preds_bagging = reshape(preds_bagging, size(xx));
    preds_boosting = reshape(preds_boosting, size(xx));
    
    % Plot the bagging decision boundary
    figure('Position',[100 100 1200 500]);
    subplot(1,2,1);
    % contourf(xx, yy, preds_bagging, [-1, 0, 1], 'LineColor','none');
    contourf(xx, yy, preds_bagging, [-1, 0, 1], 'LineWidth', 0.8, 'FaceAlpha', 0.1);
    colormap([1 0.8 0.8; 0.8 0.8 1]);
    hold on;
    % scatter(X(y==-1,1), X(y==-1,2), 40, 'r', 'filled');
    % scatter(X(y==1,1), X(y==1,2), 40, 'b', 'filled');
    scatter(X(y==-1,1), X(y==-1,2), 40, 'ro');
    scatter(X(y==1,1), X(y==1,2), 40, 'bo');
    title(sprintf('Bagging Decision Boundary (T = %d)', T));
    xlabel('x_1'); ylabel('x_2');
    grid on;
    hold off;
    
    % Plot the boosting decision boundary
    subplot(1,2,2);
    % contourf(xx, yy, preds_boosting, [-1, 0, 1], 'LineColor','none');
    contourf(xx, yy, preds_boosting, [-1, 0, 1], 'LineWidth', 0.8, 'FaceAlpha', 0.1);
    colormap([1 0.8 0.8; 0.8 0.8 1]);
    hold on;
    % scatter(X(y==-1,1), X(y==-1,2), 40, 'r', 'filled');
    % scatter(X(y==1,1), X(y==1,2), 40, 'b', 'filled');
    scatter(X(y==-1,1), X(y==-1,2), 40, 'ro');
    scatter(X(y==1,1), X(y==1,2), 40, 'bo');
    title(sprintf('Boosting Decision Boundary (T = %d)', T));
    xlabel('x_1'); ylabel('x_2');
    grid on;
    hold off;
    
    fprintf('\nNote:\n');
    fprintf(' - Bagging builds each base learner on a bootstrap sample and aggregates by majority vote.\n');
    fprintf(' - Boosting (AdaBoost) builds base learners sequentially with reweighted training examples,\n');
    fprintf('   emphasizing hard-to-classify points.\n');

    % save_all_figs_OPTION('results/bootstrap_vs_bagging_vs_boosting','png',1)
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Helper Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Decision Stump Training Function
function model = decisionStumpTrain(X, y, weights)
    % Trains a decision stump on (X, y) using the provided sample weights.
    % A decision stump splits on one feature with a threshold and polarity.
    %
    % Inputs:
    %   X       - n x d matrix of features.
    %   y       - n x 1 vector of labels in {-1, +1}.
    %   weights - n x 1 vector of sample weights.
    %
    % Output:
    %   model   - structure with fields: feature, threshold, polarity.
    
    [n, d] = size(X);
    best_error = inf;
    best_model = struct('feature', NaN, 'threshold', NaN, 'polarity', NaN);
    
    for j = 1:d
        feature_values = X(:, j);
        unique_vals = unique(feature_values);
        for i = 1:length(unique_vals)
            threshold = unique_vals(i);
            for polarity = [1, -1]
                predictions = ones(n,1);
                % Decision rule: if (polarity * feature) < (polarity * threshold) then predict +1; else -1.
                predictions( polarity * feature_values >= polarity * threshold ) = -1;
                % Compute weighted error
                err = sum(weights .* (predictions ~= y));
                if err < best_error
                    best_error = err;
                    best_model.feature = j;
                    best_model.threshold = threshold;
                    best_model.polarity = polarity;
                end
            end
        end
    end
    model = best_model;
end

%% Decision Stump Prediction Function
function predictions = decisionStumpPredict(model, X)
    % Predicts labels for X using a trained decision stump model.
    %
    % Output labels are in {-1, +1}.
    
    n = size(X,1);
    predictions = ones(n,1);
    predictions( model.polarity * X(:,model.feature) >= model.polarity * model.threshold ) = -1;
end

%% Bagging Ensemble Training Function
function models = baggingEnsembleTrain(X, y, T)
    % Trains T decision stumps on bootstrap samples of the training data.
    %
    % Inputs:
    %   X, y  - training data.
    %   T     - number of bootstrap samples (ensemble members).
    %
    % Output:
    %   models - a cell array of trained decision stump models.
    
    n = size(X,1);
    models = cell(T,1);
    for t = 1:T
        % Bootstrap sample indices (sampling with replacement)
        idx = randi(n, n, 1);
        X_boot = X(idx, :);
        y_boot = y(idx);
        % Uniform weights for bagging
        weights = ones(n,1) / n;
        model = decisionStumpTrain(X_boot, y_boot, weights);
        models{t} = model;
    end
end

%% Bagging Ensemble Prediction Function
function predictions = baggingEnsemblePredict(models, X)
    % Predicts labels for X using an ensemble of decision stumps trained by bagging.
    % The final prediction is determined by majority vote.
    
    T = length(models);
    n = size(X,1);
    pred_matrix = zeros(n, T);
    for t = 1:T
        pred_matrix(:,t) = decisionStumpPredict(models{t}, X);
    end
    % Aggregate by summing the predictions and taking the sign
    predictions = sign(sum(pred_matrix, 2));
    % In case the sum is 0, assign class +1.
    predictions(predictions == 0) = 1;
end

%% Boosting Ensemble Training Function (AdaBoost)
function models = boostingEnsembleTrain(X, y, T)
    % Trains an ensemble using AdaBoost with decision stumps.
    %
    % Inputs:
    %   X, y  - training data.
    %   T     - number of boosting iterations.
    %
    % Output:
    %   models - a cell array of structures; each structure contains
    %            a decision stump and its weight (alpha).
    
    n = size(X,1);
    weights = ones(n,1) / n;  % Initialize sample weights uniformly
    models = cell(T,1);
    
    for t = 1:T
        % Train decision stump using current weights
        model = decisionStumpTrain(X, y, weights);
        predictions = decisionStumpPredict(model, X);
        
        % Compute weighted error
        err = sum(weights .* (predictions ~= y));
        % Avoid division by zero
        if err == 0
            err = 1e-10;
        end
        % Compute model weight alpha
        alpha = 0.5 * log((1 - err) / err);
        model.alpha = alpha;
        models{t} = model;
        
        % Update weights: increase weight on misclassified samples
        weights = weights .* exp(-alpha * y .* predictions);
        weights = weights / sum(weights); % Normalize
    end
end

%% Boosting Ensemble Prediction Function
function predictions = boostingEnsemblePredict(models, X)
    % Predicts labels for X using an AdaBoost ensemble of decision stumps.
    % The final prediction is the sign of the weighted sum of predictions.
    
    T = length(models);
    n = size(X,1);
    final_pred = zeros(n,1);
    for t = 1:T
        model = models{t};
        pred = decisionStumpPredict(model, X);
        final_pred = final_pred + model.alpha * pred;
    end
    predictions = sign(final_pred);
    predictions(predictions == 0) = 1;
end
