% Model Distillation Tutorial in MATLAB
%
% This tutorial demonstrates how to perform model distillation on a simple
% 2D classification problem. We first train a teacher network on a synthetic
% dataset with two classes. The teacher is a relatively large neural network.
% Then, we train a smaller student network to mimic the teacher's behavior by
% minimizing a combined loss:
%
%    Loss = lambda * HardLoss + (1 - lambda) * T^2 * DistillLoss,
%
% where HardLoss is the cross-entropy loss between the student's predictions
% and the true labels, and DistillLoss is the KL divergence between the teacher's
% and student's softened probability distributions (obtained by dividing the
% logits by a temperature T). The teacher network's parameters are frozen during
% student training.
%
% Finally, the decision boundaries of both the teacher and student models are
% plotted.
%
% All functions (softmax, ReLU and its derivative, loss functions, etc.) are
% implemented from scratch.

clear; clc; close all; rng(1);

%% 1. Generate Synthetic Dataset
N = 200;  % total number of samples
% Class 1: Gaussian centered at (1,1)
X1 = randn(2, N/2) * 0.5 + repmat([1; 1], 1, N/2);
% Class 2: Gaussian centered at (3,3)
X2 = randn(2, N/2) * 0.5 + repmat([3; 3], 1, N/2);
X = [X1, X2];  % 2 x N data matrix
labels = [ones(1, N/2), 2*ones(1, N/2)];  % labels: 1 or 2

% One-hot encoding of labels
Y = zeros(2, N);
for i = 1:N
    Y(labels(i), i) = 1;
end

%% 2. Train Teacher Network
% Teacher architecture: 2 -> 20 -> 2
inputDim = 2;
hiddenDim_teacher = 20;
outputDim = 2;

% Initialize teacher parameters
W1_teacher = randn(hiddenDim_teacher, inputDim) * 0.01;
b1_teacher = zeros(hiddenDim_teacher, 1);
W2_teacher = randn(outputDim, hiddenDim_teacher) * 0.01;
b2_teacher = zeros(outputDim, 1);

teacher_lr = 0.01;
numIter_teacher = 5000;

for iter = 1:numIter_teacher
    % Forward pass
    Z1 = W1_teacher * X + repmat(b1_teacher, 1, N); % hidden pre-activation
    H1 = relu(Z1);
    logits_teacher = W2_teacher * H1 + repmat(b2_teacher, 1, N);
    probs_teacher = softmax(logits_teacher);  % softmax applied column-wise
    
    % Compute cross-entropy loss
    loss_teacher = -mean(sum(Y .* log(probs_teacher + 1e-8), 1));
    
    % Backpropagation
    d_logits = probs_teacher - Y;  % (2 x N)
    grad_W2_teacher = (d_logits * H1') / N;
    grad_b2_teacher = mean(d_logits, 2);
    
    d_H1 = W2_teacher' * d_logits;  % (hiddenDim_teacher x N)
    d_Z1 = d_H1 .* reluDerivative(Z1);  % (hiddenDim_teacher x N)
    grad_W1_teacher = (d_Z1 * X') / N;
    grad_b1_teacher = mean(d_Z1, 2);
    
    % Parameter updates
    W2_teacher = W2_teacher - teacher_lr * grad_W2_teacher;
    b2_teacher = b2_teacher - teacher_lr * grad_b2_teacher;
    W1_teacher = W1_teacher - teacher_lr * grad_W1_teacher;
    b1_teacher = b1_teacher - teacher_lr * grad_b1_teacher;
    
    if mod(iter, 500)==0
        fprintf('Teacher Iteration %d, Loss: %.4f\n', iter, loss_teacher);
    end
end

% Freeze teacher parameters
Teacher.W1 = W1_teacher;
Teacher.b1 = b1_teacher;
Teacher.W2 = W2_teacher;
Teacher.b2 = b2_teacher;
Teacher.T = 2;  % temperature for distillation

%% 3. Train Student Network with Distillation
% Student architecture: 2 -> 5 -> 2 (smaller network)
hiddenDim_student = 5;

W1_student = randn(hiddenDim_student, inputDim) * 0.01;
b1_student = zeros(hiddenDim_student, 1);
W2_student = randn(outputDim, hiddenDim_student) * 0.01;
b2_student = zeros(outputDim, 1);

student_lr = 0.01;
numIter_student = 5000;
lambda = 0.5;  % weight for hard loss vs. distillation loss
T = Teacher.T;

for iter = 1:numIter_student
    % Teacher forward pass (using temperature T)
    Z1_teacher = Teacher.W1 * X + repmat(Teacher.b1, 1, N);
    H1_teacher = relu(Z1_teacher);
    logits_teacher = Teacher.W2 * H1_teacher + repmat(Teacher.b2, 1, N);
    logits_teacher_T = logits_teacher / T;
    soft_targets = softmax(logits_teacher_T);  % 2 x N
    
    % Student forward pass with temperature
    Z1_student = W1_student * X + repmat(b1_student, 1, N);
    H1_student = relu(Z1_student);
    logits_student = W2_student * H1_student + repmat(b2_student, 1, N);
    logits_student_T = logits_student / T;
    student_soft = softmax(logits_student_T);  % 2 x N
    
    % Student hard predictions (without temperature)
    student_probs = softmax(logits_student);  % 2 x N
    
    % Hard loss: cross-entropy between student's hard predictions and true labels
    hard_loss = -mean(sum(Y .* log(student_probs + 1e-8), 1));
    
    % Distillation loss: KL divergence between teacher and student soft outputs
    % For each sample: KL = sum( p_teacher * (log(p_teacher) - log(p_student)) )
    kl_div = sum(soft_targets .* (log(soft_targets + 1e-8) - log(student_soft + 1e-8)), 1);
    distill_loss = mean(kl_div);
    
    % Combined loss (with temperature scaling factor T^2 for distillation loss)
    loss_student = lambda * hard_loss + (1 - lambda) * T^2 * distill_loss;
    
    % Backpropagation for student network
    % We need gradients for both the hard loss and the distillation loss.
    % For the hard loss: derivative dHard = student_probs - Y.
    d_logits_hard = (student_probs - Y);  % 2 x N
    % For the distillation loss: derivative with respect to logits_student_T
    d_logits_distill = (student_soft - soft_targets);  % 2 x N
    % Chain rule: d_logits_distill w.r.t. logits_student = (1/T)* d_logits_distill
    d_logits_distill = (1/T) * d_logits_distill;
    
    % Combined gradient on logits_student:
    d_logits_student = lambda * d_logits_hard + (1 - lambda) * T^2 * d_logits_distill;  % 2 x N
    
    % Backprop into student network output layer
    grad_W2_student = (d_logits_student * H1_student') / N;  % (2 x hiddenDim_student)
    grad_b2_student = mean(d_logits_student, 2);  % (2 x 1)
    
    % Backprop into hidden layer:
    d_H1_student = W2_student' * d_logits_student;  % (hiddenDim_student x N)
    d_Z1_student = d_H1_student .* reluDerivative(Z1_student);  % (hiddenDim_student x N)
    grad_W1_student = (d_Z1_student * X') / N;  % (hiddenDim_student x 2)
    grad_b1_student = mean(d_Z1_student, 2);    % (hiddenDim_student x 1)
    
    % Update student network parameters:
    W2_student = W2_student - student_lr * grad_W2_student;
    b2_student = b2_student - student_lr * grad_b2_student;
    W1_student = W1_student - student_lr * grad_W1_student;
    b1_student = b1_student - student_lr * grad_b1_student;
    
    if mod(iter,500)==0
        fprintf('Student Iteration %d, Hard Loss: %.4f, Distill Loss: %.4f, Combined Loss: %.4f\n', ...
            iter, hard_loss, distill_loss, loss_student);
    end
end

%% 4. Evaluate and Visualize Decision Boundaries
% Define a grid over the input space.
[xGrid, yGrid] = meshgrid(linspace(min(X(1,:))-1, max(X(1,:))+1, 100), ...
                          linspace(min(X(2,:))-1, max(X(2,:))+1, 100));
gridPoints = [xGrid(:)'; yGrid(:)'];

% Teacher predictions:
Z1_teacher = Teacher.W1 * gridPoints + repmat(Teacher.b1, 1, size(gridPoints,2));
H1_teacher = relu(Z1_teacher);
logits_teacher = Teacher.W2 * H1_teacher + repmat(Teacher.b2, 1, size(gridPoints,2));
probs_teacher = softmax(logits_teacher);
[~, pred_teacher] = max(probs_teacher, [], 1);

% Student predictions:
Z1_student = W1_student * gridPoints + repmat(b1_student, 1, size(gridPoints,2));
H1_student = relu(Z1_student);
logits_student = W2_student * H1_student + repmat(b2_student, 1, size(gridPoints,2));
probs_student = softmax(logits_student);
[~, pred_student] = max(probs_student, [], 1);

% Reshape predictions to grid shape
pred_teacher_grid = reshape(pred_teacher, size(xGrid));
pred_student_grid = reshape(pred_student, size(xGrid));

figure;
subplot(1,2,1);
    imagesc([min(X(1,:))-1, max(X(1,:))+1], [min(X(2,:))-1, max(X(2,:))+1], pred_teacher_grid);set(gca, 'YDir', 'normal');
    title('Teacher Decision Boundary');
    xlabel('Feature 1'); ylabel('Feature 2');
    colormap(jet); colorbar;
subplot(1,2,2);
    imagesc([min(X(1,:))-1, max(X(1,:))+1], [min(X(2,:))-1, max(X(2,:))+1], pred_student_grid);
    set(gca, 'YDir', 'normal');
    title('Student Decision Boundary');
    xlabel('Feature 1'); ylabel('Feature 2');
    colormap(jet); colorbar;

%% Visualize Original Data with Teacher Decision Boundary

% Create a grid covering the input space
x_min = min(X(1,:)) - 1;
x_max = max(X(1,:)) + 1;
y_min = min(X(2,:)) - 1;
y_max = max(X(2,:)) + 1;
[xGrid, yGrid] = meshgrid(linspace(x_min, x_max, 100), linspace(y_min, y_max, 100));
gridPoints = [xGrid(:)'; yGrid(:)'];

% Compute teacher network outputs on the grid.
% (Assuming your teacher network is defined by parameters W1_teacher, b1_teacher,
%  W2_teacher, b2_teacher as in the tutorial.)
Z1_teacher = W1_teacher * gridPoints + repmat(b1_teacher, 1, size(gridPoints,2));
H1_teacher = relu(Z1_teacher);
logits_teacher = W2_teacher * H1_teacher + repmat(b2_teacher, 1, size(gridPoints,2));
probs_teacher = softmax(logits_teacher);  % 2 x numGridPoints

Z1_student = W1_student * gridPoints + repmat(b1_student, 1, size(gridPoints,2));
H1_student = relu(Z1_student);
logits_student = W2_student * H1_student + repmat(b2_teacher, 1, size(gridPoints,2));
probs_student = softmax(logits_student);  % 2 x numGridPoints

% Use the probability for class 1 (first class) to define the decision boundary.
p_class1_teacher = reshape(probs_teacher(1, :), size(xGrid));
p_class1_student = reshape(probs_student(1, :), size(xGrid));

% Plot the original data and decision boundary.
figure;
% subplot(1,2,1);
    hold on;
    % Plot data for Class 1 (assumed to be from X1) and Class 2 (from X2).
    scatter(X1(1,:), X1(2,:), 50, 'ro');
    scatter(X2(1,:), X2(2,:), 50, 'bo');
    % Overlay the decision boundary as a contour line where p_class1 == 0.5.
    contourf(xGrid, yGrid, p_class1_teacher, [0.5 0.5], 'k', 'LineWidth', 2, 'FaceAlpha', 0.2);
    colormap([0.8 0.8 1; 1 0.8 0.8]);
    xlabel('Feature 1');
    ylabel('Feature 2');
    title('Teacher Decision Boundary');
    legend('Class 1', 'Class 2', 'Decision Boundary');
    hold off;
% subplot(1,2,2);
    save_all_figs_OPTION('results/model_distillation1','png',1)
figure;
    hold on;
    % Plot data for Class 1 (assumed to be from X1) and Class 2 (from X2).
    scatter(X1(1,:), X1(2,:), 50, 'ro');
    scatter(X2(1,:), X2(2,:), 50, 'bo');
    % Overlay the decision boundary as a contour line where p_class1 == 0.5.
    contourf(xGrid, yGrid, p_class1_student, [0.5 0.5], 'k', 'LineWidth', 2, 'FaceAlpha', 0.2);
    colormap([0.8 0.8 1; 1 0.8 0.8]);
    xlabel('Feature 1');
    ylabel('Feature 2');
    title('Student Decision Boundary');
    legend('Class 1', 'Class 2', 'Decision Boundary');
    hold off;

    % save_all_figs_OPTION('results/model_distillation2','png',1)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Local Function Definitions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function y = softmax(x)
    % softmax computes the softmax for each column of x.
    x = x - max(x, [], 1);
    exp_x = exp(x);
    y = exp_x ./ sum(exp_x, 1);
end

function y = relu(x)
    y = max(x, 0);
end

function d = reluDerivative(x)
    d = double(x > 0);
end
