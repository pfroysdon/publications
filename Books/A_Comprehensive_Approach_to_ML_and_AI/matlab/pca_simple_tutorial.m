% Description:
% In this demo, we use PCA to compute the principle components of the data,
% then plot the results using only the 1st principle component.


% Start of script
%-------------------------------------------------------------------------%
close all;                   	% close all figures
clearvars; clearvars -global;	% clear all variables
clc;                         	% clear the command terminal
format shortG;                 	% pick the most compact numeric display
format compact;                	% suppress excess blank lines


% PCA
%-------------------------------------------------------------------------%
randn('state',0);

% Make sample data
N = 50;
x_in(:,1) = randn(N,1) + 1;
x_in(:,2) = 0.5*randn(N,1) + 0.5;
theta = -0.707;
R = [ cos(theta) sin(theta);
    -sin(theta) cos(theta)];
x_in = x_in*R;

% Step 1: calculate the mean of input vectors
%-------------------------------------------------------------------------%
mu_x = mean(x_in);

% Step 2: subtract off the mean
%-------------------------------------------------------------------------%
x_bar = x_in - repmat(mu_x,N,1);

% Step 3: calculate the covariance of input vectors
%-------------------------------------------------------------------------%
C_x = cov(x_bar);
disp('Covariance of input');
disp(C_x);

% Step 4: calculate the eigenvalues and eigenvectors of Cx
%-------------------------------------------------------------------------%
% Produces V,D such that Cx*V = V*D.
% So the eigenvectors are the columns of V.
[V,D] = eig(C_x);

e1 = V(:,1);
disp('Eigenvector e1:'), disp(e1);
e2= V(:,2);
disp('Eigenvector e2:'), disp(e2);
d1 = D(1,1);
disp('Eigenvalue d1:'), disp(d1);
d2 = D(2,2);
disp('Eigenvalue d2:'), disp(d2);

% Verify eigenvalues and eigenvectors
disp('C_x*e1 = '), disp(C_x*e1);
disp('d1*e1 = '), disp(d1*e1);
disp('C_x*e2 = '), disp(C_x*e2);
disp('d2*e2 = '), disp(d2*e2);

figure
plot(x_in(:,1), x_in(:,2), 'o');
hold on;
plot( mu_x(1), mu_x(2), '+r', 'LineWidth', 2);
hold off;
title('Centered input vectors');
axis equal
axis([-3.0 3.0 -3.0 3.0]);

% Draw eigenvectors
figure
plot(x_bar(:,1), x_bar(:,2), 'bo');
hold on;
line([0 d1*e1(1)*2], [0 d1*e1(2)*2], 'Color', 'r', 'LineWidth', 2);
line([0 d2*e2(1)*2], [0 d2*e2(2)*2], 'Color', 'r', 'LineWidth', 2);
hold off;
title('Centered input vectors');
axis equal
axis([-3.0 3.0 -3.0 3.0]);
% save_all_figs_OPTION('results/pca1','png',1)

% Step 4: project the input data onto principal components
%-------------------------------------------------------------------------%
y = [e2'; e1']*x_bar';

figure
plot(y(1,:),y(2,:), 'o');
title('Projections onto principal components');
axis equal
axis([-3.0 3.0 -3.0 3.0]);

% Step 5: project the input data using only one principal
%-------------------------------------------------------------------------%
% component
y1 = [e1']*x_bar';
sigma_x = std(y1);

figure
plot(y1(1,:),zeros(1,length(y1)), 'o');
title('Projections onto one principal component');
axis equal
axis([-3.0 3.0 -3.0 3.0]);

% Project input data using only one principal
% component
y2 = [e2']*x_bar';
sigma_x = std(y2);

figure
plot(y2(1,:),zeros(1,length(y2)), 'o');
title('Projections onto one principal component');
axis equal
axis([-3.0 3.0 -3.0 3.0]);
% save_all_figs_OPTION('results/pca2','png',1)

% clean-up
%-------------------------------------------------------------------------%
% dock_all_figures

% save_all_figs_OPTION('pca','png')