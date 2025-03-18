% Example – PCA to compress color images
%  - Recall that a color image is made up of pixels with red, green, blue 
%    (RGB) values.
%  - Essentially, we have a collection of 3D vectors in RGB space.


% Start of script
%-------------------------------------------------------------------------%
close all;                   	% close all figures
clearvars; clearvars -global;	% clear all variables
clc;                         	% clear the command terminal
format shortG;                 	% pick the most compact numeric display
format compact;                	% suppress excess blank lines

% Read data & plot
%-------------------------------------------------------------------------%
RGB = im2double(imread('data/arizona_photo.jpg'));

% Convert 3-dimensional array array to 2D, where each row is a pixel (RGB)
X = reshape(RGB, [], 3);
N = size(X,1); % N is the number of pixels

% Plot pixels in color space. To limit the number of points, only plot
% every 100th point.
figure
hold on
for i=1:100:size(X,1)
    mycolor = X(i,:);
    mycolor = max(mycolor, [0 0 0]);
    mycolor = min(mycolor, [1 1 1]);
    plot3(X(i, 1), X(i, 2), X(i, 3),'.', 'Color', mycolor);
end
xlabel('red'), ylabel('green'), zlabel('blue');
xlim([0 1]), ylim([0 1]), zlim([0 1]);
hold off
axis equal

% Do PCA on color vectors ... keep the top two PCs.
%-------------------------------------------------------------------------%
% Get mean and covariance
m_x = mean(X);
C_x = cov(X);

% Get eigenvalues and eigenvectors of C_x.
% Produces V,D such that C_x*V = V*D.
% So the eigenvectors are the columns of V.
[V,D] = eig(C_x);
e1 = V(:,3);
disp('Eigenvector e1:'), disp(e1);
e2 = V(:,2);
disp('Eigenvector e2:'), disp(e2);
e3 = V(:,1);
disp('Eigenvector e3:'), disp(e3);
d1 = D(3,3);
disp('Eigenvalue d1:'), disp(d1);
d2 = D(2,2);
disp('Eigenvalue d2:'), disp(d2);
d3 = D(1,1);
disp('Eigenvalue d3:'), disp(d3);

% Construct matrix A such that the 1st row of A is the eigenvector 
% corresponding to the largest eigenvalue, the 2nd row is the eigenvector 
% corresponding to the second largest eigenvalue, etc. 
A = [e1'; e2'; e3'];

% Project input vectors x onto eigenvectors. For each (column) vector x,
% we will use the equation y = A*(x - mx).
% To explain the Matlab commands below:
% X is our (N,3) array of vectors; each row is a vector.
% mx is the mean of the vectors, size (1,3).
% We first subtract off the mean using X - repmat(mx,N,1).
% We then transpose that result so that each vector is a column.
% We then apply our transform A to each column.
Y = A*(X - repmat(m_x,N,1))'; % Y has size 3xN

% Display y vectors as images 
[height,width,depth] = size(RGB); 
Y1 = reshape(Y(1,:), height, width); 
Y2 = reshape(Y(2,:), height, width); 
Y3 = reshape(Y(3,:), height, width); 
figure; 
subplot(1,3,1), imshow(Y1,[]); 
subplot(1,3,2), imshow(Y2,[]); 
subplot(1,3,3), imshow(Y3,[]);

% Reconstruct image using only Y1 and Y2. For each (column) vector y,
% we will use the equation x = A'*y + mx.
% To explain the Matlab commands below:
% Y is our (3,N) array of vectors; where each column is a vector.
% A(1:k,:) is the first k rows of A.
% Y(1:k,:) is the first k rows of Y.
% A(1:k,:)' * Y(1:k,:) produces our transformed vectors (3xN); we then
% transpose that to make an array of size Nx3, and add the mean.
k = 1;
A_k = A(1:k,:);
X_r = (A(1:k,:)' * Y(1:k,:))' + repmat(m_x,N,1);

figure
hold on
for i=1:100:size(X_r,1)
    mycolor = X_r(i,:);
    mycolor = max(mycolor, [0 0 0]);
    mycolor = min(mycolor, [1 1 1]);
    plot3(X_r(i, 1), X_r(i, 2), X_r(i, 3),'.', 'Color', mycolor);
end
xlabel('red'), ylabel('green'), zlabel('blue');
xlim([0 1]), ylim([0 1]), zlim([0 1]);
hold off
axis equal

% Display original image and reconstructed image
figure, subplot(1,2,1), imshow(RGB), title('Original');
Ir(:,:,1) = reshape(X_r(:,1), height, width);
Ir(:,:,2) = reshape(X_r(:,2), height, width);
Ir(:,:,3) = reshape(X_r(:,3), height, width);
subplot(1,2,2), imshow(Ir), title('Reconstructed');

% Display original image and reconstructed image
figure
imshow(RGB)
title('Original');

Ir(:,:,1) = reshape(X_r(:,1), height, width);
Ir(:,:,2) = reshape(X_r(:,2), height, width);
Ir(:,:,3) = reshape(X_r(:,3), height, width);

figure
imshow(Ir)
title('Reconstructed');

% save_all_figs_OPTION('results/svd','png',0)





























