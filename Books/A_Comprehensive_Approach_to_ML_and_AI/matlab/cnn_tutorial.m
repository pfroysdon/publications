
close all; clear all; clc; 

%% Load and Preprocess Image
% For demonstration, we use the built-in 'cameraman.tif' image.
I = imread('data/cameraman.tif');
I = im2double(I); % Normalize image to [0,1]

% Display the original image
figure;
subplot(1,3,1);
imshow(I);
title('Original Image');

%% CNN Parameters
% Convolution kernel (e.g., edge detector)
K = [-1 -1 -1; 0 0 0; 1 1 1];
% Pooling size (2x2)
poolSize = 2;
% Fully connected layer parameters (random initialization)
% The size of the flattened feature map will be determined later.
% For demonstration, we set a single output neuron.

%% Forward Pass through CNN
% 1. Convolution: valid convolution
S = myConv2(I, K);

% 2. ReLU Activation
A = max(0, S);

% 3. Max Pooling
P = maxPool(A, poolSize);

% 4. Flatten the pooled feature map
f = P(:);

% 5. Fully connected layer: for demonstration, use random weights and bias.
W_fc = randn(1, numel(f)) * 0.01;
b_fc = 0;
y = W_fc * f + b_fc;

%% Visualization of Intermediate Steps
subplot(1,3,2);
    imagesc(A);
    colormap gray;
    title('ReLU Activation Output');
    axis image off;
subplot(1,3,3);
    imagesc(P);
    colormap gray;
    title('Max Pooling Output');
    axis image off;

save_all_figs_OPTION('cnn2','png',1); 

figure;
    subplot(1,2,1);
    imshow(I);
    title('Original Image');
subplot(1,2,2);
    imagesc(S);
    axis square
    colormap gray;
    title('Convolved Image (Valid Mode)');
    colorbar;    

% save_all_figs_OPTION('results/cnn3','png',1)

fprintf('The fully connected layer output (prediction) is: %.4f\n', y);


function S = myConv2(I, K)
    % MYCONV2 performs 2D convolution of matrix I with kernel K in valid mode.
    %   I: Input image (2D matrix)
    %   K: Convolution kernel (2D matrix)
    %   S: Output matrix after valid convolution
    %
    % The function computes the convolution result S by sliding the kernel K
    % over the input I. Only positions where the kernel fully overlaps with I
    % are considered ("valid" convolution).
    
    % Get dimensions of the input and the kernel
    [H, W] = size(I);
    [kH, kW] = size(K);
    
    % Calculate the dimensions of the output matrix
    outH = H - kH + 1;
    outW = W - kW + 1;
    
    % Pre-allocate the output matrix S with zeros
    S = zeros(outH, outW);
    
    % Loop over every valid position in I
    for i = 1:outH
        for j = 1:outW
            % Extract the current patch from I
            patch = I(i:i+kH-1, j:j+kW-1);
            % Compute the sum of elementwise multiplication of patch and K
            S(i,j) = sum(sum(patch .* K));
        end
    end
end


function P = maxPool(A, poolSize)
    % MAXPOOL performs max pooling on matrix A with a pooling window of size poolSize.
    [H, W] = size(A);
    Hp = floor(H / poolSize);
    Wp = floor(W / poolSize);
    P = zeros(Hp, Wp);
    for i = 1:Hp
        for j = 1:Wp
            patch = A((i-1)*poolSize+1:i*poolSize, (j-1)*poolSize+1:j*poolSize);
            P(i,j) = max(patch(:));
        end
    end
end

function s = sigmoid(x)
    % SIGMOID computes the sigmoid function.
    s = 1 ./ (1 + exp(-x));
end
