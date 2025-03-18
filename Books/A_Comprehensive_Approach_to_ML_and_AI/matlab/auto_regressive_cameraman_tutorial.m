function auto_regressive_deep_learning_cameraman
    % AUTO_REGRESSIVE_DEEP_LEARNING_CAMERAMAN
    %
    % This tutorial demonstrates an auto-regressive deep learning model applied
    % to the "cameraman.tif" image. The model is trained to predict the next
    % pixel intensity given a fixed window of previous pixels.
    %
    % Steps:
    %   1. Load and downsample the image.
    %   2. Flatten the image to a 1D sequence.
    %   3. Create training samples: for each pixel (after a seed window),
    %      use the previous 'window_size' pixel values as input and the current
    %      pixel as the target.
    %   4. Build a simple neural network (one hidden layer) to predict the next pixel.
    %   5. Train the network using gradient descent.
    %   6. Use the trained model auto-regressively to generate the full image.
    %   7. Reshape and display the generated image alongside the original.
    
    %% 1. Load and Preprocess the Image
    img = imread('data/cameraman.tif');    % Load built-in image
    img = im2double(img);             % Convert to double precision in [0,1]
    % Downsample for ease of training (e.g., 64x64)
    img_ds = imresize(img, [64, 64]);
    % Flatten the image into a 1D sequence
    x = img_ds(:);                   % Vector of pixel intensities
    N = length(x);                   % Total number of pixels
    
    %% 2. Create Training Data for Auto-Regressive Prediction
    window_size = 16;                % Number of previous pixels used as input
    num_samples = N - window_size;   % Total training examples
    
    % Each training sample: input = [x(i), ..., x(i+window_size-1)], target = x(i+window_size)
    X_train = zeros(window_size, num_samples);
    y_train = zeros(1, num_samples);
    for i = 1:num_samples
        X_train(:, i) = x(i:i+window_size-1);
        y_train(i) = x(i+window_size);
    end
    
    %% 3. Set Up the Neural Network Architecture
    % Network architecture:
    %   - Input layer: window_size (16)
    %   - Hidden layer: hidden_dim (e.g., 50 neurons) with ReLU activation
    %   - Output layer: 1 neuron with sigmoid activation (predicting pixel intensity in [0,1])
    input_dim = window_size;
    hidden_dim = 50;
    output_dim = 1;
    
    rng(1);  % For reproducibility
    % Initialize weights with small random values
    W1 = 0.01 * randn(hidden_dim, input_dim);
    b1 = zeros(hidden_dim, 1);
    W2 = 0.01 * randn(output_dim, hidden_dim);
    b2 = zeros(output_dim, 1);
    
    %% 4. Training Parameters
    learning_rate = 0.01;
    num_epochs = 5000;
    m = num_samples;   % Number of training samples
    losses = zeros(num_epochs, 1);
    
    %% 5. Train the Model Using Batch Gradient Descent
    for epoch = 1:num_epochs
        % Forward propagation
        % X_train has size [window_size x m]
        Z1 = W1 * X_train + repmat(b1, 1, m);  % Hidden pre-activation [hidden_dim x m]
        A1 = relu(Z1);                        % Hidden activation
        Z2 = W2 * A1 + repmat(b2, 1, m);        % Output pre-activation [1 x m]
        A2 = sigmoid(Z2);                     % Network predictions
        
        % Compute mean squared error loss
        loss = 0.5 * mean((A2 - y_train).^2);
        losses(epoch) = loss;
        
        % Backpropagation
        dZ2 = (A2 - y_train) .* sigmoid_deriv(Z2); % [1 x m]
        dW2 = (dZ2 * A1') / m;                      % [1 x hidden_dim]
        db2 = mean(dZ2, 2);                         % [1 x 1]
        
        dA1 = W2' * dZ2;                            % [hidden_dim x m]
        dZ1 = dA1 .* relu_deriv(Z1);                % [hidden_dim x m]
        dW1 = (dZ1 * X_train') / m;                 % [hidden_dim x input_dim]
        db1 = mean(dZ1, 2);                         % [hidden_dim x 1]
        
        % Update parameters
        W1 = W1 - learning_rate * dW1;
        b1 = b1 - learning_rate * db1;
        W2 = W2 - learning_rate * dW2;
        b2 = b2 - learning_rate * db2;
        
        % Optionally display progress every 500 epochs
        if mod(epoch, 500) == 0
            fprintf('Epoch %d/%d, Loss: %.6f\n', epoch, num_epochs, loss);
        end
    end
    
    %% 6. Auto-Regressive Generation of the Image
    % Use the trained model to generate an image pixel-by-pixel.
    % Seed the generation with the first 'window_size' pixels from the original image.
    gen_length = N;  % Total number of pixels to generate (same as original)
    generated_seq = zeros(gen_length, 1);
    generated_seq(1:window_size) = x(1:window_size); % Seed
    
    % For each subsequent pixel, use the previous window_size values as input.
    for i = window_size+1:gen_length
        input_seq = generated_seq(i-window_size:i-1);  % [window_size x 1]
        % Forward propagate for one sample
        z1 = W1 * input_seq + b1;
        a1 = relu(z1);
        z2 = W2 * a1 + b2;
        a2 = sigmoid(z2);
        generated_seq(i) = a2;
    end
    
    % Reshape the generated sequence into the image dimensions
    gen_img = reshape(generated_seq, size(img_ds));
    
    %% 7. Visualization
    figure('Position', [100, 100, 1200, 500]);
    
    % Show the original downsampled image
    subplot(1,2,1);
    imshow(img_ds);
    title('Original Downsampled Cameraman Image');
    
    % Show the generated image from the auto-regressive model
    subplot(1,2,2);
    imshow(gen_img);
    title('Auto-Regressive Generated Image');
    
    % Plot the training loss curve
    figure;
    plot(1:num_epochs, losses, 'LineWidth', 2);
    xlabel('Epoch');
    ylabel('Loss');
    title('Training Loss');
    grid on;
end

%% Activation Functions and Their Derivatives

function A = relu(Z)
    % ReLU activation function
    A = max(0, Z);
end

function dA = relu_deriv(Z)
    % Derivative of ReLU activation
    dA = double(Z > 0);
end

function A = sigmoid(Z)
    % Sigmoid activation function
    A = 1 ./ (1 + exp(-Z));
end

function dA = sigmoid_deriv(Z)
    % Derivative of the sigmoid function
    s = sigmoid(Z);
    dA = s .* (1 - s);
end
