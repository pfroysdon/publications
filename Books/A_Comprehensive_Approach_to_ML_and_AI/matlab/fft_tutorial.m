close all; clear all; clc; 

% Generate a sample signal: a sum of sine waves at 50 Hz and 120 Hz.
fs = 1024;            % Sampling frequency in Hz
t = 0:1/fs:1-1/fs;    % Time vector (1 second)
signal = sin(2*pi*50*t) + 0.5*sin(2*pi*120*t);

% Compute FFT using the custom function
X = myFFT(signal);

% Create a frequency vector for plotting
N = length(signal);
f = (0:N-1) * (fs/N);

% Plot the magnitude spectrum
figure;
plot(f, abs(X));
title('Magnitude Spectrum of the Signal');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
grid on;

% save_all_figs_OPTION('results/fft','png',1)

function X = myFFT(x)
    % myFFT - Recursive implementation of the Fast Fourier Transform
    % Input:
    %   x - input signal vector (length must be a power of 2)
    % Output:
    %   X - FFT of the input signal

    N = length(x);
    if N == 1
        X = x;
        return;
    end
    if mod(N, 2) ~= 0
        error('Length of input x must be a power of 2.');
    end
    % Divide the signal into even and odd parts
    x_even = myFFT(x(1:2:end));
    x_odd = myFFT(x(2:2:end));
    
    % Pre-allocate the output vector
    X = zeros(1, N);
    for k = 0:(N/2 - 1)
        twiddle = exp(-2*pi*1i*k/N);
        X(k+1) = x_even(k+1) + twiddle * x_odd(k+1);
        X(k+N/2+1) = x_even(k+1) - twiddle * x_odd(k+1);
    end
end
