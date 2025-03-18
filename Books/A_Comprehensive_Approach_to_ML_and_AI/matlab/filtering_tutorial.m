clear; close all; clc;

% filter_tutorial - Demonstrates low-pass, high-pass, and band-pass filtering.
% This script uses the bilinear transform to design digital filters.


% Parameters
T = 0.01;             % Sampling period (s)
fs = 1/T;             % Sampling frequency (Hz)
tau = 0.05;           % Time constant for low-pass filter
n = 0:1:1000;         % Discrete time vector

% Generate a toy signal: combination of low and high frequency components
x = sin(2*pi*5*n*T) + 0.5*sin(2*pi*50*n*T)  + 50*sin(2*pi*500*n*T);

% Design Low-Pass Filter (LPF)
% Continuous-time transfer function: H_LP(s) = 1/(tau*s+1)
% Bilinear transform substitution: s = (2/T)*(1-z^(-1))/(1+z^(-1))
% The resulting discrete-time transfer function coefficients are derived below.
[b_lp, a_lp] = bilinear(1, [tau 1], fs);

% Design High-Pass Filter (HPF)
% Continuous-time transfer function: H_HP(s) = (tau*s)/(tau*s+1)
[b_hp, a_hp] = bilinear([tau 0], [tau 1], fs);

% Design Band-Pass Filter (BPF)
% One approach: cascade HPF and LPF. For simplicity, we use the product of
% coefficients from HPF and LPF (note: this is a simplified approach).
[b_bp, a_bp] = cascadeFilters(b_hp, a_hp, b_lp, a_lp);

% Apply filters using MATLAB's filter function
y_lp = filter(b_lp, a_lp, x);
y_hp = filter(b_hp, a_hp, x);
y_bp = filter(b_bp, a_bp, x);

% Plot the original and filtered signals
figure;
subplot(4,1,1);
plot(n*T, x, 'k-', 'LineWidth',1.5);
title('Original Signal');
xlabel('Time (s)'); ylabel('Amplitude');
grid on;

subplot(4,1,2);
plot(n*T, y_lp, 'b-', 'LineWidth',1.5);
title('Low-Pass Filtered Signal');
xlabel('Time (s)'); ylabel('Amplitude');
grid on;

subplot(4,1,3);
plot(n*T, y_hp, 'r-', 'LineWidth',1.5);
title('High-Pass Filtered Signal');
xlabel('Time (s)'); ylabel('Amplitude');
grid on;

subplot(4,1,4);
plot(n*T, y_bp, 'm-', 'LineWidth',1.5);
title('Band-Pass Filtered Signal');
xlabel('Time (s)'); ylabel('Amplitude');
grid on;

getFFT(n*T,x,'Original Signal')
getFFT(n*T,y_lp,'Original Signal')
getFFT(n*T,y_hp,'Original Signal')
getFFT(n*T,y_bp,'Original Signal')


function [b_cascade, a_cascade] = cascadeFilters(b1, a1, b2, a2)
    % cascadeFilters - Cascade two filters by convolving their coefficients.
    % b1, a1: coefficients of the first filter
    % b2, a2: coefficients of the second filter
    % b_cascade: convolution of b1 and b2
    % a_cascade: convolution of a1 and a2
    b_cascade = conv(b1, b2);
    a_cascade = conv(a1, a2);
end


function [domFreq,domFreqMag] = getFFT(t,sig,txt)
% Compute and plot the FFT for a given signal;
% 
% Description:
% This function finds the dominant frequency and magnitude for a given
% signal and time series  
% 
% Syntax:
%   [domFreq,domFreqMag] = getFFT(t,sig,txt)
% 
% Parameters:
%   t:          mx1 time array
%   sig:        mx1 signal array
%   txt:        text to display on the plot title
% 
% Return values:
%   domFreq:    dominant frequency
%   domFreqMag: dominant frequency magnitude
% 
% Reference:
% 

% Get rid of empty or nan lines
nn = 0;
for i = 1:length(t)
    if isnan(t(i)) || isnan(sig(i)) || isempty(t(i)) || isempty(sig(i))
    else
        nn = nn+1;
        t_filtered(nn) = t(i);
        sig_filtered(nn) = sig(i);
    end   
end

Fs = 1/mean(diff(t_filtered)); % Sampling frequency
L = length(t_filtered); % Length of signal

NFFT = 2^nextpow2(L); % Next power of 2 from length of Signal
Y = fft(sig_filtered,NFFT)/L;
f = Fs/2*linspace(0,1,NFFT/2);

%single-sided amplitude as function of frequency
SSA = 2*abs(Y(1:NFFT/2));

%Find the dominant frequency and magitude
domFreqMag = max(SSA);
Dom_Freq_ndx = find(SSA == max(SSA),1,'first');
domFreq = f(Dom_Freq_ndx);

% Plot single-sided amplitude spectrum.
figure
plot(f,SSA) 
title([txt ': Single-Sided Amplitude Spectrum of Signal(t_{filtered})'])
xlabel('Frequency (Hz)')
ylabel('|Y(f)|')

end