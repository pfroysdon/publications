# filtering_tutorial.R
# Demonstrates low-pass, high-pass, and band-pass filtering.
# This script uses the bilinear transform to design digital filters.

rm(list = ls())
graphics.off()
library(signal)  # For bilinear transformation

## Parameters
T <- 0.01                # Sampling period (s)
fs <- 1 / T              # Sampling frequency (Hz)
tau <- 0.05              # Time constant for low-pass filter
n <- 0:1000            # Discrete time vector

## Generate a toy signal: combination of low and high frequency components
x <- sin(2 * pi * 5 * n * T) + 0.5 * sin(2 * pi * 50 * n * T) + 50 * sin(2 * pi * 500 * n * T)

## Design Low-Pass Filter (LPF)
# Continuous-time transfer function: H_LP(s) = 1 / (tau*s + 1)
b_lp_ct <- 1
a_lp_ct <- c(tau, 1)
# Bilinear transform to get discrete-time coefficients
lpf <- bilinear(b_lp_ct, a_lp_ct, fs)
b_lp <- lpf$b; a_lp <- lpf$a

## Design High-Pass Filter (HPF)
# Continuous-time transfer function: H_HP(s) = (tau*s) / (tau*s + 1)
b_hp_ct <- c(tau, 0)
a_hp_ct <- c(tau, 1)
hpf <- bilinear(b_hp_ct, a_hp_ct, fs)
b_hp <- hpf$b; a_hp <- hpf$a

## Design Band-Pass Filter (BPF)
# Cascade HPF and LPF by convolving coefficients
cascadeFilters <- function(b1, a1, b2, a2) {
  list(b = conv(b1, b2), a = conv(a1, a2))
}
bp <- cascadeFilters(b_hp, a_hp, b_lp, a_lp)
b_bp <- bp$b; a_bp <- bp$a

## Apply filters
y_lp <- filter(b_lp, a_lp, x)
y_hp <- filter(b_hp, a_hp, x)
y_bp <- filter(b_bp, a_bp, x)

## Plot the signals
par(mfrow = c(4, 1))
plot(n * T, x, type = "l", col = "black", lwd = 1.5, main = "Original Signal",
     xlab = "Time (s)", ylab = "Amplitude")
plot(n * T, y_lp, type = "l", col = "blue", lwd = 1.5, main = "Low-Pass Filtered Signal",
     xlab = "Time (s)", ylab = "Amplitude")
plot(n * T, y_hp, type = "l", col = "red", lwd = 1.5, main = "High-Pass Filtered Signal",
     xlab = "Time (s)", ylab = "Amplitude")
plot(n * T, y_bp, type = "l", col = "magenta", lwd = 1.5, main = "Band-Pass Filtered Signal",
     xlab = "Time (s)", ylab = "Amplitude")
grid()

## Helper function: Compute FFT and plot amplitude spectrum
getFFT <- function(t, sig, txt) {
  t_filtered <- t[!is.na(t) & !is.na(sig)]
  sig_filtered <- sig[!is.na(t) & !is.na(sig)]
  Fs <- 1 / mean(diff(t_filtered))
  L <- length(sig_filtered)
  NFFT <- 2^ceiling(log2(L))
  Y <- fft(sig_filtered, n = NFFT) / L
  f <- Fs / 2 * seq(0, 1, length.out = NFFT/2)
  SSA <- 2 * abs(Y[1:(NFFT/2)])
  domFreqMag <- max(SSA)
  domFreq <- f[which.max(SSA)]
  plot(f, SSA, type = "l", main = paste(txt, ": Single-Sided Amplitude Spectrum"),
       xlab = "Frequency (Hz)", ylab = "|Y(f)|")
  grid()
  invisible(list(domFreq = domFreq, domFreqMag = domFreqMag))
}

# Compute and plot FFT for original and filtered signals
getFFT(n * T, x, "Original Signal")
getFFT(n * T, y_lp, "Low-Pass Signal")
getFFT(n * T, y_hp, "High-Pass Signal")
getFFT(n * T, y_bp, "Band-Pass Signal")
