# fft_tutorial.R
# Generate a sample signal: a sum of sine waves at 50 Hz and 120 Hz.
fs <- 1024                      # Sampling frequency in Hz
t <- seq(0, 1 - 1/fs, by = 1/fs) # Time vector (1 second)
signal <- sin(2 * pi * 50 * t) + 0.5 * sin(2 * pi * 120 * t)

# Custom recursive FFT implementation (myFFT)
myFFT <- function(x) {
  N <- length(x)
  if (N == 1) return(x)
  if (N %% 2 != 0) stop("Length of input x must be a power of 2.")
  x_even <- myFFT(x[seq(1, N, by = 2)])
  x_odd <- myFFT(x[seq(2, N, by = 2)])
  X <- numeric(N) + 0i
  for (k in 0:((N/2)-1)) {
    twiddle <- exp(-2 * pi * 1i * k / N)
    X[k + 1] <- x_even[k + 1] + twiddle * x_odd[k + 1]
    X[k + N/2 + 1] <- x_even[k + 1] - twiddle * x_odd[k + 1]
  }
  X
}

X_fft <- myFFT(signal)
N_sig <- length(signal)
f <- seq(0, fs - fs/N_sig, length.out = N_sig)

# Plot the magnitude spectrum
plot(f, Mod(X_fft), type = "l", main = "Magnitude Spectrum of the Signal",
     xlab = "Frequency (Hz)", ylab = "Magnitude")
grid()
