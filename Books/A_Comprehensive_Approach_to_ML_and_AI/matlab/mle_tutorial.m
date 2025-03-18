% mle_example - Estimate parameters of a normal distribution using MLE.
%
% Generate synthetic data from a normal distribution with true parameters
true_mu = 5;
true_sigma = 2;
n = 1000;
data = true_mu + true_sigma * randn(n, 1);

% Compute MLE estimates
mu_hat = mean(data);
sigma2_hat = mean((data - mu_hat).^2);
sigma_hat = sqrt(sigma2_hat);

fprintf('Estimated Mean: %.4f\n', mu_hat);
fprintf('Estimated Std Dev: %.4f\n', sigma_hat);

% Plot histogram of data and fitted normal density
figure;
histogram(data, 30, 'Normalization', 'pdf');
hold on;
x_values = linspace(min(data), max(data), 100);
y_values = (1/(sigma_hat*sqrt(2*pi))) * exp(-0.5*((x_values - mu_hat)/sigma_hat).^2);
plot(x_values, y_values, 'r-', 'LineWidth', 2);
title('MLE for Normal Distribution');
xlabel('Data Value');
ylabel('Probability Density');
legend('Data Histogram', 'Fitted Normal PDF');
grid on;
hold off;

% save_all_figs_OPTION('results/mle','png',1)