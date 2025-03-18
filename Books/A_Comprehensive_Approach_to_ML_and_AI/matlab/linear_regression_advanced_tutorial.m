% Problem to be solved:
% The file regression_data.m contains the measurements of a quantity y at 
% the time instants t = 1; 2;...;100sec. Write a Matlab script that will 
% compute a least-squares approximation for y = f(t), as 
%   (i) a linear function, 
%   (ii) a quadratic polynomial
%   (iii) a cubic polynomial. 
% For each of the three models, what is the norm of the error in the
% approximation? Which is smallest, which is largest? Explain the results 
% theoretically.
% 
% Solution:                                                             
% This MATLAB script loads the supplied data, and performs a fit as a 
% Linear, Quadratic, and Cubic function of the supplied data.


% Start of script
%-------------------------------------------------------------------------%
close all;                   	% close all figures
clearvars; clearvars -global;	% clear all variables
clc;                         	% clear the command terminal
format shortG;                 	% pick the most compact numeric display
format compact;                	% suppress excess blank lines


% Simple Example
%-------------------------------------------------------------------------%
% x = 1:1:60;
% y = 0.2.*x + 5;
% 
% mu = 0;
% sigma = 1;
% rng(1);
% y_tilde = y + (sigma*randn(1,length(x))+mu);
% 
% plot(x, y_tilde, 'b*', x, y, 'r.-', 'LineWidth',1)




% Load the data
%-------------------------------------------------------------------------%
% load('LS_data.mat')
% linear_regression_data;

% define data
x = 1:1:100;
y = 1.07e-4.*x.^3 - 0.0088.*x.^2 + 0.3.*x + 2.1;

% define noisy data
mu = 0;
sigma = 2;
rng(1);
y_tilde = y + (sigma*randn(1,length(x))+mu);

% plot
plot(x,y,'.-', x,y_tilde,'o-')

[m,n] = size(y_tilde);


% Approximate f(x) as a 1-st order function using LS.
%-------------------------------------------------------------------------%

% create matricies and estimate the parameters, x.
H = [x; ones(m,n)]';
beta_ord1 = inv(H'*H)*H'*y_tilde';

% better method uses QR-decomposition:
% [Q,R] = qr(H,0);
% beta_ord1 = R\Q'*y_tilde'; % use "matrix left-divide" instead of inv(R)*Q'*y';

% compute the line-fit using the estimates in x.
f1 = beta_ord1(1).*x + beta_ord1(2);

% compute the norm of the error
e1 = norm(y_tilde-f1);


% Approximate f(x) as a 2-nd order function using LS.
%-------------------------------------------------------------------------%

% create matricies and estimate the parameters, x.
H = [x.^2; x; ones(m,n)]';
beta_ord2 = inv(H'*H)*H'*y_tilde';

% compute the line-fit using the estimates in x.
f2 = beta_ord2(1).*x.^2 + beta_ord2(2).*x + beta_ord2(3);

% compute the norm of the error
e2 = norm(y_tilde-f2);


% Approximate f(x) as a 3-rd order function using LS.
%-------------------------------------------------------------------------%

% create matricies and estimate the parameters, x.
H = [x.^3; x.^2; x; ones(m,n)]';
beta_ord3 = inv(H'*H)*H'*y_tilde';

% compute the line-fit using the estimates in x.
f3 = beta_ord3(1).*x.^3 + beta_ord3(2).*x.^2 + beta_ord3(3).*x + beta_ord3(4);

% compute the norm of the error
e3 = norm(y_tilde-f3);


% Approximate f(x) as a 3-rd order function using WLS.
%-------------------------------------------------------------------------%

% create matricies and estimate the parameters, x.
H = [x.^3; x.^2; x; ones(m,n)]';
W = 2^2.*eye(length(x));
beta_ord3_wls = inv(H'*W*H)*H'*W*y_tilde';

% compute the line-fit using the estimates in x.
f3_wls = beta_ord3_wls(1).*x.^3 + beta_ord3_wls(2).*x.^2 + beta_ord3_wls(3).*x + beta_ord3_wls(4);


% compute the norm of the error
e3_wls = norm(y_tilde-f3_wls);



% Plot & display the results.
%-------------------------------------------------------------------------%
% display the results
fprintf(1,'First-order fit coefficients:\n');
fprintf(1,'x(1) = %8.4f\n',beta_ord1(1));
fprintf(1,'x(2) = %8.4f\n\n',beta_ord1(2));

fprintf(1,'Second-order fit coefficients:\n');
fprintf(1,'x(1) = %8.4f\n',beta_ord2(1));
fprintf(1,'x(2) = %8.4f\n',beta_ord2(2));
fprintf(1,'x(3) = %8.4f\n\n',beta_ord2(3));

fprintf(1,'Third-order fit coefficients:\n');
fprintf(1,'x(1) = %8.4f\n',beta_ord3(1));
fprintf(1,'x(2) = %8.4f\n',beta_ord3(2));
fprintf(1,'x(3) = %8.4f\n',beta_ord3(3));
fprintf(1,'x(4) = %8.4f\n\n',beta_ord3(4));

fprintf(1,'First-order norm(error)  = %8.4f\n',e1);
fprintf(1,'Second-order norm(error) = %8.4f\n',e2);
fprintf(1,'Third-order norm(error)  = %8.4f\n',e3);

fprintf(1,'Third-order norm(error) LS-WLS  = %8.4f\n',e3-e3_wls);


% plot the data
figure;
plot(x, y_tilde,'o',...
     'LineWidth',1.5);
legend('data','Location','northwest');
title('Raw data');
xlabel('Time (day)');
ylabel('Stock Value ($)')
xlim([0 100]);
ylim([-10 60]);

% save_all_figs_OPTION('results/linear_regression1','png',1)

% plot the results
figure;
plot(x, y_tilde,  'o',...
     x, f1, '--',...
     'LineWidth',1.5);
legend('data','1^{st}-order fit',...
    'Location','northwest');
title('Approximate data as a 1^{st} order (Linear) function');
xlabel('Time (day)');
ylabel('Stock Value ($)')
xlim([0 100]);
ylim([-10 60]);

figure;
plot(x, y_tilde,  'o',...
     x, f1, '--',...
     x, f2, ':',...
     'LineWidth',1.5);
legend('data','1^{st}-order fit','2^{nd}-order fit',...
    'Location','northwest');
title('Approximate data as a 1^{st} and 2^{nd} order function');
xlabel('Time (day)');
ylabel('Stock Value ($)')
xlim([0 100]);
ylim([-10 60]);

figure;
plot(x, y_tilde,  'o',...
     x, f1, '--',...
     x, f2, ':',...
     x, f3, '-.',...
     'LineWidth',1.5);
legend('data','1^{st}-order fit','2^{nd}-order fit','3^{rd}-order fit',...
    'Location','northwest');
title('Approximate data as 1^{st}, 2^{nd} and 3^{rd} order functions');
xlabel('Time (day)');
ylabel('Stock Value ($)')
xlim([0 100]);
ylim([-10 60]);

% save_all_figs_OPTION('results/linear_regression2','png',1)

figure;
inc = 1;
x_1 = [];
y_tilde_1 = [];
for ii = [1,20:20:80]
    x_1(inc,1) = x(ii);
    y_tilde_1(inc,1) = y_tilde(ii);
    inc = inc + 1;
end
plot(x_1, y_tilde_1, 'o',...
     x, f1, '--',...
     x, f2, ':',...
     x, f3, '-.',...
     'LineWidth',1.5);
legend('data','1^{st}-order fit','2^{nd}-order fit','3^{rd}-order fit',...
    'Location','northwest');
title('Approximate data as 1^{st}, 2^{nd} and 3^{rd} functions');
xlabel('Time (day)');
ylabel('Stock Value ($)')
xlim([0 100]);
ylim([-10 60]);


figure;
plot(x, y_tilde, 'o',...
     x, f3, ':',...
     x, f3_wls, '-.',...
     'LineWidth',1.5);
legend('data','LS 3^{rd}-order fit','WLS 3^{rd}-order fit',...
    'Location','northwest');
title('Approximate data using LS and WLS');
xlabel('Time (days)');
ylabel('Stock Value ($)')
xlim([0 100]);
ylim([-10 60]);

figure;
plot(x, y_tilde, 'o',...
     x, f3, '*-',...
     'LineWidth',0.5);
legend('data','Least Squares 3^{rd}-order fit',...
    'Location','northwest');
title('Approximate data using Least Squares');
xlabel('Time (day)');
ylabel('Stock Value ($)')
xlim([0 100]);
ylim([-10 60]);

% dock_all_figures;
% save_all_figs_OPTION('linear_regression_stock','png')































