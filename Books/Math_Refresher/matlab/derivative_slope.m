% Derivative as a slope

% Start of script
%-------------------------------------------------------------------------%
close all;                   	% close all figures
clearvars; clearvars -global;	% clear all variables
clc;                         	% clear the command terminal
format shortG;                 	% pick the most compact numeric display
format compact;                	% suppress excess blank lines

%f(x) = 2(x)
%f'(x) = x
x1 = -2:0.1:2;
fx1 = zeros(length(x1),1);
dx1 = zeros(length(x1),1);
for ii = 1:length(x1)
    fx1(ii,1) = 2*x1(ii);
    dx1(ii,1) = 2;
end

%f(x) = x^3
%f'(x) = 3x^2
x2 = -2:0.1:2;
fx2 = zeros(length(x2),1);
dx2 = zeros(length(x2),1);
for ii = 1:length(x2)
    fx2(ii,1) = x2(ii)^3;
    dx2(ii,1) = 3*x2(ii)^2;
end

figure(1)
subplot(2,2,1)
    plot(x1,fx1,'.','LineWidth',2)
    hold on
    plot(x1,fx1,'b')
    xlabel('$$x$$','Interpreter','latex')
    ylabel('$$f(x)$$','Interpreter','latex')
    title('$$f(x) = 2x$$','Interpreter','latex')
subplot(2,2,2)
    plot(x1,dx1,'.','LineWidth',2)
    hold on
    plot(x1,dx1,'b')
    xlabel('$$x$$','Interpreter','latex')
    ylabel('$$f''(x)$$','Interpreter','latex')
    title('$$f''(x) = 2$$','Interpreter','latex')
subplot(2,2,3)
    plot(x2,fx2,'.','LineWidth',2)
    hold on
    plot(x2,fx2,'b')
    xlabel('$$x$$','Interpreter','latex')
    ylabel('$$g(x)$$','Interpreter','latex')
    title('$$g(x) = x^3$$','Interpreter','latex')
subplot(2,2,4)
    plot(x2,dx2,'.','LineWidth',2)
    hold on
    plot(x2,dx2,'b')
    xlabel('$$x$$','Interpreter','latex')
    ylabel('$$g''(x)$$','Interpreter','latex')
    title('$$g''(x) = 3x^2$$','Interpreter','latex')
    
save_all_figs_OPTION('../figures/derivative_slope','pdf')