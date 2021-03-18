% Derivative of exp

% Start of script
%-------------------------------------------------------------------------%
close all;                   	% close all figures
clearvars; clearvars -global;	% clear all variables
clc;                         	% clear the command terminal
format shortG;                 	% pick the most compact numeric display
format compact;                	% suppress excess blank lines

%f(x) = e^x
%f'(x) = e^x
x1 = -2:0.1:2;
fx1 = zeros(length(x1),1);
dx1 = zeros(length(x1),1);
for ii = 1:length(x1)
    fx1(ii,1) = exp(x1(ii));
    dx1(ii,1) = exp(x1(ii));
end


figure(1)
subplot(1,2,1)
    plot(x1,fx1,'.','LineWidth',2)
    hold on
    plot(x1,fx1,'b')
    xlabel('$$x$$','Interpreter','latex')
    ylabel('$$f(x)$$','Interpreter','latex')
    title('$$f(x) = e^x$$','Interpreter','latex')
subplot(1,2,2)
    plot(x1,dx1,'.','LineWidth',2)
    hold on
    plot(x1,dx1,'b')
    xlabel('$$x$$','Interpreter','latex')
    ylabel('$$f''(x)$$','Interpreter','latex')
    title('$$f''(x) = e^x$$','Interpreter','latex')
    
save_all_figs_OPTION('../figures/derivative_exp','pdf')