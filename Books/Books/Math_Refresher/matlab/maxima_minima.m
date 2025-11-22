% Maxima and Minima

% Start of script
%-------------------------------------------------------------------------%
close all;                   	% close all figures
clearvars; clearvars -global;	% clear all variables
clc;                         	% clear the command terminal
format shortG;                 	% pick the most compact numeric display
format compact;                	% suppress excess blank lines

%f(x) = x^2 + 2
%dx(x) = 2x
x = -4:0.1:4;
fx = zeros(length(x),1);
dx = zeros(length(x),1);
for ii = 1:length(x)
    fx(ii,1) = x(ii)^2  + 2;
    dx(ii,1) = 2*x(ii);
end


figure(1)
subplot(1,2,1)
    plot(x,fx,'.','LineWidth',2)
    hold on
    plot(x,fx,'b')
    xlabel('$$x$$','Interpreter','latex')
    ylabel('$$f(x)$$','Interpreter','latex')
    title('$$f(x) = x^2 + 2$$','Interpreter','latex')
subplot(1,2,2)
    plot(x,dx,'.','LineWidth',2)
    hold on
    plot(x,dx,'b')
    xlabel('$$x$$','Interpreter','latex')
    ylabel('$$f''(x)$$','Interpreter','latex')
    title('$$f''(x) = 2x$$','Interpreter','latex')
    
save_all_figs_OPTION('../figures/maxima_minima','pdf')