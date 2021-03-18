% Concave and Convex

% Start of script
%-------------------------------------------------------------------------%
close all;                   	% close all figures
clearvars; clearvars -global;	% clear all variables
clc;                         	% clear the command terminal
format shortG;                 	% pick the most compact numeric display
format compact;                	% suppress excess blank lines

%f(x) = x^2
%dx(x) = -x^2
x = -4:0.1:4;
fx1 = zeros(length(x),1);
fx2 = zeros(length(x),1);
for ii = 1:length(x)
    fx1(ii,1) = -x(ii)^2;
    fx2(ii,1) = x(ii)^2;
end


figure(1)
subplot(1,2,1)
    plot(x,fx1,'.','LineWidth',2)
    hold on
    plot(x,fx1,'b')
    xlabel('$$x$$','Interpreter','latex')
    ylabel('$$f(x)$$','Interpreter','latex')
    title('Concave','Interpreter','latex')
subplot(1,2,2)
    plot(x,fx2,'.','LineWidth',2)
    hold on
    plot(x,fx2,'b')
    xlabel('$$x$$','Interpreter','latex')
    ylabel('$$f(x)$$','Interpreter','latex')
    title('Convex','Interpreter','latex')
    
save_all_figs_OPTION('../figures/concave_convex','pdf')