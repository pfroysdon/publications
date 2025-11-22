% Exercise 4.6

% Start of script
%-------------------------------------------------------------------------%
close all;                   	% close all figures
clearvars; clearvars -global;	% clear all variables
clc;                         	% clear the command terminal
format shortG;                 	% pick the most compact numeric display
format compact;                	% suppress excess blank lines

%f(x) = (x^2 + 2x)/(x^2)
x = -4:0.1:4;
fx = zeros(length(x),1);
for ii = 1:length(x)
    %fx(ii,1) = (x(ii) + (2/x(ii)))/(x(ii)^2);
    fx(ii,1) = (x(ii) + (2/x(ii)))/1;
end


figure(1)
plot(x,fx,'.','LineWidth',2)
hold on
plot(x,fx,'b')
xlabel('$$x$$','Interpreter','latex')
ylabel('$$f(x)$$','Interpreter','latex')
title('$$f(x) = \frac{x^2 + 2x}{x^2}$$','Interpreter','latex')
% ylim([-2 2])
    
save_all_figs_OPTION('../figures/ex_3_6','pdf')