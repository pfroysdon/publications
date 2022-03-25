% Limits

% Start of script
%-------------------------------------------------------------------------%
close all;                   	% close all figures
clearvars; clearvars -global;	% clear all variables
clc;                         	% clear the command terminal
format shortG;                 	% pick the most compact numeric display
format compact;                	% suppress excess blank lines

x = -20:1:20;
f1 = zeros(length(x),1);
f2 = zeros(length(x),1);

for ii = 1:length(x)
    f1(ii,1) = sqrt(x(ii));
    f2(ii,1) = 1/x(ii);
end

figure(1)
subplot(1,2,1)
    plot(x,f1,'.','LineWidth',1)
    hold on
    plot(x,f1,'b')
    xlabel('$$x$$','Interpreter','latex')
    ylabel('$$f(x)$$','Interpreter','latex')
    title('$$f(x) = \sqrt x$$','Interpreter','latex')
subplot(1,2,2)
    plot(x,f2,'.','LineWidth',1)
    hold on
    plot(x,f2,'b')
    xlabel('$$x$$','Interpreter','latex')
    ylabel('$$f(x)$$','Interpreter','latex')
    title('$$f(x) = \frac{1}{x}$$','Interpreter','latex')

save_all_figs_OPTION('../figures/limits','pdf')