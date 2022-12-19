% Continuous and Discontinuous Functions

% Start of script
%-------------------------------------------------------------------------%
close all;                   	% close all figures
clearvars; clearvars -global;	% clear all variables
clc;                         	% clear the command terminal
format shortG;                 	% pick the most compact numeric display
format compact;                	% suppress excess blank lines

%f(x) = sqrt(x)
x1 = 0:0.1:10;
f1 = zeros(length(x1),1);
for ii = 1:length(x1)
    f1(ii,1) = sqrt(x1(ii));
end

%f(x) = e^x
x2 = -2:0.1:2;
f2 = zeros(length(x2),1);
for ii = 1:length(x2)
    f2(ii,1) = exp(x2(ii));
end

%f(x) = 1 + (1/x^2)
x3 = -4:0.1:4;
f3 = zeros(length(x3),1);
for ii = 1:length(x3)
    f3(ii,1) = 1 + (1/x3(ii)^2);
end

%f(x) = floor(x)
x4 = 0:0.1:5.9;
f4 = zeros(length(x4),1);
for ii = 1:length(x4)
    f4(ii,1) = floor(x4(ii));
end

figure(1)
subplot(2,2,1)
    plot(x1,f1,'.','LineWidth',2)
    hold on
    plot(x1,f1,'b')
    xlabel('$$x$$','Interpreter','latex')
    ylabel('$$f(x)$$','Interpreter','latex')
    title('$$f(x) = \sqrt x$$','Interpreter','latex')
subplot(2,2,2)
    plot(x2,f2,'.','LineWidth',2)
    hold on
    plot(x2,f2,'b')
    xlabel('$$x$$','Interpreter','latex')
    ylabel('$$f(x)$$','Interpreter','latex')
    title('$$f(x) = e^{x}$$','Interpreter','latex')
subplot(2,2,3)
    plot(x3,f3,'.','LineWidth',2)
    hold on
    plot(x3,f3,'b')
    xlabel('$$x$$','Interpreter','latex')
    ylabel('$$f(x)$$','Interpreter','latex')
    title('$$f(x) = 1 + \frac{1}{x^2}$$','Interpreter','latex')
subplot(2,2,4)
    plot(x4,f4,'.','LineWidth',1)
    xlabel('$$x$$','Interpreter','latex')
    ylabel('$$f(x)$$','Interpreter','latex')
    title('$$f(x) = floor(x)$$','Interpreter','latex')
    
save_all_figs_OPTION('../figures/cont_and_disc_func','pdf')