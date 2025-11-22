% Indefinite Integral

% Start of script
%-------------------------------------------------------------------------%
close all;                   	% close all figures
clearvars; clearvars -global;	% clear all variables
clc;                         	% clear the command terminal
format shortG;                 	% pick the most compact numeric display
format compact;                	% suppress excess blank lines

%f(x) = x^2 - 4
%F1(x) = (1/3)x^3 - 4x
%F2(x) = (1/3)x^3 - 4x + 1
%F3(x) = (1/3)x^3 - 4x - 1
x = -4:0.1:4;
fx = zeros(length(x),1);
Fx1 = zeros(length(x),1);
Fx2 = zeros(length(x),1);
Fx3 = zeros(length(x),1);
for ii = 1:length(x)
    fx(ii,1) = x(ii)^2 - 4;
    Fx1(ii,1) = (1/3)*x(ii)^3 - 4*x(ii);
    Fx2(ii,1) = (1/3)*x(ii)^3 - 4*x(ii) + 1;
    Fx3(ii,1) = (1/3)*x(ii)^3 - 4*x(ii) - 1;
end


figure(1)
subplot(2,1,1)
    plot(x,fx,'.','LineWidth',2)
    hold on
    plot(x,fx,'b')
    xlabel('$$x$$','Interpreter','latex')
    ylabel('$$f(x)$$','Interpreter','latex')
    title('$$f(x) = x^2 - 4$$','Interpreter','latex')
subplot(2,1,2)
    plot(x,Fx1,':',x,Fx2,'-.',x,Fx3,'--','LineWidth',2)
    xlabel('$$x$$','Interpreter','latex')
    ylabel('$$\int f(x) dx$$','Interpreter','latex')
    title('$$\int f(x) dx = \frac{1}{3}x^3 - 4x + \{-1,0,1\}$$','Interpreter','latex')
    
save_all_figs_OPTION('../figures/indefinite_integral','pdf')