% normal distribution


% Start of script
%-------------------------------------------------------------------------%
close all;                   	% close all figures
clearvars; clearvars -global;	% clear all variables
clc;                         	% clear the command terminal
format shortG;                 	% pick the most compact numeric display
format compact;                	% suppress excess blank lines

%f(x) = randn(x)
rng default; % For reproducibility
r1 = normrnd(0,1,1000,1);
r2 = normrnd(0,2,1000,1);
h = histfit(r1);
x1 = h(2, 1).XData;
y1 = h(2, 1).YData;
h = histfit(r2);
x2 = h(2, 1).XData;
y2 = h(2, 1).YData;
close;

figure(1)
plot(x1,y1,':',x2,y2,'-.','LineWidth',2)
xlabel('$$x$$','Interpreter','latex')
ylabel('$$f(x)$$','Interpreter','latex')
title('$$f(x)$$','Interpreter','latex')
legend('$$\mu=0,\sigma^2=1$$','$$\mu=0,\sigma^2=2$$','Interpreter','latex')
xlim([-6 6])

save_all_figs_OPTION('../figures/normal_distribution','pdf')