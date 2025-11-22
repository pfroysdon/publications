% Sequences

% Start of script
%-------------------------------------------------------------------------%
close all;                   	% close all figures
clearvars; clearvars -global;	% clear all variables
clc;                         	% clear the command terminal
format shortG;                 	% pick the most compact numeric display
format compact;                	% suppress excess blank lines

n = 20;
x = 1:n;
An = zeros(n,1);
Bn = zeros(n,1);
Cn = zeros(n,1);

for ii = 1:20
    An(ii,1) = 2 - (1/ii^2);
    Bn(ii,1) = (ii^2 +1)/ii;
    Cn(ii,1) = (-1)^ii * (1-(1/ii));
end
figure(1)
subplot(1,3,1)
    plot(x,An,'.','LineWidth',1)
    hold on
    plot(x,An,'b')
    xlabel('$$n$$','Interpreter','latex')
    ylabel('$$A_n$$','Interpreter','latex')
subplot(1,3,2)
    plot(x,Bn,'.','LineWidth',1)
    hold on
    plot(x,Bn,'b')
    xlabel('$$n$$','Interpreter','latex')
    ylabel('$$B_n$$','Interpreter','latex')
subplot(1,3,3)
    plot(x,Cn,'.','LineWidth',1)
    hold on
    plot(x,Cn,'b')
    xlabel('$$n$$','Interpreter','latex')
    ylabel('$$C_n$$','Interpreter','latex')

save_all_figs_OPTION('../figures/sequences','pdf')