% Taylor Series

% Start of script
%-------------------------------------------------------------------------%
close all;                   	% close all figures
clearvars; clearvars -global;	% clear all variables
clc;                         	% clear the command terminal
format shortG;                 	% pick the most compact numeric display
format compact;                	% suppress excess blank lines

syms x
f = sin(x)/x;
t6 = taylor(f, x)
t8 = taylor(f, x, 'Order', 8)
t10 = taylor(f, x, 'Order', 10)

figure(1)
subplot(121)
    fplot(f,'LineWidth',2)
    hold on
    fplot(t6,':','LineWidth',2)
    fplot(t8,'-.','LineWidth',2)
    fplot(t10,'--','LineWidth',2)
    hold off
    xlim([-20 20])
    xlabel('$$x$$','Interpreter','latex')
    ylabel('$$y$$','Interpreter','latex')
    legend('$$\sin(x)/x$$',...
        '$$\mathcal{O}(x^6)$$ approx.',...
        '$$\mathcal{O}(x^8)$$ approx.',...
        '$$\mathcal{O}(x^{10})$$ approx.',...
        'Location', 'South','Interpreter','latex');
    title('Taylor Series Expansion')
subplot(122)
    fplot(f,'LineWidth',2)
    hold on
    fplot(t6,':','LineWidth',2)
    fplot(t8,'-.','LineWidth',2)
    fplot(t10,'--','LineWidth',2)
    hold off
    xlim([-6 6])
    ylim([-0.5 1.5])
    xlabel('$$x$$','Interpreter','latex')
    ylabel('$$y$$','Interpreter','latex')
    legend('$$\sin(x)/x$$',...
        '$$\mathcal{O}(x^6)$$ approx.',...
        '$$\mathcal{O}(x^8)$$ approx.',...
        '$$\mathcal{O}(x^{10})$$ approx.',...
        'Location', 'North','Interpreter','latex');
    title('ZOOM: Taylor Series Expansion')

set(gcf, 'Position',  [300, 300, 800, 400])
    
save_all_figs_OPTION('../figures/taylor_series','pdf')