close all; clear all; clc; 

SixMPG = [13;15;23;29;32;34];
figure;
histogram(SixMPG)

figure;
pdSix = fitdist(SixMPG,'Kernel','Width',4);
x = 0:.1:45;
ySix = pdf(pdSix,x);
plot(x,ySix,'k-','LineWidth',2)

% Plot each individual pdf and scale its appearance on the plot
hold on
for i=1:6
    pd = makedist('Normal','mu',SixMPG(i),'sigma',4);
    y = pdf(pd,x);
    y = y/6;
    plot(x,y,'b:','LineWidth',2)
end
xlabel('X'); ylabel('Estimated Density');
title('Kernel Density Estimate for Mixture of Gaussians');
hold off

% save_all_figs_OPTION('results/kde2','png',1)


% Create kernel distribution objects
load 'data/carbig.mat'
pd1 = fitdist(MPG,'kernel');
pd2 = fitdist(MPG,'kernel','Width',1);
pd3 = fitdist(MPG,'kernel','Width',5);

% Compute each pdf
x = -10:1:60;
y1 = pdf(pd1,x);
y2 = pdf(pd2,x);
y3 = pdf(pd3,x);

% Plot each pdf
figure;
plot(x,y1,'Color','r','LineStyle','-','LineWidth',2)
hold on
plot(x,y2,'Color','k','LineStyle',':','LineWidth',2)
plot(x,y3,'Color','b','LineStyle','--','LineWidth',2)
legend({'Bandwidth = Default','Bandwidth = 1','Bandwidth = 5'})
xlabel('X'); ylabel('Estimated Density');
title('Kernel Density Estimate for Automotive MPG');
hold off

% save_all_figs_OPTION('results/kde3','png',1)