% Central Limit Theorem

% Start of script
%-------------------------------------------------------------------------%
close all;                   	% close all figures
clearvars; clearvars -global;	% clear all variables
clc;                         	% clear the command terminal
format shortG;                 	% pick the most compact numeric display
format compact;                	% suppress excess blank lines

% load coin flip
coin_flip

% calculate distribution
n = 1000;
% t = rand(n,1);
trials = 1:n;
heads = 0;
P_heads = ones(size(t));
for ii = 1:n
    if (t(ii) < 0.5)
        heads = heads + 1;
    end
    P_heads(ii) = heads/ii;  
end

% animate plot
figure(1)
for ii = 1:n
    plot(trials(1:ii),P_heads(1:ii),'LineWidth',2)
    xlim([-50 1000])
    ylim([0 1]);
    xlabel('Trial Number')
    ylabel('Probabilty of Heads after n trials')
    title ('Trial Number vs Percent Heads')
    drawnow;
end

save_all_figs_OPTION('../figures/law-of-large-numbers','pdf')
