% Chapter 4

% Start of script
%-------------------------------------------------------------------------%
close all;                   	% close all figures
clearvars; clearvars -global;	% clear all variables
clc;                         	% clear the command terminal
format shortG;                 	% pick the most compact numeric display
format compact;                	% suppress excess blank lines

% Central Limit Theorem
%-------------------------------------------------------------------------%
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
figure
for ii = 1:n
    plot(trials(1:ii),P_heads(1:ii))
    ylim([0 1]);
    xlabel('Trial Number')
    ylabel('Probabilty of Heads after n trials')
    title ('Trial Number vs Percent Heads')
    drawnow;
end

%% Ex 4.1
%-------------------------------------------------------------------------%

n = 1:50;

% 1
A_n = 2 - 1./(x.^2);
figure
for ii = n
    plot(n(1:ii),A_n(1:ii))
    ylim([1 2])
    xlabel('Trial Number')
    ylabel('A_n')
    title ('Trial Number vs A_n')
    drawnow;
end

% 2
B_n = (n.^2 + 1)./n;
figure
for ii = n
    plot(n(1:ii),B_n(1:ii))
    ylim([0 60])
    xlabel('Trial Number')
    ylabel('B_n')
    title ('Trial Number vs B_n')
    drawnow;
end

% 3
C_n = (-1).^n .* (1-(1./n));
figure
for ii = n
    plot(n(1:ii),C_n(1:ii))
    ylim([-1 1])
    xlabel('Trial Number')
    ylabel('C_n')
    title ('Trial Number vs C_n')
    drawnow;
end


%% Fig 4.3
%-------------------------------------------------------------------------%

x = -50:1:50;

% f(x) = sqrt(x)
f_x = real(sqrt(x));
figure
for ii = 1:length(x)
    plot(x(1:ii),f_x(1:ii))
    ylim([0 10])
    xlabel('Trial Number')
    ylabel('f(x)')
    title ('Trial Number vs f(x)')
    drawnow;
end

% f(x) = 1/x
f_x = 1 ./ x;
figure
for ii = 1:length(x)
    plot(x(1:ii),f_x(1:ii))
    ylim([-1 1])
    xlabel('Trial Number')
    ylabel('f(x)')
    title ('Trial Number vs f(x)')
    drawnow;
end


%% Ex 4.4
%-------------------------------------------------------------------------%

% f(x) = sqrt(x)
x = 1:50;
f_x = sqrt(x);
figure
for ii = 1:length(x)
    plot(x(1:ii),f_x(1:ii))
    ylim([0 10])
    xlabel('Trial Number')
    ylabel('f(x)')
    title ('Trial Number vs f(x) = sqrt(x)')
    drawnow;
end


% f(x) = e^x
x = -5:1:5;
f_x = exp(x);
figure
for ii = 1:length(x)
    plot(x(1:ii),f_x(1:ii))
    ylim([0 150])
    xlabel('Trial Number')
    ylabel('f(x)')
    title ('Trial Number vs f(x) = e^x')
    drawnow;
end


% f(x) = 1+(1/x^2)
x = -2:0.05:2;
f_x = 1+(1 ./x.^2);
figure
for ii = 1:length(x)
    plot(x(1:ii),f_x(1:ii))
    ylim([0 150])
    xlabel('Trial Number')
    ylabel('f(x)')
    title ('Trial Number vs f(x) = 1+(1/x^2)')
    drawnow;
end


% f(x) = floor(x)
x = 0:0.05:5;
f_x = floor(x);
figure
for ii = 1:length(x)
    plot(x(1:ii),f_x(1:ii),'*')
    ylim([0 5])
    xlabel('Trial Number')
    ylabel('f(x)')
    title ('Trial Number vs f(x) = floor(x)')
    drawnow;
end
