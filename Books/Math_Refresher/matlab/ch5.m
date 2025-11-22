% Chapter 5

% Start of script
%-------------------------------------------------------------------------%
close all;                   	% close all figures
clearvars; clearvars -global;	% clear all variables
clc;                         	% clear the command terminal
format shortG;                 	% pick the most compact numeric display
format compact;                	% suppress excess blank lines


% derivative
%-------------------------------------------------------------------------%
% f(x) = 2x
x = -10:10;
f_x = 2.*x;
df_x = 2.*ones(length(x));
figure
subplot(211)
plot(x,f_x,'r','Linewidth',2)
title('f(x) = 2x')
subplot(212)
plot(x,df_x,'r:','Linewidth',2)
title('f ''(x) = 2')

% g(x) = x^3
x = -10:10;
g_x = x.^3;
dg_x = 3.*x.^2;
figure
subplot(211)
plot(x,g_x,'r','Linewidth',2)
title('g(x) = x^3')
subplot(212)
plot(x,dg_x,'r:','Linewidth',2)
title('g ''(x) = 3x^2')

% f(x) = e^x
x = -3:0.01:3;
f_x = exp(x);
df_x = exp(x);
figure
subplot(211)
plot(x,f_x,'r','Linewidth',2)
title('f(x) = e^x')
subplot(212)
plot(x,df_x,'r:','Linewidth',2)
title('f ''(x) = e^x')

% f(x) = log(x)
x = 0:0.01:3;
f_x = log(x);
df_x = 1./x;
figure
subplot(211)
plot(x,f_x,'r','Linewidth',2)
title('f(x) = log(x)')
subplot(212)
plot(x,df_x,'r:','Linewidth',2)
title('f ''(x) = 1/x')


% Taylor series
%-------------------------------------------------------------------------%
syms x
f = sin(x)/x;
t6 = taylor(f, x)
t8 = taylor(f, x, 'Order', 8)
t10 = taylor(f, x, 'Order', 10)
ezplot(t6)
hold on
ezplot(t8)
ezplot(t10)
ezplot(f)

xlim([-6 6])

legend('approximation of sin(x)/x up to O(x^6)',...
'approximation of sin(x)/x up to O(x^8)',...
'approximation of sin(x)/x up to O(x^{10})',...
'sin(x)/x',...
'Location', 'South');

title('Taylor Series Expansion')
hold off



