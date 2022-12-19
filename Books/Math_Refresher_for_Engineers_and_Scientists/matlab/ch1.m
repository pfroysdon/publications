% Chapter 1

% Start of script
%-------------------------------------------------------------------------%
close all;                   	% close all figures
clearvars; clearvars -global;	% clear all variables
clc;                         	% clear the command terminal
format shortG;                 	% pick the most compact numeric display
format compact;                	% suppress excess blank lines


% functions
%-------------------------------------------------------------------------%
% f(x) = x^2
x = -2:0.1:2;
f_x = x.^2 - 1;
figure
subplot(121)
plot(x,f_x,'b.','Linewidth',3)
hold on
plot(x,f_x,'b')
hold off
subplot(122)
plot(f_x,x,'b.','Linewidth',3)
hold on
plot(f_x,x,'b')
hold off
saveas(gcf,'..\figures\functions','pdf')


% f(x) = 0.75x-6
x = -2:0.5:6;
f_x = 0.75.*x - 6;
figure
plot(x,f_x,'b.','Linewidth',3)
hold on
plot(x,f_x,'b')
plot(0,-6,'ro',...
     4,-3,'ro','Linewidth',2)
hold off
saveas(gcf,'..\figures\simple_line','pdf')


% f(x) = sin(x)./x
x = -6:0.1:6;
f_x = sin(x)./x;
figure
plot(x,f_x,'b.','Linewidth',3)
hold on
plot(x,f_x,'b')
hold off
saveas(gcf,'..\figures\graph13','pdf')


% f(x) = x^2 piecewise
x1 = 0:0.1:4;
f_x1 = -x1.^2+2;
x2 = -4:0.1:0;
f_x2 = x2.^2-2;
figure
plot(f_x1,x1,'b.',...
     f_x2,x2,'b.','Linewidth',3)
hold on
plot(f_x1,x1,'b',...
     f_x2,x2,'b')
plot(2,0,'b*',...
     -2,0,'b*','Linewidth',2)
hold off
saveas(gcf,'..\figures\graph15','pdf')


% f(x) = y = -3x + 2
x = -3:0.1:3;
f_x = -3.*x + 2;
figure
plot(x,f_x,'b.','Linewidth',3)
hold on
plot(x,f_x,'b')
hold off
saveas(gcf,'..\figures\ex1_20_1','pdf')


% f(x) = y = sqrt(2).*x - 3
x = -3:0.1:3;
f_x = sqrt(2).*x - 3;
figure
plot(x,f_x,'b.','Linewidth',3)
hold on
plot(x,f_x,'b')
hold off
saveas(gcf,'..\figures\ex1_20_2','pdf')



% Summation
%-------------------------------------------------------------------------%

% summation "by hand"
%------------------------------------%
n = 10; % set the stopping criteria
x = 1:1:n; % create the vector x
x_i = x(1); % initialize x_i with teh first index
% sum over the remaining indicies
for ii = 2:n
    x_i = x_i + x(ii);
end

result_1 = x_i

% compare to the built-in summation function
result_2 = sum(x)


% summation with a constant "by hand"
%------------------------------------%
n = 10; % set the stopping criteria
c = 2; % some constant
x = 1:1:n; % create the vector x
x_i = c*x(1); % initialize x_i with teh first index
% sum over the remaining indicies
for ii = 2:n
    x_i = x_i + c*x(ii);
end

result_1 = x_i


x = 1:1:n; % create the vector x
x_i = x(1); % initialize x_i with teh first index
% sum over the remaining indicies
for ii = 2:n
    x_i = x_i + x(ii);
end

result_2 = c*x_i


% Products
%-------------------------------------------------------------------------%

% products "by hand"
%------------------------------------%
n = 10; % set the stopping criteria
x = 1:1:n; % create the vector x
x_i = x(1); % initialize x_i with teh first index
% sum over the remaining indicies
for ii = 2:n
    x_i = x_i * x(ii);
end

result_1 = x_i

% compare to the built-in summation function
result_2 = prod(x)


% products with a constant "by hand"
%------------------------------------%
n = 10; % set the stopping criteria
c = 2; % some constant
x = 1:1:n; % create the vector x
x_i = c*x(1); % initialize x_i with teh first index
% sum over the remaining indicies
for ii = 2:n
    x_i = x_i * c*x(ii);
end

result_1 = x_i


x = 1:1:n; % create the vector x
x_i = x(1); % initialize x_i with teh first index
% sum over the remaining indicies
for ii = 2:n
    x_i = x_i * x(ii);
end

result_2 = c^n * x_i


% Factorials
%-------------------------------------------------------------------------%

x = 5;

% factorial
x_fac = x; % initialize
for ii = 1:x-1
    x_fac = x_fac * (x-ii);
end

result_1 = x_fac

% compare to the built-in function
result_2 = factorial(x)


% Modulo
%-------------------------------------------------------------------------%

result = mod(100,30)

result = mod(14,4)


% Log and exp
%-------------------------------------------------------------------------%

% examples
%------------------------------------%
x = 10;

% this will give us y = log_10(x)
result_1 = log10(x)

% this will give us 10^y = x
result_2 = 10^(log10(x))

% this will give us y = log_e(x)
result_1 = exp(log(x))

% this will give us e^y = x
result_2 = exp(log(x))

% properties of exponents
%------------------------------------%
a = 5;
x = 2;
y = 3;

% prop 1
result_1 = a^x * a^y
result_2 = a^(x+y)

% prop 2
result_1 = a^(-x)
result_2 = 1/(a^x)

% prop 3
result_1 = a^x / a^y
result_2 = a^(x-y)

% prop 4
result_1 = (a^x)^y
result_2 = a^(x*y)

% prop 5
result_1 = a^0


% properties of logs
%------------------------------------%
x = 2;
y = 3;

% prop 1
result_1 = log(x*y)
result_2 = log(x) + log(y)

% prop 2
result_1 = log(x^y)
result_2 = y*log(x)

% prop 3
result_1 = log(1/x)
result_2 = -log(x)

% prop 4
result_1 = log(x/y)
result_2 = log(x) - log(y)

% prop 5
result_1 = log(1)


% quadratic formula
%-------------------------------------------------------------------------%
% ax^2 + bx + c = 0

% example 2
% x^2 + 3x - 4 = 0
a = 1;
b = 3;
c = -4;

x_p = (-b + sqrt(b^2 - 4*a*c))/(2*a)
x_m = (-b - sqrt(b^2 - 4*a*c))/(2*a)


% Functions
%-------------------------------------------------------------------------%

% Paraboloid of One Sheet
%------------------------------------%
% Declare some constants.
a=2;
b=2;
c=1;
numPts = 30;
z_shift = 1; 
scale = 1;

% Define range of x and y axes.
x_0 = linspace(-2*a, 2*a, numPts);
y_0 = linspace(-2*b, 2*b, numPts);
[x, y] = meshgrid(x_0, y_0);

% Construct function.
% z = scale * sqrt(z_shift + x.^2 / a^2 + y.^2 / b^2);
z = scale * (z_shift + x.^2 / a^2 + y.^2 / b^2);
z = reshape(z, [], length(y_0));

figure;
surf(z);
hold on;
contour(z);
hold off;
grid on;
title('Paraboloid of One Sheet')
xlabel x
ylabel y
zlabel z


% Elliptical Paraboloid
%------------------------------------%
[r,theta] = meshgrid([0:0.05:5],[0:pi/50:2*pi]);

x = r.*cos(theta);
y = r.*sin(theta);
z = x.^2 + y.^2;

figure;
surf(x,y,z)
hold on;
contour(x,y,z);
hold off;
title('Elliptical Paraboloid')
xlabel x
ylabel y
zlabel z


% Hyperbolic Paraboloid
%------------------------------------%
x = [-10:.5:10];
y=[-10:.5:10];

[X, Y] = meshgrid(x,y);

Z = X.^2-Y.^2;

figure;
surf(X,Y,Z)
title('Hyperbolic Paraboloid')
xlabel y
ylabel x
zlabel z




