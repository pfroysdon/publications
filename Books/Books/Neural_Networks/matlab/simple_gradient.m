% Start of script
%-------------------------------------------------------------------------%
close all;  clear all; clc; 

% Simulate
%-------------------------------------------------------------------------%   
x_old = 0; % The value does not matter as long as abs(x_new - x_old) > precision
x_new = 6; % The algorithm starts at x=6
gamma = 0.01; % step size
precision = 0.00001;

function y=f(x)
    y = x^4 - 3*x^2 + 2;
end
    
function y=df(x)
	y = 4 * x^3 - 9 * x^2;
end

while abs(x_new - x_old) > precision
	x_old = x_new;
	x_new = x_new - gamma * df(x_old);
end

fprintf('The local minimum occurs at %1.0f\n', x_new)

n = 10;
x = linspace(1, 10, n);
y = zeros(1,n);
for i =1:n
    y(i) = f(x(i));
end
 
figure(1)    
plot(x, y, 'o-', x_new, f(x_new), 'r*')
legend('f(x)','optimal f(x)')
title('f(x) = x^4 - 3x^2 + 2')
% savefig('gradient_descent.png')