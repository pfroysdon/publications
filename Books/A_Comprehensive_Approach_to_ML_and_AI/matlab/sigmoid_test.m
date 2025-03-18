% Start of script
%-------------------------------------------------------------------------%
close all;  clear all; clc; 

% Simulate
%-------------------------------------------------------------------------%   
% neuron
x = (-8:0.1:8);
f = 1 ./ (1 + exp(-x));
figure;
plot(x, f)
xlabel('x')
ylabel('f(x)')
% savefig('sigmoid.png')

% adjusting the weights
w = [0.5,1.0,2.0];
l1 = 'w = 0.5';
l2 = 'w = 1.0';
l3 = 'w = 2.0';
figure;
for i=1:3
	f = 1 ./ (1 + exp(-x.*w(i)));
	plot(x, f);
    hold on
end
xlabel('x')
ylabel('h_w(x)')
legend(l1,l2,l3)
% savefig('Weight-adjustment-example.png')

% Effect of Bias
w = 5.0;
b = [-8.0,0.0,8.0];
l1 = 'b = -8.0';
l2 = 'b = 0.0';
l3 = 'b = 8.0';
figure;
for i=1:3
	f = 1 ./ (1 + exp(-(x*w+b(i))));
	plot(x, f);
    hold on
end
xlabel('x')
ylabel('h_wb(x)')
legend(l1,l2,l3)
% savefig('Bias-adjustment-example.png')