% Start of script
%-------------------------------------------------------------------------%
close all;                   	% close all figures
clearvars;                      % clear all variables
clc;                         	% clear the command terminal
format shortG;                 	% picks most compact numeric display
format compact;                	% suppress excess blank lines

% Vectors
%-------------------------------------------------------------------------%
v = [1 2 3]

v = [1; 2; 3]

u = [1 1 1];
v = [2 2 2];

result_1 = u + v

result_1 = u - v

c = 2

result_1 = c * v

result_1 = dot(u,v)

result_2 = u*v'

% Matricies
%-------------------------------------------------------------------------%
A = [1 2 3;
     4 5 6]
 
B = [1 2 1;
     2 1 2]
 
result_1 = A + B

s = 2;

result_1 = s*A

result_2 = A*B'

% Laws of Matrix Algebra
%-------------------------------------------------------------------------%
A = [1 2 3; 4 5 6; 7 8 9]

B = [2 2 2; 2 2 2; 2 2 2]
 
C = [5 5 5; 5 5 5; 5 5 5]
 
% Associative
%-------------------------%
result_1 = (A+B)+C
result_2 = A+(B+C)

result_1 = (A*B)*C
result_2 = A*(B*C)

% Commutative
%-------------------------%
result_1 = A+B
result_2 = B+A

% Distributive
%-------------------------%
result_1 = A*(B+C)
result_2 = A*B + A*C

result_1 = (A+B)*C
result_2 = A*C + B*C

% Order Matters!
%-------------------------%
result_1 = A*B
result_2 = B*A


% Transpose
%-------------------------------------------------------------------------%
A = [1 2 3; 4 5 6; 7 8 9]
A'

a = [1 2 3]
a'


% Properties of the transpose
%-------------------------------------------------------------------------%
result_1 = (A+B)'
result_2 = A' + B'

result_1 = (A')'
result_2 = A

result_1 = (s*A)'
result_2 = s*A'

result_1 = (A*B)'
result_2 = B'*A'


% System of Linear Equations
%-------------------------------------------------------------------------%
% Given:
% x - 3y = -3
% 2x + y = 8
% solve for x and y
% first find A and b
A = [1 -3; 2  1]
b = [-3; 8]

% make augmented matrix
A_bar = [A b]

% put into reduced row echelon form
A_bar = rref(A_bar)

% solve for x and y
% ... or use A^-1 such that x = A^-1 *b
x = inv(A)*b

  
% Properties of the Inverse
%-------------------------------------------------------------------------%
A = [1 1 1; 0 2 3; 5 5 1]
B = [1 0 4; 0 2 0; 0 0 1]

result_1 = A*inv(A)
result_2 = inv(A)*A

result_1 = inv(inv(A))

result_1 = inv(A*B)
result_2 = inv(B)*inv(A)


% Linear Systems and Inverses
%-------------------------------------------------------------------------%
b = [10;50;10];
x = inv(A)*b


% Determinants
%-------------------------------------------------------------------------%
det(A)















