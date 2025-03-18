% This script solves a Traveling Salesman Problem (TSP) using a MILP 
% formulation (MTZ formulation) with MATLAB's intlinprog.
%
% The TSP is defined on a set of cities (with (x,y) coordinates). The goal
% is to find the shortest route that visits each city exactly once and 
% returns to the starting city.

close all; clear all; clc;

% 1. Define the Problem Data: City Coordinates
% For demonstration, we use 10 cities with random coordinates.
numCities = 10;
rng(1); % For reproducibility
cityCoords = rand(numCities,2) * 100;  % Cities in a 100x100 region

% Plot the cities
figure;
scatter(cityCoords(:,1), cityCoords(:,2), 100, 'filled');
title('City Locations');
xlabel('X Coordinate');
ylabel('Y Coordinate');
grid on;
text(cityCoords(:,1)+1, cityCoords(:,2), cellstr(num2str((1:numCities)')), ...
    'FontSize',12, 'Color', 'b');

% save_all_figs_OPTION('results/milp1','png',1)

% 2. Compute the Distance Matrix
% Euclidean distance between cities
D = squareform(pdist(cityCoords));

% Set the diagonal to a very high number to avoid self-loops (we force these to zero later)
for i = 1:numCities
    D(i,i) = 0;
end

% 3. Formulate the MILP Using the MTZ Formulation
%
% Decision Variables:
%   x(i,j) = 1 if the tour goes directly from city i to city j, 0 otherwise.
%   u(i) are continuous variables for subtour elimination (for cities 2...n).
%
% Total variables:
%   - x: numCities*numCities binary variables
%   - u: (numCities - 1) continuous variables (we fix u(1)=0)
%
% The variable vector is:
%   z = [ x(1,1) ... x(1,numCities) x(2,1) ... x(numCities,numCities) u(2) ... u(numCities) ]'

n = numCities;
numX = n * n;
numU = n - 1;
numVars = numX + numU;

% Indices for decision variables
xInd = 1:numX;
uInd = numX+1 : numVars;

% Objective Function: Minimize total distance
% f = [vec(D); zeros(numU,1)]
f = [D(:); zeros(numU,1)];

% Binary variables: x are binary; u are continuous.
intcon = xInd;  % Only x variables are integer (binary)

% Lower and upper bounds:
% For x: 0 <= x <= 1
% For u: 1 <= u <= n-1 (for cities 2...n)
lb = zeros(numVars,1);
ub = ones(numVars,1);
lb(uInd) = 1;
ub(uInd) = n - 1;

% 3a. Equality Constraints: 
% Each city must have exactly one outgoing edge and one incoming edge.
% For each city i: sum_{j=1}^n x(i,j) = 1.
% For each city j: sum_{i=1}^n x(i,j) = 1.

Aeq = [];
beq = [];

% Outgoing constraints: For each city i
Aeq_out = zeros(n, numVars);
for i = 1:n
    idx = (i-1)*n + (1:n);
    Aeq_out(i, idx) = 1;
end
beq_out = ones(n,1);

% Incoming constraints: For each city j
Aeq_in = zeros(n, numVars);
for j = 1:n
    idx = j:n:numX;
    Aeq_in(j, idx) = 1;
end
beq_in = ones(n,1);

Aeq = [Aeq_out; Aeq_in];
beq = [beq_out; beq_in];

% Force no self-loops: x(i,i) = 0 for all i.
Aeq_self = zeros(n, numVars);
for i = 1:n
    Aeq_self(i, (i-1)*n + i) = 1;
end
beq_self = zeros(n,1);

Aeq = [Aeq; Aeq_self];
beq = [beq; beq_self];

% 3b. Inequality Constraints: Subtour Elimination (MTZ constraints)
% For cities 2...n (i and j from 2 to n):
%   u(i) - u(j) + n*x(i,j) <= n-1, for i ~= j.
%
% There will be (n-1)*(n-1) constraints.
numMTZ = (n-1)*(n-1);
Aineq = zeros(numMTZ, numVars);
bineq = (n - 1) * ones(numMTZ,1);
constraintCount = 0;
for i = 2:n
    for j = 2:n
        if i ~= j
            constraintCount = constraintCount + 1;
            % u(i) - u(j) + n*x(i,j) <= n - 1
            % Map u(i): index = numX + (i-1)  (because u(1) is not used; we fix u(1)=0)
            % Map u(j): index = numX + (j-1)
            Aineq(constraintCount, numX + (i-1)) = 1;
            Aineq(constraintCount, numX + (j-1)) = -1;
            % x(i,j) where i and j in {2,...,n} are at index: (i-1)*n + j
            Aineq(constraintCount, (i-1)*n + j) = n;
        end
    end
end

% 4. Solve the MILP using intlinprog
options = optimoptions('intlinprog','Display','iter','MaxTime',300);
[z, fval, exitflag, output] = intlinprog(f, intcon, Aineq, bineq, Aeq, beq, lb, ub, options);

if exitflag ~= 1
    error('The solver did not find an optimal solution.');
end

% Extract the solution for x and reshape to matrix form
x_sol = z(xInd);
X = reshape(x_sol, n, n);

% For clarity, force a numerical tolerance for binary decisions.
X = round(X);

fprintf('Optimal tour length: %.2f\n', fval);
disp('Decision variable matrix X (rows: from city, columns: to city):');
disp(X);

% 5. Extract the Tour from the Decision Matrix X
tour = zeros(n+1,1);
currentCity = 1;  % Start at city 1
tour(1) = currentCity;
for k = 2:n
    nextCity = find(X(currentCity, :) == 1);
    tour(k) = nextCity;
    currentCity = nextCity;
end
% Return to the starting city
tour(n+1) = tour(1);

fprintf('Optimal tour: ');
fprintf('%d -> ', tour(1:end-1));
fprintf('%d\n', tour(end));

% 6. Visualization of the Tour
figure;
plot(cityCoords(:,1), cityCoords(:,2), 'ko', 'MarkerSize', 10, 'MarkerFaceColor', 'k');
hold on;
for i = 1:n
    text(cityCoords(i,1)+1, cityCoords(i,2), num2str(i), 'FontSize',15, 'Color', 'b');
end
for i = 1:n
    cityFrom = tour(i);
    cityTo = tour(i+1);
    plot([cityCoords(cityFrom,1), cityCoords(cityTo,1)], ...
         [cityCoords(cityFrom,2), cityCoords(cityTo,2)], 'r-', 'LineWidth',2);
end
title('Optimal TSP Tour');
xlabel('X Coordinate');
ylabel('Y Coordinate');
grid on;
hold off;

% save_all_figs_OPTION('results/milp2','png',1)