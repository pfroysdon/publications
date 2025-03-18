close all; clear all; clc;

% Consider the LP: maximize 3x1 + 2x2
% Subject to: x1 + x2 <= 4, x1 <= 2, x2 <= 3, x1,x2 >= 0.
c = [3; 2];
A = [1 1;
     1 0;
     0 1];
b = [4; 2; 3];

[x_opt, obj_val] = simplexMethod(c, A, b);
fprintf('Optimal solution: x1 = %.2f, x2 = %.2f\n', x_opt(1), x_opt(2));
fprintf('Optimal objective value: %.2f\n', obj_val);

% Visualization: Plot the feasible region and optimal point for 2D LP
x1 = linspace(0, 5, 200);
x2_1 = 4 - x1; % x1 + x2 <= 4
x2_2 = 3*ones(size(x1)); % x2 <= 3
x2_3 = 2*ones(size(x1)); % Not used directly since x1<=2 constraint implies x1 in [0,2]
figure;
hold on;

% Feasible region boundaries
plot(x1, x2_1, 'r-', 'LineWidth', 2);
plot([2,2], [0,3], 'b-', 'LineWidth', 2);
plot([0,5], [3,3], 'g-', 'LineWidth', 2);
fill([0, 2, 2, 0], [0, 0, 3, 3], 'y', 'FaceAlpha', 0.3);
plot(x_opt(1), x_opt(2), 'ko', 'MarkerSize', 10, 'MarkerFaceColor', 'k');
title('Feasible Region and Optimal Solution');
xlabel('x1');
ylabel('x2');
legend('x1 + x2 = 4', 'x1 = 2', 'x2 = 3', 'Feasible Region', 'Optimal Solution');
grid on;
hold off;

% save_all_figs_OPTION('results/linearProgramming','png',1)


function [x_opt, obj_val] = simplexMethod(c, A, b)
    % simplexMethod - A basic implementation of the Simplex algorithm.
    % Inputs:
    %   c - Coefficient vector for the objective function (n x 1)
    %   A - Constraint coefficient matrix (m x n)
    %   b - Right-hand side vector (m x 1)
    % Outputs:
    %   x_opt - Optimal solution vector (n x 1)
    %   obj_val - Optimal objective value

    [m, n] = size(A);
    % Formulate standard form by adding slack variables
    A_slack = [A eye(m)];
    c_slack = [c; zeros(m,1)];
    % Initial basic solution: slack variables
    B = n+1:n+m;   % Basic variable indices (for slack variables)
    N = 1:n;       % Nonbasic variable indices (original variables)
    x = zeros(n+m,1);
    x(B) = b;

    % Main loop of the simplex algorithm
    while true
        % Compute simplex multipliers (dual variables)
        lambda = c_slack(B)' / A_slack(:,B);
        % Compute reduced costs
        r = c_slack' - lambda * A_slack;
        % Check for optimality (for maximization, all reduced costs <= 0)
        if all(r(N) <= 1e-6)
            break;
        end
        % Choose entering variable (most positive reduced cost among nonbasic variables)
        [~, j_idx] = max(r(N));
        entering = N(j_idx);

        % Compute ratios for leaving variable using the proper row indexing:
        % Each basic variable corresponds to a row (1 to m)
        ratios = x(B) ./ A_slack(:, entering);
        ratios(A_slack(:, entering) <= 0) = inf;
        [minRatio, minIndex] = min(ratios);
        if isinf(minRatio)
            error('The problem is unbounded.');
        end
        % The leaving variable is B(minIndex)
        leaving = B(minIndex);

        % Pivot: use the row index (minIndex) corresponding to the leaving variable.
        pivot = A_slack(minIndex, entering);
        A_slack(minIndex, :) = A_slack(minIndex, :) / pivot;
        x(leaving) = x(leaving) / pivot;
        for i = [1:minIndex-1, minIndex+1:m]
            factor = A_slack(i, entering);
            A_slack(i, :) = A_slack(i, :) - factor * A_slack(minIndex, :);
        end
        % Update solution vector for basic variables
        x(entering) = x(leaving);
        x(B) = A_slack(:,B) \ b;
        % Update basis indices
        B(minIndex) = entering;
        N(j_idx) = leaving;
    end
    x_opt = x(1:n);
    obj_val = c' * x_opt;
end


