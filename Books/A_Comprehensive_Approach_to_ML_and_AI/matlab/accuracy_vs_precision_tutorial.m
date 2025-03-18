% Clean up
clear; clc; close all;

% Create a figure sized for clarity
figure('Position',[100 100 1200 600]);

%------------------------------------------------
% Common parameters for circles
%------------------------------------------------
radii = [1,2,3,4,5];    % Radii of concentric circles
angles_for_x = 0;     % Angle(s) where 'X' is placed (0 = top)
nCirclePoints = 200;  % Number of points to plot each circle

%------------------------------------------------
% Subplot (a): top - circles, bottom - distribution
%------------------------------------------------
% -- Top row (a)
subplot(2,4,1); 
hold on; axis equal;
title('(a)');
% Plot concentric circles centered at (0,0)
for r = radii
    [xc, yc] = circle(0,0,r,nCirclePoints);
    plot(xc, yc, 'k'); 
    % % Place an 'X' at specified angle(s)
    % for theta = angles_for_x
    %     xX = r*cosd(theta);
    %     yX = r*sind(theta);
    %     plot(xX, yX, 'kx', 'MarkerSize',10, 'LineWidth',2);
    % end
end
xX = 1.5*cosd(0); yX = 1.5*sind(0);
plot(xX, yX, 'kx', 'MarkerSize',10, 'LineWidth',2);
xX = 4.5*cosd(80); yX = 4.5*sind(80);
plot(xX, yX, 'kx', 'MarkerSize',10, 'LineWidth',2);
xX = 4.5*cosd(150); yX = 4.5*sind(150);
plot(xX, yX, 'kx', 'MarkerSize',10, 'LineWidth',2);
xX = 2.5*cosd(200); yX = 2.5*sind(200);
plot(xX, yX, 'kx', 'MarkerSize',10, 'LineWidth',2);
xX = 4.5*cosd(300); yX = 4.5*sind(300);
plot(xX, yX, 'kx', 'MarkerSize',10, 'LineWidth',2);

% Plot dashed lines
xline(0, 'r--','LineWidth',1.5);  % Red dashed line at x=0
xline(0.75, 'g--','LineWidth',1.5);  % Green dashed line at x=2
xlim([-5 5]); ylim([-5 5]);
axis square
xlabel('X'); ylabel('Y');
set(gca,'Box','on');

% -- Bottom row (a)
subplot(2,4,5);
hold on;
% Create an example distribution
xvals = linspace(-10,10,200);
mu_a = 0.75;   % mean
sigma_a = 2; % std
yvals_a = normpdf(xvals, mu_a, sigma_a);
plot(xvals, yvals_a, 'k', 'LineWidth',1.5);
% Vertical dashed lines
xline(0,'r--','LineWidth',1.5);
xline(mu_a,'g--','LineWidth',1.5);
title('(a) Distribution');
xlabel('x'); ylabel('PDF');
xlim([-5 5]);
axis square
grid on;

%------------------------------------------------
% Subplot (b): top - circles offset, bottom - distribution
%   (You can change centers, lines, etc. to match your figure)
%------------------------------------------------
% -- Top row (b)
subplot(2,4,2); 
hold on; axis equal;
title('(b)');
% % Suppose we shift the center of the circles slightly, e.g. center=(1,0)
% centerB = [1, 0];
% for r = radii
%     [xc, yc] = circle(centerB(1), centerB(2), r, nCirclePoints);
%     plot(xc, yc, 'k');
%     % 'X' at top of each circle
%     for theta = angles_for_x
%         xX = centerB(1) + r*cosd(theta);
%         yX = centerB(2) + r*sind(theta);
%         plot(xX, yX, 'kx', 'MarkerSize',10, 'LineWidth',2);
%     end
% end
for r = radii
    [xc, yc] = circle(0,0,r,nCirclePoints);
    plot(xc, yc, 'k'); 
    % % Place an 'X' at specified angle(s)
    % for theta = angles_for_x
    %     xX = r*cosd(theta);
    %     yX = r*sind(theta);
    %     plot(xX, yX, 'kx', 'MarkerSize',10, 'LineWidth',2);
    % end
end
xX = 2*cosd(0); yX = 2*sind(0);
plot(xX, yX, 'kx', 'MarkerSize',10, 'LineWidth',2);
xX = 2*cosd(90); yX = 2*sind(90);
plot(xX, yX, 'kx', 'MarkerSize',10, 'LineWidth',2);
xX = 2*cosd(180); yX = 2*sind(180);
plot(xX, yX, 'kx', 'MarkerSize',10, 'LineWidth',2);
xX = 2*cosd(270); yX = 2*sind(270);
plot(xX, yX, 'kx', 'MarkerSize',10, 'LineWidth',2);
xX = 0*cosd(0); yX = 0*sind(0);
plot(xX, yX, 'kx', 'MarkerSize',10, 'LineWidth',2);

% Dashed lines
xline(0, 'r--','LineWidth',1.5);  
xline(0, 'g--','LineWidth',1.5);
xlim([-5 5]); ylim([-5 5]);
xlabel('X'); ylabel('Y');
axis square
set(gca,'Box','on');

% -- Bottom row (b)
subplot(2,4,6);
hold on;
mu_b = 0;      % shift mean
sigma_b = 1; % narrower or broader as needed
yvals_b = normpdf(xvals, mu_b, sigma_b);
plot(xvals, yvals_b, 'k', 'LineWidth',1.5);
xline(0,'r--','LineWidth',1.5);
xline(mu_b,'g--','LineWidth',1.5);
title('(b) Distribution');
xlabel('x'); ylabel('PDF');
xlim([-5 5]);
axis square
grid on;

%------------------------------------------------
% Subplot (c)
%------------------------------------------------
% -- Top row (c)
subplot(2,4,3); 
hold on; axis equal;
title('(c)');
% % Maybe shift circles far left
% centerC = [-3, 0];
% for r = radii
%     [xc, yc] = circle(centerC(1), centerC(2), r, nCirclePoints);
%     plot(xc, yc, 'k');
%     for theta = angles_for_x
%         xX = centerC(1) + r*cosd(theta);
%         yX = centerC(2) + r*sind(theta);
%         plot(xX, yX, 'kx', 'MarkerSize',10, 'LineWidth',2);
%     end
% end
for r = radii
    [xc, yc] = circle(0,0,r,nCirclePoints);
    plot(xc, yc, 'k'); 
    % % Place an 'X' at specified angle(s)
    % for theta = angles_for_x
    %     xX = r*cosd(theta);
    %     yX = r*sind(theta);
    %     plot(xX, yX, 'kx', 'MarkerSize',10, 'LineWidth',2);
    % end
end
xX = 4.5*cosd(100); yX = 4.5*sind(100);
plot(xX, yX, 'kx', 'MarkerSize',10, 'LineWidth',2);
xX = 4.5*cosd(102); yX = 4.5*sind(102);
plot(xX, yX, 'kx', 'MarkerSize',10, 'LineWidth',2);
xX = 4.5*cosd(104); yX = 4.5*sind(104);
plot(xX, yX, 'kx', 'MarkerSize',10, 'LineWidth',2);
xX = 4.5*cosd(106); yX = 4.5*sind(106);
plot(xX, yX, 'kx', 'MarkerSize',10, 'LineWidth',2);
xX = 4.5*cosd(108); yX = 4.5*sind(108);
plot(xX, yX, 'kx', 'MarkerSize',10, 'LineWidth',2);

xline(0,'r--','LineWidth',1.5);
xline(-1.2,'g--','LineWidth',1.5);
xlim([-5 5]); ylim([-5 5]);
xlabel('X'); ylabel('Y');
axis square
set(gca,'Box','on');

% -- Bottom row (c)
subplot(2,4,7);
hold on;
mu_c = -1.2;    
sigma_c = 0.2;  
yvals_c = normpdf(xvals, mu_c, sigma_c);
plot(xvals, yvals_c, 'k','LineWidth',1.5);
xline(0,'r--','LineWidth',1.5);
xline(mu_c,'g--','LineWidth',1.5);
title('(c) Distribution');
xlabel('x'); ylabel('PDF');
xlim([-5 5]);
axis square
grid on;

%------------------------------------------------
% Subplot (d)
%------------------------------------------------
% -- Top row (d)
subplot(2,4,4); 
hold on; axis equal;
title('(d)');
% Circles all centered at (0,0) again
for r = radii
    [xc, yc] = circle(0,0,r,nCirclePoints);
    plot(xc, yc, 'k');
    % for theta = angles_for_x
    %     xX = r*cosd(theta);
    %     yX = r*sind(theta);
    %     plot(xX, yX, 'kx', 'MarkerSize',10, 'LineWidth',2);
    % end
end

xX = 0.5*cosd(0); yX = 0.5*sind(0);
plot(xX, yX, 'kx', 'MarkerSize',10, 'LineWidth',2);
xX = 0.5*cosd(90); yX = 0.5*sind(90);
plot(xX, yX, 'kx', 'MarkerSize',10, 'LineWidth',2);
xX = 0.5*cosd(180); yX = 0.5*sind(180);
plot(xX, yX, 'kx', 'MarkerSize',10, 'LineWidth',2);
xX = 0.5*cosd(270); yX = 0.5*sind(270);
plot(xX, yX, 'kx', 'MarkerSize',10, 'LineWidth',2);
xX = 0*cosd(0); yX = 0*sind(0);
plot(xX, yX, 'kx', 'MarkerSize',10, 'LineWidth',2);

% Both lines coincide (as if red & green at x=0)
xline(0, 'r--','LineWidth',1.5);
xline(0, 'g--','LineWidth',1.5);
xlim([-5 5]); ylim([-5 5]);
xlabel('X'); ylabel('Y');
axis square
set(gca,'Box','on');

% -- Bottom row (d)
subplot(2,4,8);
hold on;
% Very narrow distribution around 0
mu_d = 0;
sigma_d = 0.2;
yvals_d = normpdf(xvals, mu_d, sigma_d);
plot(xvals, yvals_d, 'k','LineWidth',1.5);
xline(0,'r--','LineWidth',1.5);
xline(0,'g--','LineWidth',1.5);
title('(d) Distribution');
xlabel('x'); ylabel('PDF');
xlim([-5 5]);
axis square
grid on;

% save_all_figs_OPTION('results/accuracy_vs_precision','png',1)

% --------------------------------------------------
% Helper function: returns x,y points of a circle
% centered at (x0, y0) with radius r
% --------------------------------------------------
function [xC, yC] = circle(x0, y0, r, N)
    if nargin<4
        N = 200; % default resolution
    end
    theta = linspace(0,2*pi,N);
    xC = x0 + r*cos(theta);
    yC = y0 + r*sin(theta);
end
