% Expectation & Maximization demo
%
%  Description:
%   EM algorithm for k multidimensional Gaussian mixture estimation.
%
%  Notes:
%   See subfunctions for descriptions.


%-------------------------------------------------------------------------%
% Main function
%-------------------------------------------------------------------------%

% Start of script
close all;                   	% close all figures
clearvars; clearvars -global;	% clear all variables
clc;                         	% clear the command terminal
format shortG;                 	% picks most compact numeric display
format compact;                	% suppress excess blank lines
addpath(genpath('utilities'));  % include local library
startup;                        % set defaults
rng(2)


% Generate dataset
%----------------------------------%
% generate random variables & mixtures
nRand = 2;
X = randn(500, 2)*1 + 2;
for ii = 2:nRand
    X = [X;randn(500, 2)*1 + 4.5];
end

% calculation of E&M
[W,M,V,L] = EM_GM(X,nRand,1e-5,500,1,[]);

% K-Means estimate
opts = statset('Display','off');
[~, ctrs] = kmeans(X,nRand,'Distance','city','Replicates',5,'Options',opts);

% plot raw results
figure
plot(X(:,1),X(:,2),'r.');
xlabel('1^{st} dimension');
ylabel('2^{nd} dimension');
title('Gaussian Mixture');

save_all_figs_OPTION('results/gmm1','png',1)

% plot k-Means and E-M results
Plot_GM(W,M,V,X);
hold on;
for ii = 1:length(ctrs)
    plot(ctrs(ii,1),ctrs(ii,2),'b*','LineWidth',2);
end
hold off;
title('Gaussian Mixture estimated by EM (black) & k-Means (blue)');
xlim([-2 8]);
ylim([-2 8]);

save_all_figs_OPTION('results/gmm2','png',1)

% plot the number of iterations to converge a solution
figure
plot(L,'*-');
xlabel('# of iterations')
ylabel('Likelihood')
title('Likelihood vs. Iteration')


% missing data
%----------------------------------%
% reduce points
idx = sort(abs(floor(rand(700,1).*1000)),'ascend');
idx(idx>1000,:) = 1000;
X2 = X;
X2(idx,:)= NaN;
X2(any(isnan(X2),2),:) = [];

% re-calculate k-Means
[~, ctrs2] = kmeans(X2,nRand,'Distance','city','Replicates',5,'Options',opts);

% re-calculate E-M
[W2,M2,V2,L2] = EM_GM(X2,nRand,1e-5,500,1,[]);

% plot k-Means and E-M results
Plot_GM(W2,M2,V2,X2);
hold on;
for ii = 1:length(ctrs2)
    plot(ctrs2(ii,1),ctrs2(ii,2),'b*','LineWidth',2);
end
hold off;
title('Gaussian Mixture estimated by EM (black) & k-Means (blue)');
xlim([-2 8]);
ylim([-2 8]);

% plot the number of iterations to converge a solution
figure
plot(L2,'*-');
xlabel('# of iterations')
ylabel('Likelihood')
title('Likelihood vs. Iteration')


% save the results
%----------------------------------%
% save_all_figs_OPTION('EM','png')



% Subfunctions
% EM
%-------------------------------------------------------------------------%
function [W,M,V,L] = EM_GM(X,k,ltol,maxiter,fflag,Init)
% EM_GM EM algorithm for Gaussian mixture
%
%  Syntax:  [W,M,V,L] = EM_GM(X,k,ltol,maxiter,pflag,Init)
%
%  Description:
%   EM algorithm for k multidimensional Gaussian mixture estimation.
%
% Inputs:
%   X(n,d) - input data, n=number of observations, d=dimension of variable
%   k - maximum number of Gaussian components allowed
%   ltol - percentage of the log likelihood difference between 2 iterations ([] for none)
%   maxiter - maximum number of iteration allowed ([] for none)
%   fflag - 1 for fast calculation flag (uses Matlab vectorization), 0 otherwise ([] for none)
%   Init - structure of initial W, M, V: Init.W, Init.M, Init.V ([] for none)
%
% Ouputs:
%   W(1,k) - estimated weights of GM
%   M(d,k) - estimated mean vectors of GM
%   V(d,d,k) - estimated covariance matrices of GM
%   L - log likelihood of estimates
%
%  Example:
%     % generate random variables & mixtures
%     x1 = randn(500, 2)*3 + 10;
%     x2 = randn(500, 2)*2 + 1;
%     x3 = randn(500, 2)*3 + 3;
%     X = [x1; x2; x3]';
%
%     % calculation of E&M
%     [W,M,V,L] = EM_GM(X',3,1e-5,500,1,[]);

% Initialize W, M, V,L
t = cputime;
if isempty(Init),
    [W,M,V] = init_EM(X,k); L = 0;
else
    W = Init.W;
    M = Init.M;
    V = Init.V;
end

% Initialize log likelihood
Ln = Likelihood(X,k,W,M,V);
Lo = 2*Ln;

% EM algorithm (iterate until convergence)
niter = 0;
while (abs(100*(Ln-Lo)/Lo)>ltol) & (niter<=maxiter)
    
    % E-step
    % If we know the Gaussians, we can assign the points by relative
    % probability density of each Gaussian at each point.
    E = Expectation(X,k,W,M,V,fflag);
    
    % M-step
    % If we know the assignment, we can estimate the Gaussians by weighted
    % means of the points assigned to each of them.
    [W,M,V] = Maximization(X,k,E,fflag);
    Lo = Ln;
    
    % Likelihood
    % We use logarithms to avoid under flow and do the sum:
    Ln = Likelihood(X,k,W,M,V);
    niter = niter + 1;
    L(niter) = Ln;
end

% print results
fprintf('CPU time used for EM_GM: %5.2fs\n',cputime-t);
fprintf('Number of iterations: %d\n',niter-1);

end


% Init_EM
%-------------------------------------------------------------------------%
function [W,M,V] = init_EM(X,k)
%
%   X(n,d) - input data, n=number of observations, d=dimension of variable
%   k - maximum number of Gaussian components allowed
%   W(1,k) - estimated weights of GM
%   M(d,k) - estimated mean vectors of GM
%   V(d,d,k) - estimated covariance matrices of GM
%   Ci(nx1) - cluster indices
%   C(k,d) - cluster centroid (i.e. mean)
%
% Check the size of the data X
[n,d] = size(X);
% find the number of clusters present
[Ci,C] = kmeans(X,k,...
    'Start','cluster', ...
    'Maxiter',100, ...
    'EmptyAction','drop', ...
    'Display','off');
% Repeat only if NaNs (empty set) exist
while sum(isnan(C))>0,
    [Ci,C] = kmeans(X,k,...
        'Start','cluster', ...
        'Maxiter',100, ...
        'EmptyAction','drop', ...
        'Display','off');
end
M = C';
% initialize the Vp structure
Vp = repmat(struct('count',0,'X',zeros(n,d)),1,k);
% Separate cluster points
for i=1:n,
    % number of clusters
    Vp(Ci(i)).count = Vp(Ci(i)).count + 1;
    % data for the respective cluster
    Vp(Ci(i)).X(Vp(Ci(i)).count,:) = X(i,:);
end
V = zeros(d,d,k);
for i=1:k,
    % weight for the respective cluster
    W(i) = Vp(i).count/n;
    % estimated covariance matrices for the respective cluster
    V(:,:,i) = cov(Vp(i).X(1:Vp(i).count,:));
end

end


% Expectation
%-------------------------------------------------------------------------%
function E = Expectation(X,k,W,M,V,fflag)
%
%   X(n,d) - input data, n=number of observations, d=dimension of variable
%   k - maximum number of Gaussian components allowed
%   W(1,k) - estimated weights of GM
%   M(d,k) - estimated mean vectors of GM
%   V(d,d,k) - estimated covariance matrices of GM
%   E(n,k) - Expectation of GM
%   fflag - fast calculation flag (uses Matlab vectorization)
%
if fflag == 1 % fast flag true
    % new method with vectorization (fast!)
    [n,d] = size(X);
    E = zeros(n,k);
    for j = 1:k,
        % initialize V
        if V(:,:,j)==zeros(d,d)
            V(:,:,j)=ones(d,d)*eps;
        end
        % calc E using Multivariate normal probability density function (pdf).
        E(:,j) = W(j).*mvnpdf( X, M(:,j)', V(:,:,j) );
    end
    total = repmat(sum(E,2),1,j);
    E = E./total;
else
    % old method with for loops (too slow)
    % Check the size of the data X
    [n,d] = size(X);
    % Initialize the variables
    a = (2*pi)^(0.5*d);
    S = zeros(1,k);
    iV = zeros(d,d,k);
    E = zeros(n,k);
    % first loop for the estimated covariance matrices of GM
    for j=1:k
        % if empty, init to epsilon
        if V(:,:,j)==zeros(d,d)
            V(:,:,j)=ones(d,d)*eps;
        end
        % S = square-root of the determinant of V
        S(j) = sqrt(det(V(:,:,j)));
        % iV = matrix inverse of V
        iV(:,:,j) = inv(V(:,:,j));
    end
    % second loop for the Expectation of GM
    for i=1:n,
        for j=1:k,
            % transpose of X - est. mean vectors
            dXM = X(i,:)'-M(:,j);
            % exponent term for the gaussian exponential family
            pl = exp(-0.5*dXM'*iV(:,:,j)*dXM)/(a*S(j));
            % Expectation of estimated weights of the GM
            E(i,j) = W(j)*pl;
        end
        E(i,:) = E(i,:)/sum(E(i,:));
    end
    total = repmat(sum(E,2),1,j);
    E = E./total;
end

end


% Maximization
%-------------------------------------------------------------------------%
function [W,M,V] = Maximization(X,k,E,fflag)
%
%   X(n,d) - input data, n=number of observations, d=dimension of variable
%   k - maximum number of Gaussian components allowed
%   E(n,k) - Expectation of GM
%   W(1,k) - estimated weights of GM
%   M(d,k) - estimated mean vectors of GM
%   V(d,d,k) - estimated covariance matrices of GM
%   fflag - fast calculation flag (uses Matlab vectorization)
%
if fflag == 1 % fast flag true
    % new method with vectorization (fast!)
    [n,d] = size(X);
    W = sum(E);
    M = X'*E./repmat(W,d,1);
    for i=1:k,
        dXM = X - repmat(M(:,i)',n,1);
        % calc W as sparse matrix formed from diagonals.
        Wsp = spdiags(E(:,i),0,n,n);
        % re-calculate V
        V(:,:,i) = dXM'*Wsp*dXM/W(i);
    end
    W = W/n;
else
    % old method with for loops (too slow)
    % Check the size of the data X
    [n,d] = size(X);
    % Initialize the variables
    W = zeros(1,k);
    M = zeros(d,k);
    V = zeros(d,d,k);
    % first loop for the estimated mean vectors of GM
    for i=1:k,  % Compute weights
        for j=1:n,
            % Weights + Expectation
            W(i) = W(i) + E(j,i);
            % estimated mean vectors + Expectation times the data
            M(:,i) = M(:,i) + E(j,i)*X(j,:)';
        end
        % updated means div. by the weights
        M(:,i) = M(:,i)/W(i);
    end
    % second loop for the estimated covariance matrices of GM
    for i=1:k,
        for j=1:n,
            % transpose of X - est. mean vectors
            dXM = X(j,:)'-M(:,i);
            % estimated covariance matrices of GM
            V(:,:,i) = V(:,:,i) + E(j,i)*dXM*dXM';
        end
        V(:,:,i) = V(:,:,i)/W(i);
    end
    % updated estimated weights of GM
    W = W/n;
end

end


% Likelihood
%-------------------------------------------------------------------------%
function L = Likelihood(X,k,W,M,V)
% Compute L based on K. V. Mardia, "Multivariate Analysis", Academic Press,
% 1979, PP. 96-97 to enhance computational speed.
%
%   X(n,d) - input data, n=number of observations, d=dimension of variable
%   k - maximum number of Gaussian components allowed
%   W(1,k) - estimated weights of GM
%   M(d,k) - estimated mean vectors of GM
%   V(d,d,k) - estimated covariance matrices of GM
%
% Check the size of the data X
[n,d] = size(X);
U = mean(X)';
S = cov(X);
L = 0;
for i=1:k,
    % inverse of V
    iV = inv(V(:,:,i));
    % split into parts
    ll_1 = -0.5*(n)*log(det(2*pi*V(:,:,i)));
    ll_2 = -0.5*(n-1)*(trace(iV*S)+(U-M(:,i))'*iV*(U-M(:,i)));
    % calculate Likelihood
    L = L + W(i)*( ll_1 + ll_2 );
end

end


% Plot_GM V2
%-------------------------------------------------------------------------%
function Plot_GM(W,M,V,X)
% Plot_GM(W,M,V,X)
%
% Plots a 1D or 2D Gaussian mixture.
%
% Inputs:
%   W(1,k)   - weights of each GM component, k is the number of component
%   M(d,k)   - mean of each GM component, d is the dimension of the random variable
%   V(d,d,k) - covariance of each GM component
%   flag(1,1)- optional for 2D only,
%               'e' for standard deviation ellipse plot only (default),
%               'l' for landscape plot only using function MESHC,
%               'b' for both ellipse and landscape plots
%   X(n,d)   - optional for 2D only, data used to learn the GM
figure
[d,k] = size(M);
S = zeros(d,k);
R1 = zeros(d,k);
R2 = zeros(d,k);
for i=1:k,  % Determine plot range as 4 x standard deviations
    S(:,i) = sqrt(diag(V(:,:,i)));
    R1(:,i) = M(:,i)-4*S(:,i);
    R2(:,i) = M(:,i)+4*S(:,i);
end
Rmin = min(min(R1));
Rmax = max(max(R2));
R = [Rmin:0.001*(Rmax-Rmin):Rmax];

if d==1,
    clf, hold on
    Q = zeros(size(R));
    for i=1:k,
        P = W(i)*normpdf(R,M(:,i),sqrt(V(:,:,i)));
        Q = Q + P;
        plot(R,P,'r-'); grid on,
    end
    plot(R,Q,'k-');
    xlabel('X');
    ylabel('Probability density');
    title('Gaussian Mixture estimated by EM');
else % d==2
    Plot_Std_Ellipse(M,V,R,X);
end

end


% Plot_Std_Ellipse
%-------------------------------------------------------------------------%
function Plot_Std_Ellipse(M,V,R,X)
clf, hold on
if ~isempty(X),
    plot(X(:,1),X(:,2),'r.');
end
[d,k] = size(M);
for i=1:k,
    if V(:,:,i)==zeros(d,d),
        V(:,:,i) = eyes(d,d)*eps;
    end
    [Ev,D] = eig(V(:,:,i));
    iV = inv(V(:,:,i));
    % Find the larger projection
    P = [1,0;0,0];  % X-axis projection operator
    P1 = P * 2*sqrt(D(1,1)) * Ev(:,1);
    P2 = P * 2*sqrt(D(2,2)) * Ev(:,2);
    if abs(P1(1)) >= abs(P2(1)),
        Plen = P1(1);
    else
        Plen = P2(1);
    end
    count = 1;
    step = 0.001*Plen;
    Contour1 = zeros(2001,2);
    Contour2 = zeros(2001,2);
    for x = -Plen:step:Plen,
        a = iV(2,2);
        b = x * (iV(1,2)+iV(2,1));
        c = (x^2) * iV(1,1) - 1;
        Root1 = (-b + sqrt(b^2 - 4*a*c))/(2*a);
        Root2 = (-b - sqrt(b^2 - 4*a*c))/(2*a);
        if isreal(Root1),
            Contour1(count,:) = [x,Root1] + M(:,i)';
            Contour2(count,:) = [x,Root2] + M(:,i)';
            count = count + 1;
        end
    end
    Contour1 = Contour1(1:count-1,:);
    Contour2 = [Contour1(1,:);Contour2(1:count-1,:);Contour1(count-1,:)];
    plot(M(1,i),M(2,i),'k*','LineWidth',2);
    plot(Contour1(:,1),Contour1(:,2),'k-','LineWidth',2);
    plot(Contour2(:,1),Contour2(:,2),'k-','LineWidth',2);
end
xlabel('1^{st} dimension');
ylabel('2^{nd} dimension');
title('Gaussian Mixture estimated by EM');
Rmin = R(1);
Rmax = R(length(R));
axis([Rmin Rmax Rmin Rmax])

end





