%Make sure in the DFC directory before attempting to run

data = readtable('ethprices.csv');

ETH_USD = data(:,2);

returns = tick2ret(ETH_USD.ETH);

most_recent_price = ETH_USD.ETH(end)

nVariables  = 2;
expReturn   = [mean(returns), mean(returns)];
sigma       = [std(returns), std(returns)/2] ;
correlation_A = [1 0.9; 0.9 1];
correlation_B = [1 0.1; 0.1 1];
correlation_C = [1 -0.9; -0.9 1];
t           = 0;
X           = most_recent_price;
X           = X(ones(nVariables,1));

nPeriods = 100;      % # of simulated observations
dt       =   1;      % time increment = 1 day
rng(142857,'twister')

% [S,T] = simulate(GBM, nPeriods, 'DeltaTime', dt, 'nTrials', 1000);

GBM = gbm(diag(expReturn), diag(sigma), 'Correlation', correlation_A, 'StartState', X)
[A,T] = simBySolution(GBM, nPeriods, 'DeltaTime', dt, 'nTrials', 10000);

GBM = gbm(diag(expReturn), diag(sigma), 'Correlation', correlation_B, 'StartState', X)
[B,T] = simBySolution(GBM, nPeriods, 'DeltaTime', dt, 'nTrials', 10000);

GBM = gbm(diag(expReturn), diag(sigma), 'Correlation', correlation_C, 'StartState', X)
[C,T] = simBySolution(GBM, nPeriods, 'DeltaTime', dt, 'nTrials', 10000);


% histogram(squeeze(X(end,2,:)), 30), xlabel('Price'), ylabel('Frequency')
% title('Histogram of Prices after 100 days')

% subplot(2,1,1)
% plot(T, S(:,:,554)), xlabel('Trading Day'),ylabel('Price')
% title('1st Path of Multi-Dim Market Model:Euler Approximation')
% subplot(2,1,2)

plot(T, A(:,:)), xlabel('Day'),ylabel('Price')
title('1st Path of Multi-Dim Market Model:Analytic Solution')