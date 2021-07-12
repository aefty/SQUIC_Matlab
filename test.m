p=1024;
n=100;
lambda=.4;
Y = randn(p,n);

% to compute sample covariance matrix
[S,info_times_S]=SQUIC_S(Y, lambda);


% to compute sparse sampel covariance matrix X, and its inverse W, etc.
[X,W,info_times,info_objective,info_logdetX,info_trSX]=SQUIC(Y, lambda);

