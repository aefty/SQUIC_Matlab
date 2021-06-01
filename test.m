p=1000;
n=100;
lambda=.3;
verbose = 0;
Y = randn(p,n);
[S,info_times_S]=SQUIC_S(Y, lambda, verbose);

[X,W,info_times,info_objective,info_logdetX,info_trSX]=SQUIC(Y, lambda);